// src/program/qwen35_serve.cpp - the official server line protocol for the Qwen35 path (M2).
//
// `strata generate --qwen35-model <gguf> --qwen35-pack <dir> --serve` stops being a one-shot runner and becomes
// a resident engine for the official server (`serve/server.py`): it loads the model once, prints
// `READY <ctx> stop`, then serves any number of requests from stdin.  The line protocol is the official
// engine's own (parsed by serve/server.py's StrataEngine):
//
//   client -> engine:
//     GEN <max_new> [key=value ...] <id,id,...>   one request, sampled greedy unless temperature>0
//     SCORE <id,id,...> [topk=N]                  read-only candidate logits: prefill, generate nothing
//     CAPS                                        print the capability table (one JSON line), keep serving
//     STOP                                        cancel the running request; the process keeps serving
//     QUIT                                        exit 0
//   engine -> client:
//     PP <done> <total> <ms> <fresh tok/s>        once per prefill chunk (ubatch-sized)
//     T <id>                                      one generated token, EOS included
//     LOGPROBS <pos> <k> <id>:<logprob> ...       SCORE's answer: top-k of the FULL-vocabulary log-softmax
//     DONE <produced> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <reused>
//          <hits> <lookups> <ram_blobs> <file_blobs> <file_mb> <prompt_read> <offloaded>
//     CAPS <json>                                 the --capabilities JSON (P2-1)
//     ERR <message>                               request-level refusal; the engine keeps serving
//
// SCORE (P2-1) reads a prompt through the SAME prefill path as GEN (ubatch chunks, PP lines) and then takes
// the last position's logits: it emits no token.  It answers `LOGPROBS <pos> <k> <id>:<logprob> ...` where
// pos is the 0-based last prompt position and logprob is the log-softmax over EVERY logit of the vocabulary
// (no top-k normalization), followed by `DONE ... score` with produced=0.  State semantics are (b)
// independent, by construction: SCORE starts from `llama_memory_clear` like GEN and resets the session again
// after printing, so the next request - GEN or SCORE - always starts fresh (reuse stays 0).
// CAPS answers the same JSON as `--capabilities` (see qwen35_capabilities_json and
// docs/backends/qwen35_llama_capabilities_and_score.md).
//
// Sampling keys implemented: temperature / top_p / top_k / min_p / seed.  A non-default `penalty_*` (or a
// `logit_bias`) is REFUSED with ERR instead of being ignored - the phase-1 engine has no such sampler, and a
// silently different answer would be worse than a refusal.  Keys of the native engine's conversation cache
// (ckpt / pin / cvec / pcie_frac / spec_min_p / aux_cpus / img_keys) are skipped like the official engine
// skips unknown keys: they tune caching/placement this phase does not have (no cross-request prefix reuse,
// reuse is reported as 0).
//
// Nothing of the native engine's --serve is reused: STRATA_QWEN35's MTP draft/verify family and the
// two-request diagnostic are refused before the model load (fail closed), and every request starts from
// `llama_memory_clear(..., true)` (KV cells + GDN conv/SSM/rollback planes), so no context leaks between
// requests - including after a STOP mid-prefill or mid-decode.  `reuse` stays 0: no prefix is carried over.
//
// Scope of phase 1 (deliberately unimplemented, all documented in docs/backends/qwen35_llama_serve_protocol.md):
// penalties/logit_bias, prefix reuse, continuous batching (INFO batch_slots is never printed), vision (GENI),
// disk sessions (SAVE/RESTORE), MTP.  Unknown directives answer `ERR ...` and the loop continues - it never
// exits or crashes on input it does not know.
#include "strata/program/qwen35_model.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/qwen35_bridge.hpp"
#include "strata/core/qwen35_mtp.hpp"
#include "strata/core/qwen35_mtp_txn.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <io.h>
#else
#  include <unistd.h>
#endif

namespace strata::program {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// NaN is a real failure here too (the CLI path rejects it the same way): the request ends with
/// DONE ... nonfinite and the token that would have been sampled is not emitted.
bool logits_have_nan(const float* logits, int64_t n) {
    for (int64_t i = 0; i < n; ++i)
        if (std::isnan(logits[i])) return true;
    return false;
}

/// One prompt chunk / one decode step; output logits are requested for the last row only (the context runs
/// with n_outputs_max = 1).  Same shape as the runner's own fill_batch.
bool fill_batch(llama_batch_ext* batch, const llama_token* tokens, int count, int position) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        if (idx < 0) return false;
        const llama_pos pos = position + i;
        if (!llama_batch_ext_set_pos(batch, idx, &pos)) return false;
    }
    return llama_batch_ext_set_output_logits(batch, count - 1, true);
}

/// One parsed GEN request.  Defaults mirror the official engine's: greedy, no filters, seed unset.
struct GenRequest {
    int64_t max_new = 0;
    std::vector<llama_token> ids;
    float temperature = 0.0f;   ///< <= 0: greedy (the official "temperature absent" default)
    float top_p = 1.0f;
    int32_t top_k = 0;          ///< 0 = off
    float min_p = 0.0f;
    uint64_t seed = 0;          ///< 0 = the request asked for none (a clock seed is used, like the official engine)
};

bool parse_scalar(const std::string& val, char*& end) {
    return !val.empty() && end != nullptr && *end == '\0';
}

/// `<id,id,...>` (no spaces inside; see the callers) appended to `out`.  Every part must be a decimal id in
/// [0, llama_token max].  Returns false with `err` set.  GEN and SCORE share this so both refuse id lists the
/// same way.
bool parse_id_list(const char* ids, const char* ids_end, std::vector<llama_token>& out, std::string& err) {
    if (ids_end == ids) { err = "bad request: no token ids"; return false; }
    for (const char* q = ids; q != ids_end; ++q) {
        if (*q == ' ') { err = "bad request: token ids"; return false; }
    }
    const std::string ids_txt(ids, (size_t) (ids_end - ids));
    const int64_t n_vocab_max = std::numeric_limits<llama_token>::max();
    for (size_t q = 0;;) {
        const char* at = ids_txt.c_str() + q;
        char* ie = nullptr;
        const long long v = std::strtoll(at, &ie, 10);
        if (ie == at || v < 0 || v > n_vocab_max) { err = "bad request: token ids"; return false; }
        out.push_back((llama_token) v);
        const size_t adv = (size_t) (ie - ids_txt.c_str());
        if (adv == ids_txt.size()) break;
        if (ids_txt[adv] != ',' || adv + 1 == ids_txt.size()) { err = "bad request: token ids"; return false; }
        q = adv + 1;
    }
    return true;
}

/// `GEN <max_new> [key=value ...] <id,id,...>`; the ids start at the first whitespace token without '=' and run
/// to the end of the line (same split rule as the official engine's parser).  Returns false with `err` set.
bool parse_gen_line(const std::string& line, GenRequest& req, std::string& err) {
    const char* p = line.c_str() + 4;   // past "GEN "
    while (*p == ' ') ++p;
    if (*p < '0' || *p > '9') { err = "bad request: max_new"; return false; }
    char* end = nullptr;
    const long long max_new = std::strtoll(p, &end, 10);
    if (max_new < 1) { err = "bad request: max_new"; return false; }
    req.max_new = max_new;
    p = end;
    for (;;) {
        while (*p == ' ') ++p;
        if (*p == '\0') break;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        const std::string tok(start, (size_t) (p - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) { p = start; break; }   // the token list starts here
        const std::string key = tok.substr(0, eq);
        const std::string val = tok.substr(eq + 1);
        char* ve = nullptr;
        if (key == "temperature") {
            const float v = std::strtof(val.c_str(), &ve);
            if (!parse_scalar(val, ve) || !std::isfinite(v) || v < 0.0f) { err = "bad request: temperature"; return false; }
            req.temperature = v;
        } else if (key == "top_p") {
            const float v = std::strtof(val.c_str(), &ve);
            if (!parse_scalar(val, ve) || !std::isfinite(v) || !(v > 0.0f) || v > 1.0f) { err = "bad request: top_p"; return false; }
            req.top_p = v;
        } else if (key == "top_k") {
            const long long v = std::strtoll(val.c_str(), &ve, 10);
            if (!parse_scalar(val, ve) || v < 0 || v > std::numeric_limits<int32_t>::max()) { err = "bad request: top_k"; return false; }
            req.top_k = (int32_t) v;
        } else if (key == "min_p") {
            const float v = std::strtof(val.c_str(), &ve);
            if (!parse_scalar(val, ve) || !std::isfinite(v) || v < 0.0f || v > 1.0f) { err = "bad request: min_p"; return false; }
            req.min_p = v;
        } else if (key == "seed") {
            const unsigned long long v = std::strtoull(val.c_str(), &ve, 10);
            if (!parse_scalar(val, ve)) { err = "bad request: seed"; return false; }
            req.seed = v;
        } else if (key == "penalty_last_n" || key == "penalty_repeat" || key == "penalty_freq" ||
                   key == "penalty_present") {
            // Non-default penalties are refused, never ignored: this engine has no penalty sampler.
            const double v = std::strtod(val.c_str(), &ve);
            if (!parse_scalar(val, ve)) { err = "bad request: " + key; return false; }
            const double def = key == "penalty_repeat" ? 1.0 : 0.0;
            if (v != def) { err = key + " is not supported by this engine"; return false; }
        } else if (key == "logit_bias") {
            err = "logit_bias is not supported by this engine";
            return false;
        }
        // Any other key is skipped exactly like the official engine skips unknown keys (the conversation
        // cache's ckpt/pin/cvec/pcie_frac/spec_min_p/aux_cpus and the image keys implicates).
    }
    const char* ids = p;
    const char* ids_end = ids + std::strlen(ids);
    while (ids_end > ids && ids_end[-1] == ' ') --ids_end;   // a trailing space is noise, not an id list
    return parse_id_list(ids, ids_end, req.ids, err);
}

/// One parsed SCORE request (P2-1): a prompt and the number of candidate logits to report (default 5).
struct ScoreRequest {
    std::vector<llama_token> ids;
    int32_t topk = 5;
};

/// `SCORE <id,id,...> [topk=N]`: the only non-key token is the id list (it may come before or after the keys).
/// `topk` is the only key; any other key is refused - a silently ignored `topk` typo would answer a different
/// question than the one asked.  `topk` must be a positive integer (the run clamps it to the vocabulary size).
bool parse_score_line(const std::string& line, ScoreRequest& req, std::string& err) {
    const char* p = line.c_str() + 5;   // past "SCORE"
    const char* ids = nullptr;
    const char* ids_end = nullptr;
    for (;;) {
        while (*p == ' ') ++p;
        if (*p == '\0') break;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        const std::string tok(start, (size_t) (p - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            if (ids != nullptr) { err = "bad request: token ids"; return false; }   // one id list, not two
            ids = start;
            ids_end = p;
            continue;
        }
        const std::string key = tok.substr(0, eq);
        const std::string val = tok.substr(eq + 1);
        if (key != "topk") { err = "bad request: unknown key " + key; return false; }
        char* ve = nullptr;
        const long long v = std::strtoll(val.c_str(), &ve, 10);
        if (!parse_scalar(val, ve) || v < 1 || v > std::numeric_limits<int32_t>::max()) {
            err = "bad request: topk";
            return false;
        }
        req.topk = (int32_t) v;
    }
    if (ids == nullptr) { err = "bad request: no token ids"; return false; }
    return parse_id_list(ids, ids_end, req.ids, err);
}

/// temperature > 0: the standard chain top_k -> top_p -> min_p -> temp -> dist(seed).  temperature <= 0
/// (absent included): greedy, like the official engine.
llama_sampler* make_sampler(const GenRequest& req) {
    if (!(req.temperature > 0.0f)) return llama_sampler_init_greedy();
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (chain == nullptr) return nullptr;
    if (req.top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(req.top_k));
    if (req.top_p < 1.0f) llama_sampler_chain_add(chain, llama_sampler_init_top_p(req.top_p, 1));
    if (req.min_p > 0.0f) llama_sampler_chain_add(chain, llama_sampler_init_min_p(req.min_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(req.temperature));
    const uint64_t seed = req.seed != 0 ? req.seed
                                        : (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    llama_sampler_chain_add(chain, llama_sampler_init_dist((uint32_t) seed));
    return chain;
}

bool env_is_exactly_one(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

int64_t meta_int(const llama_model* model, const char* key, int64_t fallback) {
    char buf[64] = {0};
    if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) <= 0 || buf[0] == '\0') return fallback;
    return std::strtoll(buf, nullptr, 10);
}

/// stdin is read on its own thread so a STOP reaches a request that is still running (the consumer sent
/// itself away); everything here lives behind a shared_ptr because the reader is detached and may still sit
/// in read(2) when the engine exits.
struct StdinQueue {
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
};

/// The prompt read GEN and SCORE share (P2-1): ubatch-sized chunks with one PP line after each, then the
/// prompt -> decode transition.  `prompt_ms` is measured after the transition, like phase 1.  A false
/// `loan_returned` is a real failure; `cancelled` means a STOP ended the read part-way.
struct PrefillRun {
    int64_t read_n = 0;
    bool failed = false;
    bool cancelled = false;
    bool loan_returned = false;
    double prompt_ms = 0.0;
};

PrefillRun run_prefill(llama_context* ctx, llama_batch_ext* batch, const std::vector<llama_token>& ids,
                       int64_t chunk_max, const std::shared_ptr<StdinQueue>& stdin_shared) {
    PrefillRun r;
    const auto prefill_begin = Clock::now();
    const int64_t total = (int64_t) ids.size();
    for (int64_t at = 0; at < total; at += chunk_max) {
        if (stdin_shared->stop.load()) { r.cancelled = true; break; }
        const int count = (int) std::min<int64_t>(chunk_max, total - at);
        if (!fill_batch(batch, ids.data() + at, count, (int) at)) { r.failed = true; break; }
        const double chunk_begin_ms = ms_since(prefill_begin);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            r.failed = true;
            break;
        }
        r.read_n = at + count;
        const double now_ms = ms_since(prefill_begin);
        const double chunk_ms = now_ms - chunk_begin_ms;
        std::printf("PP %lld %lld %.0f %.1f\n", (long long) r.read_n, (long long) total, now_ms,
                    chunk_ms > 0.0 ? 1000.0 * (double) count / chunk_ms : 0.0);
        std::fflush(stdout);
    }
    r.loan_returned = core::qwen35_bridge_finish_prompt();
    r.prompt_ms = ms_since(prefill_begin);
    return r;
}

/// The state reset every request starts with (GEN and SCORE): cell metadata + KV cells and the recurrent
/// (GDN conv/SSM/rollback) planes - the proven reset the two-request diagnostic trusts.
void reset_session(llama_context* ctx) {
    if (llama_memory_t mem = llama_get_memory(ctx)) {
        llama_memory_clear(mem, true);
    }
}

}  // namespace

std::string qwen35_capabilities_json(const Qwen35RunOptions& o, int64_t context_max, int64_t nextn_layers) {
    // mtp.available: the model's declared MTP (NextN) layer count - qwen35moe.nextn_predict_layers, the same
    // key the llama loader reads into hparams.n_layer_nextn.  With a loaded model the caller passes the value;
    // otherwise only the GGUF metadata header is read (never a tensor payload), and an open/parse error leaves
    // availability false.  The production UD-IQ3_S file has no such key (false); the MTP experiment has 1.
    bool mtp_available = false;
    if (nextn_layers >= 0) {
        mtp_available = nextn_layers > 0;
    } else if (!o.model.empty()) {
        try {
            const strata::GgufModel m({o.model});
            const strata::MetaValue* v = m.meta().get("qwen35moe.nextn_predict_layers");
            if (v != nullptr && v->is_num()) mtp_available = (int64_t) v->num() > 0;
        } catch (...) {
            // Documented in the capabilities doc: unreadable metadata leaves mtp.available false.
        }
    }
    // expert_formats: the compile-time grouped-expert kernel whitelist (the GPU native-expert path; see
    // kernels::iq_expert_role_format / iq_kernels.cu's STRATA_GU_FMTS and STRATA_D_FMTS).  A pack whose layer
    // pair fails kernels::native_expert_supported is refused at startup, so these are the formats a model can
    // actually carry.  The array is the union of the two roles; the roles' own split is in the doc.
    static const struct { int id; const char* name; } kExpertFormats[] = {
        {2, "Q4_0"}, {3, "Q4_1"}, {6, "Q5_0"}, {7, "Q5_1"}, {8, "Q8_0"}, {12, "Q4_K"}, {13, "Q5_K"},
        {14, "Q6_K"}, {16, "IQ2_XXS"}, {17, "IQ2_XS"}, {18, "IQ3_XXS"}, {20, "IQ4_NL"}, {21, "IQ3_S"},
        {22, "IQ2_S"}, {23, "IQ4_XS"}, {29, "IQ1_M"}, {42, "Q2_0"},
    };
    std::string fmts;
    for (const auto& f : kExpertFormats) {
        if (!strata::kernels::iq_expert_role_format(f.id)) continue;
        if (!fmts.empty()) fmts += ',';
        fmts += '"';
        fmts += f.name;
        fmts += '"';
    }
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "{\"protocol\":\"strata-serve-v1\",\"continuous_requests\":true,\"cancel\":true,"
                  "\"state_isolation\":true,\"prefix_cache\":false,\"batching\":false,"
                  "\"mtp\":{\"available\":%s,\"enabled\":%s,\"default\":false,\"k\":null},"
                  "\"expert_formats\":[%s],\"context_max\":%lld,\"logits_read\":true,"
                  "\"state_export\":false,\"disk_kv\":false,\"vision\":false}",
                  mtp_available ? "true" : "false", env_is_exactly_one("STRATA_QWEN35_MTP") ? "true" : "false",
                  fmts.c_str(), (long long) context_max);
    return std::string(buf);
}

int qwen35_serve(const Qwen35RunOptions& o) {
    if (o.model.empty() || o.pack_dir.empty()) {
        std::fprintf(stderr, "strata qwen35 serve: needs --qwen35-model and --qwen35-pack\n");
        return 2;
    }
    // Phase-1 scope guards: a set gate whose behavior the serve path does not implement is refused before the
    // model load, never silently ignored.  All default OFF; with none set this is inert.
    if (core::qwen35_mtp_enabled() || core::qwen35_mtp_selftest_requested() || core::qwen35_mtp_draft_enabled() ||
        core::qwen35_mtp_txn_enabled() ||
        env_is_exactly_one("STRATA_QWEN35_MTP_T3_VERIFY_ORACLE") ||
        env_is_exactly_one("STRATA_QWEN35_MTP_T3_ATTN_SLICE") ||
        env_is_exactly_one("STRATA_QWEN35_MTP_T3_FA_INPUTS")) {
        std::fprintf(stderr, "strata qwen35 serve: the MTP draft/verify path is not implemented in serve mode; "
                             "unset the STRATA_QWEN35_MTP* gates\n");
        return 2;
    }
    if (env_is_exactly_one("STRATA_QWEN35_TWO_REQUEST")) {
        std::fprintf(stderr, "strata qwen35 serve: the two-request measurement (STRATA_QWEN35_TWO_REQUEST=1) is a "
                             "CLI-mode diagnostic; serve has no request-count limit and refuses the gate\n");
        return 2;
    }
    if (const char* replay = std::getenv("STRATA_QWEN35_LAYER_REPLAY"); replay != nullptr && replay[0] != '\0') {
        std::fprintf(stderr, "strata qwen35 serve: the layer-replay diagnostic is a CLI mode; refusing\n");
        return 2;
    }

    llama_backend_init();
    {
        // Same logger policy as the runner: everything but DEBUG goes to stderr, and stderr is not the protocol.
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level == GGML_LOG_LEVEL_DEBUG) return;
            std::fputs(text, stderr);
        }, nullptr);
    }
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr && !o.cuda_backend.empty()) {
        if (ggml_backend_load(o.cuda_backend.c_str()) == nullptr) {
            std::fprintf(stderr, "strata qwen35 serve: cannot load the CUDA backend %s\n", o.cuda_backend.c_str());
            return 1;
        }
    }
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr) {
        std::fprintf(stderr, "strata qwen35 serve: no GPU backend is registered (this build has no GGML_CUDA; "
                             "pass --qwen35-cuda-backend <ggml-cuda.dll>)\n");
        return 1;
    }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(nullptr, llama_model_free);
    {
        // The same model parameters the CLI runner uses, with MTP off by scope (the gate is refused above).
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = (int32_t) o.ngl;
        mp.load_mtp = false;
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_name("CPU");
        ggml_backend_buffer_type_t cpu_buft = cpu_dev != nullptr ? ggml_backend_dev_buffer_type(cpu_dev)
                                                                 : ggml_backend_cpu_buffer_type();
        const llama_model_tensor_buft_override overrides[] = {
            {".*ffn_.*_exps.*", cpu_buft},
            {nullptr, nullptr},
        };
        mp.tensor_buft_overrides = overrides;
        model.reset(llama_model_load_from_file(o.model.c_str(), mp));
    }
    if (model == nullptr) {
        std::fprintf(stderr, "strata qwen35 serve: cannot load %s\n", o.model.c_str());
        return 1;
    }
    {
        char arch[128] = {0};
        if (llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch)) <= 0 ||
            std::strcmp(arch, "qwen35moe") != 0) {
            std::fprintf(stderr, "strata qwen35 serve: %s is '%s', not a qwen35moe model\n", o.model.c_str(), arch);
            return 2;
        }
    }
    const llama_vocab* vocab = llama_model_get_vocab(model.get());
    const int64_t n_layers = llama_model_n_layer(model.get());
    const int64_t n_embd = llama_model_n_embd(model.get());
    const int64_t n_ff = meta_int(model.get(), "qwen35moe.expert_feed_forward_length", 0);
    const int64_t n_expert = meta_int(model.get(), "qwen35moe.expert_count", 0);
    const int64_t n_used = meta_int(model.get(), "qwen35moe.expert_used_count", 0);
    if (n_layers <= 0 || n_embd <= 0 || n_ff <= 0 || n_expert <= 0 || n_used <= 0) {
        std::fprintf(stderr, "strata qwen35 serve: the model does not carry the qwen35moe MoE metadata (layers %lld, "
                             "embd %lld, ff %lld, experts %lld, used %lld)\n", (long long) n_layers,
                     (long long) n_embd, (long long) n_ff, (long long) n_expert, (long long) n_used);
        return 2;
    }
    core::Qwen35BridgeConfig cfg;
    cfg.model = o.model;
    cfg.pack_dir = o.pack_dir;
    cfg.n_layers = n_layers;
    cfg.n_expert = n_expert;
    cfg.n_embd = n_embd;
    cfg.n_ff = n_ff;
    cfg.k = n_used;
    cfg.cache_slots = o.cache_slots;
    cfg.profile = o.profile;
    cfg.routing_log = o.routing_log;
    cfg.pool_workers = o.pool_workers;
    std::string err;
    if (!core::qwen35_bridge_init(cfg, err)) {
        std::fprintf(stderr, "strata qwen35 serve: %s\n", err.c_str());
        return 1;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) o.max_context;
    cp.n_batch = (uint32_t) std::max<int64_t>(512, (int64_t) o.ubatch);
    cp.n_ubatch = (uint32_t) std::max<int64_t>(1, (int64_t) o.ubatch);
    cp.n_threads = (int32_t) o.threads;
    cp.n_threads_batch = (int32_t) o.threads;
    cp.op_offload = true;
    cp.n_seq_max = 1;
    cp.n_outputs_max = 1;   // fill_batch flags exactly one output row per process call
    cp.no_perf = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (ctx == nullptr) {
        std::fprintf(stderr, "strata qwen35 serve: context init failed\n");
        core::qwen35_bridge_shutdown();
        return 1;
    }
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()),
                                                                             llama_batch_ext_free);
    if (batch == nullptr) {
        std::fprintf(stderr, "strata qwen35 serve: batch init failed\n");
        core::qwen35_bridge_shutdown();
        return 1;
    }
    // Past graph reservation: decode-graph nodes are the external producer from here on (the same seam the
    // CLI runner arms, with the same escape hatch).
    if (std::getenv("STRATA_QWEN35_NO_SEAM") == nullptr) core::qwen35_arm_decode_seam(true);
    // P2-1: the capability table this process answers for CAPS - the loaded model's actual context and MTP
    // layer count, not the argv defaults (the CLI --capabilities path computes it without a load).
    const std::string caps_json = qwen35_capabilities_json(
        o, (int64_t) llama_n_ctx(ctx.get()), meta_int(model.get(), "qwen35moe.nextn_predict_layers", 0));
    // READY advertises the context requests are actually checked against.
    std::printf("READY %lld stop\n", (long long) llama_n_ctx(ctx.get()));
    std::fflush(stdout);

    const auto stdin_shared = std::make_shared<StdinQueue>();
    std::thread([st = stdin_shared] {
        // read(2), not std::cin: glibc's exit() flushes every stdio stream and waits for stdin's lock, which
        // getline holds while it waits for input - an engine ending on QUIT would hang in exit().
        std::string l, buf;
        char chunk[4096];
        auto getline_fd = [&](std::string& out) -> bool {
            for (;;) {
                const size_t nlpos = buf.find('\n');
                if (nlpos != std::string::npos) {
                    out.assign(buf, 0, nlpos);
                    buf.erase(0, nlpos + 1);
                    return true;
                }
#if defined(_WIN32)
                const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                const ssize_t n = ::read(0, chunk, sizeof chunk);
                if (n < 0 && errno == EINTR) continue;
#endif
                if (n <= 0) {
                    if (buf.empty()) return false;
                    out.swap(buf);
                    buf.clear();
                    return true;
                }
                buf.append(chunk, (size_t) n);
            }
        };
        while (getline_fd(l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { st->stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(st->mu);
            st->lines.push_back(std::move(l));
            st->cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(st->mu);
        st->eof = true;
        st->cv.notify_one();
    }).detach();

    const int64_t total_ctx = (int64_t) llama_n_ctx(ctx.get());
    const int64_t n_vocab = llama_vocab_n_tokens(vocab);
    // The ACTUAL micro-batch the context reports (never a re-print of the request): PP chunks track it.
    const int64_t chunk_max = std::max<int64_t>(1, (int64_t) llama_n_ubatch(ctx.get()));
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(nullptr, llama_sampler_free);
    std::vector<int32_t> score_idx;   // SCORE's top-k scratch, reused across requests
    std::string line;
    long long serve_reqs = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(stdin_shared->mu);
            stdin_shared->cv.wait(lk, [&] { return !stdin_shared->lines.empty() || stdin_shared->eof; });
            if (stdin_shared->lines.empty()) break;   // stdin EOF: exit 0
            line = std::move(stdin_shared->lines.front());
            stdin_shared->lines.pop_front();
        }
        if (line.empty()) continue;   // empty lines are ignored
        if (line == "QUIT") break;
        // A literal STOP here is an idle one (the reader consumes STOP lines itself).
        if (line == "STOP") continue;
        if (line == "CAPS") {   // P2-1: the capability table, the same JSON --capabilities prints
            std::printf("CAPS %s\n", caps_json.c_str());
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("SCORE ", 0) == 0) {
            // P2-1: read-only candidate logits.  Same prefill as GEN, no token generated, and the session is
            // reset again after the answer (semantics (b): the next request always starts fresh).
            ScoreRequest sreq;
            std::string serr;
            if (!parse_score_line(line, sreq, serr)) {
                std::printf("ERR %s\n", serr.c_str());
                std::fflush(stdout);
                continue;
            }
            bool bad_id = false;
            for (size_t i = 0; i < sreq.ids.size() && !bad_id; ++i) {
                if (sreq.ids[i] < 0 || (int64_t) sreq.ids[i] >= n_vocab) bad_id = true;
            }
            if (bad_id) {
                std::printf("ERR a token id is outside the vocabulary\n");
                std::fflush(stdout);
                continue;
            }
            if ((int64_t) sreq.ids.size() > total_ctx) {
                std::printf("ERR prompt (%lld tokens) exceeds the context (%lld)\n",
                            (long long) sreq.ids.size(), (long long) total_ctx);
                std::fflush(stdout);
                continue;
            }
            stdin_shared->stop.store(false);   // a STOP between requests is stale (the GEN rule)
            reset_session(ctx.get());
            const int64_t total = (int64_t) sreq.ids.size();
            const PrefillRun pre = run_prefill(ctx.get(), batch.get(), sreq.ids, chunk_max, stdin_shared);
            if (pre.failed) {
                std::printf("ERR prefill failed\n");
                std::fflush(stdout);
                continue;
            }
            if (!pre.loan_returned) {
                std::printf("ERR the prompt workspace loan was not returned\n");
                std::fflush(stdout);
                continue;
            }
            if (pre.cancelled) {
                // A STOP ended the read part-way: there is no complete last-position row to score.
                std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld\n",
                            0LL, (long long) total, pre.prompt_ms, 0.0, "cancel", 0LL, 0LL, 0LL, 0LL, 0LL, 0LL,
                            0LL, 0.0, (long long) pre.read_n, 0LL);
                std::fflush(stdout);
                std::fprintf(stderr, "serve: req %lld finish=cancel produced=0 prompt_ms=%.1f decode_ms=0.0 score\n",
                             ++serve_reqs, pre.prompt_ms);
                reset_session(ctx.get());
                continue;
            }
            const float* logits = llama_get_logits_ith(ctx.get(), -1);
            if (logits == nullptr) {
                std::printf("ERR score produced no logits\n");
                std::fflush(stdout);
                continue;
            }
            if (logits_have_nan(logits, n_vocab)) {
                std::fprintf(stderr, "strata qwen35 serve: the head logits are not all finite; answering ERR "
                                     "(score)\n");
                std::printf("ERR score: the head logits are not all finite\n");
                std::fflush(stdout);
                reset_session(ctx.get());
                continue;
            }
            // log_softmax over the FULL vocabulary: logprob = logit - (max + log(sum exp(logit - max))).  The
            // denominator is all n_vocab logits; the top-k below selects from it, it does not renormalize.
            float mx = logits[0];
            for (int64_t i = 1; i < n_vocab; ++i) {
                if (logits[i] > mx) mx = logits[i];
            }
            double sum = 0.0;
            for (int64_t i = 0; i < n_vocab; ++i) sum += std::exp((double) logits[i] - (double) mx);
            const double lse = (double) mx + std::log(sum);
            const int64_t k = std::min<int64_t>(sreq.topk, n_vocab);
            // Top-k by (logit desc, id asc): an exact tie answers the same order every run.
            score_idx.resize((size_t) n_vocab);
            std::iota(score_idx.begin(), score_idx.end(), 0);
            std::partial_sort(score_idx.begin(), score_idx.begin() + (size_t) k, score_idx.end(),
                              [logits](int32_t a, int32_t b) {
                                  return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
                              });
            std::printf("LOGPROBS %lld %lld", (long long) (total - 1), (long long) k);
            for (int64_t j = 0; j < k; ++j) {
                const int32_t id = score_idx[(size_t) j];
                std::printf(" %d:%.4f", (int) id, (double) logits[id] - lse);
            }
            std::printf("\n");
            std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld\n",
                        0LL, (long long) total, pre.prompt_ms, 0.0, "score", 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL,
                        0.0, (long long) pre.read_n, 0LL);
            std::fflush(stdout);
            std::fprintf(stderr, "serve: req %lld finish=score produced=0 prompt_ms=%.1f decode_ms=0.0 topk=%lld\n",
                         ++serve_reqs, pre.prompt_ms, (long long) k);
            reset_session(ctx.get());
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: CAPS | SCORE <id,id,...> [topk=N] | GEN <max_new> [key=value ...] "
                        "<id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        GenRequest req;
        std::string perr;
        if (!parse_gen_line(line, req, perr)) {
            std::printf("ERR %s\n", perr.c_str());
            std::fflush(stdout);
            continue;
        }
        bool bad_id = false;
        for (size_t i = 0; i < req.ids.size() && !bad_id; ++i) {
            if (req.ids[i] < 0 || (int64_t) req.ids[i] >= n_vocab) bad_id = true;
        }
        if (bad_id) {
            std::printf("ERR a token id is outside the vocabulary\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) req.ids.size() + req.max_new > total_ctx) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n",
                        (long long) req.ids.size(), (long long) req.max_new, (long long) total_ctx);
            std::fflush(stdout);
            continue;
        }
        sampler.reset(make_sampler(req));
        if (sampler == nullptr) {
            std::printf("ERR sampler init failed\n");
            std::fflush(stdout);
            continue;
        }
        // A STOP that arrived between requests is stale (the official engine's rule).
        stdin_shared->stop.store(false);
        // Fresh request state (KV cells + the recurrent GDN planes), the same reset SCORE and the two-request
        // diagnostic use.
        reset_session(ctx.get());
        // Prefill, one ubatch-sized chunk at a time; PP after every chunk (the server's progress lines).
        const int64_t total = (int64_t) req.ids.size();
        const PrefillRun pre = run_prefill(ctx.get(), batch.get(), req.ids, chunk_max, stdin_shared);
        if (pre.failed) {
            std::printf("ERR prefill failed\n");
            std::fflush(stdout);
            continue;
        }
        if (!pre.loan_returned) {
            std::printf("ERR the prompt workspace loan was not returned\n");
            std::fflush(stdout);
            continue;
        }
        long long produced = 0;
        double decode_ms = 0.0;
        const char* finish = "length";
        if (pre.cancelled) {
            finish = "cancel";
        } else {
            const auto decode_begin = Clock::now();
            for (int64_t i = 0; i < req.max_new; ++i) {
                if (stdin_shared->stop.load()) { finish = "cancel"; break; }
                const float* logits = llama_get_logits_ith(ctx.get(), -1);
                if (logits == nullptr) {
                    std::printf("ERR decode produced no logits at output %lld\n", produced);
                    std::fflush(stdout);
                    finish = nullptr;   // ERR is the answer; no DONE follows a failed request
                    break;
                }
                if (logits_have_nan(logits, n_vocab)) {
                    std::fprintf(stderr, "strata qwen35 serve: the head logits are not all finite at output %lld; "
                                         "ending the request (finish=nonfinite)\n", produced);
                    finish = "nonfinite";
                    break;
                }
                const llama_token tok = llama_sampler_sample(sampler.get(), ctx.get(), -1);
                std::printf("T %d\n", (int) tok);
                std::fflush(stdout);
                ++produced;
                if (llama_vocab_is_eog(vocab, tok)) { finish = "stop"; break; }
                if (i + 1 < req.max_new) {
                    if (!fill_batch(batch.get(), &tok, 1, (int) (total + i)) ||
                        llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                        std::printf("ERR decode failed at output %lld\n", produced);
                        std::fflush(stdout);
                        finish = nullptr;
                        break;
                    }
                }
            }
            decode_ms = ms_since(decode_begin);
        }
        if (finish != nullptr) {
            // Field order is the official engine's (serve/server.py's _parse_done): reuse/hits/lookups and the
            // tier counters are 0 because phase 1 carries nothing across requests; prompt_read is how much of
            // the prompt was actually read (fewer than <prompt_tokens> when a STOP ended the read part-way).
            std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld\n",
                        produced, (long long) total, pre.prompt_ms, decode_ms, finish,
                        0LL /* accepted */, 0LL /* offered */, 0LL /* reused */, 0LL /* hits */, 0LL /* lookups */,
                        0LL /* ram blobs */, 0LL /* file blobs */, 0.0 /* file MB */, (long long) pre.read_n,
                        0LL /* offloaded */);
            std::fflush(stdout);
            std::fprintf(stderr, "serve: req %lld finish=%s produced=%lld prompt_ms=%.1f decode_ms=%.1f\n",
                         ++serve_reqs, finish, produced, pre.prompt_ms, decode_ms);
        }
    }
    core::qwen35_bridge_shutdown();
    return 0;
}

}  // namespace strata::program
