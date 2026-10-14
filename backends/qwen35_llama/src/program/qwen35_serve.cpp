// src/program/qwen35_serve.cpp - the official server line protocol for the Qwen35 path (M2).
//
// `strata generate --qwen35-model <gguf> --qwen35-pack <dir> --serve` stops being a one-shot runner and becomes
// a resident engine for the official server (`serve/server.py`): it loads the model once, prints
// `READY <ctx> stop`, then serves any number of requests from stdin.  The line protocol is the official
// engine's own (parsed by serve/server.py's StrataEngine):
//
//   client -> engine:
//     GEN <max_new> [key=value ...] <id,id,...>   one request, sampled greedy unless temperature>0
//     STOP                                        cancel the running request; the process keeps serving
//     QUIT                                        exit 0
//   engine -> client:
//     PP <done> <total> <ms> <fresh tok/s>        once per prefill chunk (ubatch-sized)
//     T <id>                                      one generated token, EOS included
//     DONE <produced> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <reused>
//          <hits> <lookups> <ram_blobs> <file_blobs> <file_mb> <prompt_read> <offloaded>
//     ERR <message>                               request-level refusal; the engine keeps serving
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

#include "strata/core/qwen35_bridge.hpp"
#include "strata/core/qwen35_mtp.hpp"
#include "strata/core/qwen35_mtp_txn.hpp"

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

/// `GEN <max_new> [key=value ...] <id,id,...>`; the ids start at the first whitespace token without '=' and run
/// to the end of the line (same split rule as the official engine's parser).  Returns false with `err` set.
bool parse_gen_line(const std::string& line, GenRequest& req, std::string& err) {
    const int64_t n_vocab_max = std::numeric_limits<llama_token>::max();
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
    if (ids_end == ids) { err = "bad request: no token ids"; return false; }
    for (const char* q = ids; q != ids_end; ++q) {
        if (*q == ' ') { err = "bad request: token ids"; return false; }
    }
    const std::string ids_txt(ids, (size_t) (ids_end - ids));
    for (size_t q = 0;;) {
        const char* at = ids_txt.c_str() + q;
        char* ie = nullptr;
        const long long v = std::strtoll(at, &ie, 10);
        if (ie == at || v < 0 || v > n_vocab_max) { err = "bad request: token ids"; return false; }
        req.ids.push_back((llama_token) v);
        const size_t adv = (size_t) (ie - ids_txt.c_str());
        if (adv == ids_txt.size()) break;
        if (ids_txt[adv] != ',' || adv + 1 == ids_txt.size()) { err = "bad request: token ids"; return false; }
        q = adv + 1;
    }
    return true;
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

}  // namespace

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
    std::string line;
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
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
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
        // Fresh request state: cell metadata + KV cells and the recurrent r/s/p buffers (GDN conv + SSM +
        // rollback planes), exactly the reset the two-request diagnostic trusts.
        if (llama_memory_t mem = llama_get_memory(ctx.get())) {
            llama_memory_clear(mem, true);
        }
        // Prefill, one ubatch-sized chunk at a time; PP after every chunk (the server's progress lines).
        const auto prefill_begin = Clock::now();
        const int64_t total = (int64_t) req.ids.size();
        int64_t read_n = 0;
        bool prefill_failed = false;
        bool prefill_cancelled = false;
        for (int64_t at = 0; at < total; at += chunk_max) {
            if (stdin_shared->stop.load()) { prefill_cancelled = true; break; }
            const int count = (int) std::min<int64_t>(chunk_max, total - at);
            if (!fill_batch(batch.get(), req.ids.data() + at, count, (int) at)) { prefill_failed = true; break; }
            const double chunk_begin_ms = ms_since(prefill_begin);
            if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                prefill_failed = true;
                break;
            }
            read_n = at + count;
            const double now_ms = ms_since(prefill_begin);
            const double chunk_ms = now_ms - chunk_begin_ms;
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) read_n, (long long) total, now_ms,
                        chunk_ms > 0.0 ? 1000.0 * (double) count / chunk_ms : 0.0);
            std::fflush(stdout);
        }
        // The prompt -> decode transition must run even after a failed or cancelled prompt read: the lent
        // cache slots are refilled and the decode tier sees them again (a false return is a real failure).
        const bool loan_returned = core::qwen35_bridge_finish_prompt();
        const double prompt_ms = ms_since(prefill_begin);
        if (prefill_failed) {
            std::printf("ERR prefill failed\n");
            std::fflush(stdout);
            continue;
        }
        if (!loan_returned) {
            std::printf("ERR the prompt workspace loan was not returned\n");
            std::fflush(stdout);
            continue;
        }
        long long produced = 0;
        double decode_ms = 0.0;
        const char* finish = "length";
        if (prefill_cancelled) {
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
                        produced, (long long) total, prompt_ms, decode_ms, finish,
                        0LL /* accepted */, 0LL /* offered */, 0LL /* reused */, 0LL /* hits */, 0LL /* lookups */,
                        0LL /* ram blobs */, 0LL /* file blobs */, 0.0 /* file MB */, (long long) read_n,
                        0LL /* offloaded */);
            std::fflush(stdout);
        }
    }
    core::qwen35_bridge_shutdown();
    return 0;
}

}  // namespace strata::program
