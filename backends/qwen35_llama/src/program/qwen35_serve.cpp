// src/program/qwen35_serve.cpp - the official server line protocol for the Qwen35 path (M2).
//
// `strata generate --qwen35-model <gguf> --qwen35-pack <dir> --serve` stops being a one-shot runner and becomes
// a resident engine for the official server (`serve/server.py`): it loads the model once, prints
// `READY <ctx> stop`, then serves any number of requests from stdin.  The line protocol is the official
// engine's own (parsed by serve/server.py's StrataEngine):
//
//   client -> engine:
//     GEN <max_new> [key=value ...] <id,id,...>   one request, sampled greedy unless temperature>0
//     SCORE <id,id,...> [topk=N] [cand=id,id,..]  read-only logits: prefill, generate nothing
//     CAPS                                        print the capability table (one JSON line), keep serving
//     SAVE <path>                                 P2-5: export the current session state to a file
//     LOAD <path>                                 P2-5: import a session state; GEN then continues it
//     STATE                                       P2-5: one JSON line of session bookkeeping
//     STOP                                        cancel the running request; the process keeps serving
//     QUIT                                        exit 0
//   engine -> client:
//     PP <done> <total> <ms> <fresh tok/s>        once per prefill chunk (ubatch-sized)
//     T <id>                                      one generated token, EOS included
//     LOGPROBS <pos> <k> <id>:<logprob> ...       SCORE's answer: top-k of the FULL-vocabulary log-softmax
//     CANDLOGPROBS <pos> <n> <id>:<logprob> ...   P2R-1: SCORE's `cand=` answer - the requested ids, in
//                                                 request order, from the same row/lse as LOGPROBS
//     DONE <produced> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <reused>
//          <hits> <lookups> <ram_blobs> <file_blobs> <file_mb> <prompt_read> <offloaded>
//     CAPS <json>                                 the --capabilities JSON (P2-1)
//     SAVED <bytes> <pos> <ms>                    SAVE's answer (state blob size, cursor, elapsed)
//     LOADED <bytes> <pos> <ms>                   LOAD's answer (same fields, after the import)
//     STATE <json>                                the session summary (one line)
//     ERR <message>                               request-level refusal; the engine keeps serving
//
// SCORE (P2-1) reads a prompt through the SAME prefill path as GEN (ubatch chunks, PP lines) and then takes
// the last position's logits: it emits no token.  It answers `LOGPROBS <pos> <k> <id>:<logprob> ...` where
// pos is the 0-based last prompt position and logprob is the log-softmax over EVERY logit of the vocabulary
// (no top-k normalization), followed by `DONE ... score` with produced=0.  P2R-1 adds the optional
// `cand=<id,...>` key: the same row and the same logsumexp answer every requested id - whatever its top-k
// rank - as `CANDLOGPROBS <pos> <n> <id>:<logprob> ...` in request order (after LOGPROBS, before DONE), and
// `topk=0` (legal only together with `cand=`) suppresses the LOGPROBS line.  A request without `cand=` is
// byte-identical to the P2-1 engine.  State semantics are (b) independent, by construction: SCORE starts
// from `llama_memory_clear` like GEN and resets the session again after printing, so the next request -
// GEN or SCORE - always starts fresh (reuse stays 0).
// CAPS answers the same JSON as `--capabilities` (see qwen35_capabilities_json and
// docs/backends/qwen35_llama_capabilities_and_score.md).
//
// SAVE / LOAD / STATE (P2-5) are the single-session state boundary: SAVE writes the CURRENT sequence-0 state
// as the raw bytes `llama_state_seq_get_data` produces (attention KV + GDN conv/SSM planes + cell
// positions - no new serialization format), LOAD imports such a file, STATE prints one JSON line about the
// session.  A successful SAVE or LOAD arms session continuation: the next GEN appends its prompt at the
// current cursor instead of resetting first (`reused` reports the carried positions), which is how a
// restored session "continues".  Without SAVE/LOAD the per-request reset below is untouched.  LOAD imports
// into a TEMPORARY context and swaps only after the import fully succeeded, so a bad path/file leaves the
// live session untouched; SCORE's semantics (b) reset clears the session (documented in
// docs/backends/qwen35_llama_session_state.md - the full contract, boundaries and P2-6 plan).
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
// the official engine's SAVE/RESTORE disk-session grammar, MTP.  Unknown directives answer `ERR ...` and the
// loop continues - it never exits or crashes on input it does not know.
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
#include <cctype>
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
#  include <windows.h>   // MoveFileExA: replace-on-save for the session file
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

/// P2R-1: the hard cap of explicit `cand=` ids per SCORE (refused above it; `topk` covers wide dumps).
constexpr size_t kScoreMaxCand = 2048;

/// One parsed SCORE request (P2-1; `cand=` is P2R-1): a prompt, the number of candidate logits to report
/// (default 5; 0 = answer only `cand`, no LOGPROBS line) and the optional explicit candidate token ids.
struct ScoreRequest {
    std::vector<llama_token> ids;
    int32_t topk = 5;
    bool cand_present = false;
    std::vector<llama_token> cand;
};

/// Does the line carry a `cand=` key?  Pass 1 of the two-pass parse: `topk=0` is legal only together with
/// `cand=`, but the `topk` token may come first, so the `cand` presence has to be knowable when `topk` is
/// checked - and a request without `cand=` must keep the legacy key-error order and texts byte-for-byte.
bool score_line_has_cand_key(const std::string& line) {
    const char* p = line.c_str() + 5;   // past "SCORE"
    for (;;) {
        while (*p == ' ') ++p;
        if (*p == '\0') return false;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        const std::string tok(start, (size_t) (p - start));
        const size_t eq = tok.find('=');
        if (eq != std::string::npos && eq == 4 && tok.compare(0, 4, "cand") == 0) return true;
    }
}

/// `SCORE <id,id,...> [topk=N] [cand=id,id,...]`: the only non-key token is the id list (it may come before
/// or after the keys).  `topk` and `cand` are the only keys; any other key is refused - a silently ignored
/// key typo would answer a different question than the one asked.  `topk` must be a non-negative integer
/// (the run clamps it to the vocabulary size); `topk=0` answers only the explicit candidates and therefore
/// needs a `cand=` list.  A repeated `topk` takes the last value (unchanged); a repeated `cand=` key is
/// refused - a silently dropped id list would be worse than an error.  `cand=` ids must be decimal,
/// non-negative, pairwise distinct (duplicates are refused, never deduplicated silently) and at most 2048
/// many; whether they are inside the model's vocabulary is checked by the caller, exactly like the prompt
/// ids and with the same error text.
bool parse_score_line(const std::string& line, ScoreRequest& req, std::string& err) {
    const char* p = line.c_str() + 5;   // past "SCORE"
    const char* ids = nullptr;
    const char* ids_end = nullptr;
    const bool has_cand = score_line_has_cand_key(line);
    bool cand_seen = false;
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
        if (key == "topk") {
            char* ve = nullptr;
            const long long v = std::strtoll(val.c_str(), &ve, 10);
            if (!parse_scalar(val, ve) || v < 0 || v > std::numeric_limits<int32_t>::max() ||
                (v == 0 && !has_cand)) {
                err = "bad request: topk";
                return false;
            }
            req.topk = (int32_t) v;
            continue;
        }
        if (key == "cand") {
            if (cand_seen) { err = "bad request: duplicate cand key"; return false; }
            cand_seen = true;
            req.cand_present = true;
            if (val.empty()) { err = "bad request: cand"; return false; }
            for (size_t q = 0;;) {
                const char* at = val.c_str() + q;
                char* ie = nullptr;
                const long long v = std::strtoll(at, &ie, 10);
                if (ie == at || v < 0 || v > std::numeric_limits<llama_token>::max()) {
                    err = "bad request: cand";
                    return false;
                }
                req.cand.push_back((llama_token) v);
                const size_t adv = (size_t) (ie - val.c_str());
                if (adv == val.size()) break;
                if (val[adv] != ',' || adv + 1 == val.size()) { err = "bad request: cand"; return false; }
                q = adv + 1;
            }
            if (req.cand.size() > kScoreMaxCand) {
                err = "bad request: cand has more than " + std::to_string(kScoreMaxCand) +
                      " ids (use topk= for a wide distribution)";
                return false;
            }
            std::vector<llama_token> sorted(req.cand);
            std::sort(sorted.begin(), sorted.end());
            for (size_t i = 1; i < sorted.size(); ++i) {
                if (sorted[i] == sorted[i - 1]) {
                    err = "bad request: duplicate cand id " + std::to_string(sorted[i]);
                    return false;
                }
            }
            continue;
        }
        err = "bad request: unknown key " + key;
        return false;
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
                       int64_t chunk_max, const std::shared_ptr<StdinQueue>& stdin_shared, int64_t base_pos) {
    PrefillRun r;
    const auto prefill_begin = Clock::now();
    const int64_t total = (int64_t) ids.size();
    for (int64_t at = 0; at < total; at += chunk_max) {
        if (stdin_shared->stop.load()) { r.cancelled = true; break; }
        const int count = (int) std::min<int64_t>(chunk_max, total - at);
        if (!fill_batch(batch, ids.data() + at, count, (int) (base_pos + at))) { r.failed = true; break; }
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

/// P2-5 session bookkeeping: `active` is set by a successful SAVE or LOAD and makes GEN continue the session
/// instead of resetting; it is cleared whenever the state is reset for real (SCORE's semantics (b) reset).
/// `from_load` distinguishes a session that came out of a file (LOAD) from one that was merely checkpointed
/// (SAVE); `path`/`file_bytes` describe the last successful session file.  STATE reports these facts.
struct SessionBook {
    bool active = false;
    bool from_load = false;
    std::string path;
    size_t file_bytes = 0;
};

/// The cursor a continuation starts at: last occupied position + 1 (0 when the sequence is empty).
int64_t state_cursor(llama_context* ctx) {
    const llama_pos last = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    return last < 0 ? 0 : (int64_t) last + 1;
}

/// The exact byte size of what SAVE would write for sequence 0 (llama's own size walk; no new format).
size_t state_blob_size(llama_context* ctx) {
    return llama_state_seq_get_size(ctx, 0);
}

/// Absolute-path shape check (drive-qualified or UNC on Windows, leading '/' elsewhere).
bool path_is_absolute(const std::string& p) {
    if (p.size() < 2) return false;
#if defined(_WIN32)
    if (p[0] == '\\' && p[1] == '\\') return true;   // UNC \\server\share
    return std::isalpha((unsigned char) p[0]) && p[1] == ':' && p.size() > 2 && (p[2] == '\\' || p[2] == '/');
#else
    return p[0] == '/';
#endif
}

/// Session paths travel through the line protocol and ANSI file APIs: restrict them to printable ASCII.
bool path_printable_ascii(const std::string& p) {
    for (const char c : p) {
        if ((unsigned char) c < 0x20 || (unsigned char) c > 0x7e) return false;
    }
    return true;
}

/// Rejects any '..' component (path traversal) - the check is textual, symbolic links are not resolved.
bool path_has_dotdot(const std::string& p) {
    for (size_t at = 0; at < p.size();) {
        while (at < p.size() && (p[at] == '\\' || p[at] == '/')) ++at;
        const size_t start = at;
        while (at < p.size() && p[at] != '\\' && p[at] != '/') ++at;
        if (at - start == 2 && p[start] == '.' && p[start + 1] == '.') return true;
    }
    return false;
}

/// '/'-separated, ASCII-lowercased (Windows), no trailing separator: the form the allow-list compares.
std::string normalized_path(const std::string& p) {
    std::string s = p;
    for (char& c : s) {
        if (c == '\\') c = '/';
#if defined(_WIN32)
        if (c >= 'A' && c <= 'Z') c = (char) (c - 'A' + 'a');
#endif
    }
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    return s;
}

bool path_under_root(const std::string& np, const std::string& root) {
    const std::string nr = normalized_path(root);
    if (nr.empty() || np.size() < nr.size() || np.compare(0, nr.size(), nr) != 0) return false;
    return np.size() == nr.size() || np[nr.size()] == '/';
}

/// The full SAVE/LOAD path gate: absolute, printable ASCII, no '..', and - when the optional
/// STRATA_QWEN35_SESSION_ROOTS allow-list (semicolon-separated absolute directories) is set - contained in
/// one of its roots.  The roots are compared normalized but not resolved (no symlink/junction following).
bool session_path_allowed(const std::string& path, std::string& err) {
    if (path.empty()) { err = "session: empty path"; return false; }
    if (path.size() > 1024) { err = "session: path longer than 1024 characters"; return false; }
    if (!path_printable_ascii(path)) { err = "session: path must be printable ASCII"; return false; }
    if (!path_is_absolute(path)) { err = "session: path must be absolute"; return false; }
    if (path.back() == '\\' || path.back() == '/') { err = "session: path must name a file, not a directory"; return false; }
    if (path_has_dotdot(path)) { err = "session: '..' is not allowed in the path"; return false; }
    const char* roots_env = std::getenv("STRATA_QWEN35_SESSION_ROOTS");
    if (roots_env != nullptr && roots_env[0] != '\0') {
        const std::string roots = roots_env;
        const std::string np = normalized_path(path);
        bool ok = false;
        for (size_t at = 0; at <= roots.size();) {
            size_t sep = roots.find(';', at);
            if (sep == std::string::npos) sep = roots.size();
            if (sep > at && path_under_root(np, roots.substr(at, sep - at))) { ok = true; break; }
            if (sep == roots.size()) break;
            at = sep + 1;
        }
        if (!ok) { err = "session: path is outside STRATA_QWEN35_SESSION_ROOTS"; return false; }
    }
    return true;
}

constexpr int64_t kSessionFileCap = (int64_t) 1 << 30;   // 1 GiB: a size gate, not a measurement

bool read_file_bytes(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { *err = "session: cannot open '" + path + "' for reading"; return false; }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    if (sz < 0) { std::fclose(f); *err = "session: cannot size '" + path + "'"; return false; }
    if ((int64_t) sz < 8) {
        std::fclose(f);
        *err = "session: '" + path + "' is too small to be a sequence state";
        return false;
    }
    if ((int64_t) sz > kSessionFileCap) {
        std::fclose(f);
        *err = "session: '" + path + "' exceeds the 1 GiB session-file cap";
        return false;
    }
    std::fseek(f, 0, SEEK_SET);
    out->resize((size_t) sz);
    const size_t got = std::fread(out->data(), 1, out->size(), f);
    const int close_rc = std::fclose(f);
    if (got != out->size() || close_rc != 0) { *err = "session: cannot read '" + path + "'"; return false; }
    return true;
}

/// Writes `<path>.tmp`, then replaces `path` with it (MoveFileExA / rename): a failed save never leaves the
/// previous file truncated; the stray .tmp is removed on every failure.
bool write_file_atomic(const std::string& path, const uint8_t* data, size_t size, std::string* err) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { *err = "session: cannot open '" + tmp + "' for writing"; return false; }
    const size_t put = std::fwrite(data, 1, size, f);
    const int close_rc = std::fclose(f);
    if (put != size || close_rc != 0) {
        std::remove(tmp.c_str());
        *err = "session: cannot write '" + tmp + "'";
        return false;
    }
#if defined(_WIN32)
    if (MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) == 0) {
        std::remove(tmp.c_str());
        *err = "session: cannot replace '" + path + "'";
        return false;
    }
#else
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        *err = "session: cannot replace '" + path + "'";
        return false;
    }
#endif
    return true;
}

/// Minimal JSON string quoting for paths (they are printable ASCII by the gate above; only " and \ need it).
std::string json_quote(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    out += '"';
    return out;
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
                  "\"score_explicit_candidate_ids\":true,\"score_max_candidate_ids\":%lld,"
                  "\"state_export\":true,\"disk_kv\":false,\"vision\":false}",
                  mtp_available ? "true" : "false", env_is_exactly_one("STRATA_QWEN35_MTP") ? "true" : "false",
                  fmts.c_str(), (long long) context_max, (long long) kScoreMaxCand);
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
    // P2-5: LOAD builds its temporary context with EXACTLY these parameters before swapping it in - a
    // restored session must run under the same context shape it was exported from.
    const llama_context_params cp_saved = cp;
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
    SessionBook session;              // P2-5: armed by SAVE/LOAD, cleared by any real reset (SCORE's (b))
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
        if (line == "STATE") {   // P2-5: the one-line session summary (fields: session_state doc)
            const int64_t cursor = state_cursor(ctx.get());
            const size_t live_bytes = state_blob_size(ctx.get());
            std::string js = "{\"loaded\":";
            js += (session.active && session.from_load) ? "true" : "false";
            js += ",\"pos\":";
            js += std::to_string(cursor);
            js += ",\"bytes\":";
            js += std::to_string(live_bytes);
            js += ",\"path\":";
            js += (session.active && !session.path.empty()) ? json_quote(session.path) : "null";
            js += ",\"covers\":[\"attn_kv\",\"gdn_state\",\"conv\",\"pos\"]";
            js += ",\"session\":";
            js += session.active ? "true" : "false";
            js += ",\"origin\":";
            js += session.active ? (session.from_load ? "\"load\"" : "\"save\"") : "null";
            js += ",\"file_bytes\":";
            js += (session.active && session.file_bytes > 0) ? std::to_string(session.file_bytes) : "null";
            js += ",\"max_context\":";
            js += std::to_string(total_ctx);
            js += ",\"not_covered\":[\"logits\",\"sampler_rng_state\",\"expert_cache\",\"bridge_counters\","
                  "\"rollback_history\"]}";
            std::printf("STATE %s\n", js.c_str());
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("SAVE ", 0) == 0) {
            // P2-5: export the CURRENT sequence-0 state as the raw llama_state_seq bytes.  On success the
            // session is armed: the next GEN appends at the cursor (continuation) instead of resetting.
            std::string path = line.substr(5);
            while (!path.empty() && path.front() == ' ') path.erase(0, 1);
            while (!path.empty() && path.back() == ' ') path.pop_back();
            const auto save_begin = Clock::now();
            std::string derr;
            if (!session_path_allowed(path, derr)) {
                std::printf("ERR %s\n", derr.c_str());
                std::fflush(stdout);
                continue;
            }
            std::vector<uint8_t> blob(state_blob_size(ctx.get()));
            if (blob.empty()) {
                std::printf("ERR session: the state size walk failed\n");
                std::fflush(stdout);
                continue;
            }
            const size_t got = llama_state_seq_get_data(ctx.get(), blob.data(), blob.size(), 0);
            if (got != blob.size()) {
                std::printf("ERR session: the state export returned %zu of %zu bytes\n", got, blob.size());
                std::fflush(stdout);
                continue;
            }
            if (!write_file_atomic(path, blob.data(), blob.size(), &derr)) {
                std::printf("ERR %s\n", derr.c_str());
                std::fflush(stdout);
                continue;
            }
            session.active = true;
            session.from_load = false;
            session.path = path;
            session.file_bytes = blob.size();
            std::printf("SAVED %zu %lld %.1f\n", blob.size(), (long long) state_cursor(ctx.get()),
                        ms_since(save_begin));
            std::fflush(stdout);
            std::fprintf(stderr, "serve: session save '%s' bytes=%zu pos=%lld\n", path.c_str(), blob.size(),
                         (long long) state_cursor(ctx.get()));
            continue;
        }
        if (line.rfind("LOAD ", 0) == 0) {
            // P2-5: import a session file.  The bytes go into a TEMPORARY context first (same model and
            // context parameters); only a fully successful import is swapped in, so a bad path, a foreign
            // file or a corrupt blob leaves the live session untouched - never a half-restored state.
            std::string path = line.substr(5);
            while (!path.empty() && path.front() == ' ') path.erase(0, 1);
            while (!path.empty() && path.back() == ' ') path.pop_back();
            const auto load_begin = Clock::now();
            std::string derr;
            if (!session_path_allowed(path, derr)) {
                std::printf("ERR %s\n", derr.c_str());
                std::fflush(stdout);
                continue;
            }
            std::vector<uint8_t> blob;
            if (!read_file_bytes(path, &blob, &derr)) {
                std::printf("ERR %s\n", derr.c_str());
                std::fflush(stdout);
                continue;
            }
            // The llama sequence-state header is <u32 magic 0xaf143cd8><i32 seq_id>; checking it here
            // turns the common "not a state file" / "wrong sequence" mistakes into precise errors.
            uint32_t magic = 0;
            int32_t seq_header = -1;
            std::memcpy(&magic, blob.data(), sizeof(magic));
            std::memcpy(&seq_header, blob.data() + sizeof(magic), sizeof(seq_header));
            if (magic != 0xaf143cd8u) {
                std::printf("ERR session: '%s' is not a llama sequence-state blob (bad header)\n", path.c_str());
                std::fflush(stdout);
                continue;
            }
            if (seq_header != 0) {
                std::printf("ERR session: '%s' was saved from sequence %d; this engine restores sequence 0\n",
                            path.c_str(), (int) seq_header);
                std::fflush(stdout);
                continue;
            }
            llama_context* tmp = llama_init_from_model(model.get(), cp_saved);
            if (tmp == nullptr) {
                std::printf("ERR session: the temporary import context could not be created\n");
                std::fflush(stdout);
                continue;
            }
            const size_t got = llama_state_seq_set_data(tmp, blob.data(), blob.size(), 0);
            llama_batch_ext* tmp_batch = got != 0 ? llama_batch_ext_init(tmp) : nullptr;
            if (got == 0 || tmp_batch == nullptr) {
                if (tmp_batch != nullptr) llama_batch_ext_free(tmp_batch);
                llama_free(tmp);
                std::printf("ERR session: the import failed (incompatible model/context or corrupt state); "
                            "the live session is unchanged\n");
                std::fflush(stdout);
                continue;
            }
            batch.reset(tmp_batch);   // frees the old batch first; the old context is freed next
            ctx.reset(tmp);
            session.active = true;
            session.from_load = true;
            session.path = path;
            session.file_bytes = blob.size();
            std::printf("LOADED %zu %lld %.1f\n", blob.size(), (long long) state_cursor(ctx.get()),
                        ms_since(load_begin));
            std::fflush(stdout);
            std::fprintf(stderr, "serve: session load '%s' bytes=%zu pos=%lld\n", path.c_str(), blob.size(),
                         (long long) state_cursor(ctx.get()));
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
            for (size_t i = 0; i < sreq.cand.size() && !bad_id; ++i) {
                if (sreq.cand[i] < 0 || (int64_t) sreq.cand[i] >= n_vocab) bad_id = true;
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
            session = SessionBook{};   // semantics (b): the SCORE reset ends any active session
            const int64_t total = (int64_t) sreq.ids.size();
            const PrefillRun pre = run_prefill(ctx.get(), batch.get(), sreq.ids, chunk_max, stdin_shared, 0);
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
            // P2R-1: the explicit candidates are answered from this SAME row and this SAME lse - they never
            // trigger a second forward, and their logprob is exactly `logits[id] - lse` as printed here.
            float mx = logits[0];
            for (int64_t i = 1; i < n_vocab; ++i) {
                if (logits[i] > mx) mx = logits[i];
            }
            double sum = 0.0;
            for (int64_t i = 0; i < n_vocab; ++i) sum += std::exp((double) logits[i] - (double) mx);
            const double lse = (double) mx + std::log(sum);
            const int64_t k = std::min<int64_t>(sreq.topk, n_vocab);
            if (sreq.topk > 0) {
                // Top-k by (logit desc, id asc): an exact tie answers the same order every run.  `topk=0`
                // (legal only with `cand=`) skips this line entirely - an empty LOGPROBS would be no answer.
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
            }
            if (sreq.cand_present) {
                // Request order, no sort and no dedup (duplicates were refused at parse time): the answer
                // mirrors the question.
                std::printf("CANDLOGPROBS %lld %lld", (long long) (total - 1), (long long) sreq.cand.size());
                for (const llama_token id : sreq.cand) {
                    std::printf(" %d:%.4f", (int) id, (double) logits[id] - lse);
                }
                std::printf("\n");
            }
            std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld\n",
                        0LL, (long long) total, pre.prompt_ms, 0.0, "score", 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL,
                        0.0, (long long) pre.read_n, 0LL);
            std::fflush(stdout);
            if (sreq.cand_present) {
                std::fprintf(stderr, "serve: req %lld finish=score produced=0 prompt_ms=%.1f decode_ms=0.0 "
                                     "topk=%lld cand=%lld\n",
                             ++serve_reqs, pre.prompt_ms, (long long) k, (long long) sreq.cand.size());
            } else {
                std::fprintf(stderr, "serve: req %lld finish=score produced=0 prompt_ms=%.1f decode_ms=0.0 "
                                     "topk=%lld\n",
                             ++serve_reqs, pre.prompt_ms, (long long) k);
            }
            reset_session(ctx.get());
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: CAPS | STATE | SAVE <path> | LOAD <path> | "
                        "SCORE <id,id,...> [topk=N] [cand=id,id,...] | "
                        "GEN <max_new> [key=value ...] <id,id,...>\n");
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
        // P2-5: after a successful SAVE/LOAD the session is armed and this GEN APPENDS its prompt at the
        // cursor (the continuation); without a session the request is isolated exactly as before - fresh
        // state from position 0.  `base_pos` is also what the DONE `reused` field reports.
        const bool session_gen = session.active;
        const int64_t base_pos = session_gen ? state_cursor(ctx.get()) : 0;
        if (base_pos + (int64_t) req.ids.size() + req.max_new > total_ctx) {
            if (session_gen) {
                std::printf("ERR session (%lld) + prompt (%lld tokens) + max_new (%lld) exceeds the context "
                            "(%lld)\n", (long long) base_pos, (long long) req.ids.size(), (long long) req.max_new,
                            (long long) total_ctx);
            } else {
                std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n",
                            (long long) req.ids.size(), (long long) req.max_new, (long long) total_ctx);
            }
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
        // diagnostic use - skipped when the request continues an armed session.
        if (!session_gen) reset_session(ctx.get());
        // Prefill, one ubatch-sized chunk at a time; PP after every chunk (the server's progress lines).
        const int64_t total = (int64_t) req.ids.size();
        const PrefillRun pre = run_prefill(ctx.get(), batch.get(), req.ids, chunk_max, stdin_shared, base_pos);
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
                    if (!fill_batch(batch.get(), &tok, 1, (int) (base_pos + total + i)) ||
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
            // Field order is the official engine's (serve/server.py's _parse_done).  `reused` reports the
            // positions carried into the request: 0 for an isolated request, the session cursor for a
            // continuation (P2-5); hits/lookups and the tier counters stay 0 - there is no cache tier behind
            // them.  prompt_read is how much of the prompt was actually read (fewer than <prompt_tokens> when
            // a STOP ended the read part-way).
            std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld\n",
                        produced, (long long) total, pre.prompt_ms, decode_ms, finish,
                        0LL /* accepted */, 0LL /* offered */, (long long) (session_gen ? base_pos : 0) /* reused */,
                        0LL /* hits */, 0LL /* lookups */,
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
