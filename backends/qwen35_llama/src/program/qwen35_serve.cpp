// src/program/qwen35_serve.cpp - the official server line protocol for the Qwen35 path (M2).
//
// `strata generate --qwen35-model <gguf> --qwen35-pack <dir> --serve` stops being a one-shot runner and becomes
// a resident engine for the official server (`serve/server.py`): it loads the model once, prints
// `READY <ctx> stop`, then serves any number of requests from stdin.  The line protocol is the official
// engine's own (parsed by serve/server.py's StrataEngine):
//
//   client -> engine:
//     GEN <max_new> [key=value ...] <id,id,...>   one request, sampled greedy unless temperature>0
//                                                 P2R-3a: [cont=1] - explicit session continuation
//     SCORE <id,id,...> [topk=N] [cand=id,id,..]  read-only logits: prefill, generate nothing
//                                                 P2R-3a: [keep=1] - state-keeping score (no reset)
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
//     SAVED <bytes> <pos> <ms>                    SAVE's answer (container file size, cursor, elapsed)
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
// GEN or SCORE - always starts fresh (reuse stays 0).  P2R-3a adds the opt-in `keep=1` key: the SAME
// read-only answer, but the session is NOT reset - the prompt is appended at the current session cursor
// (a reset first only when no session was armed), `DONE`'s `reused` reports the carried positions, and the
// session stays armed afterwards so the caller can continue it (GEN `cont=1` or a plain session GEN); a
// successful keep request that found no armed session arms one with `STATE` `origin="score-keep"`.  A
// request without `keep=` is byte-identical to the pre-P2R-3a engine.
// CAPS answers the same JSON as `--capabilities` (see qwen35_capabilities_json and
// docs/backends/qwen35_llama_capabilities_and_score.md).
//
// SAVE / LOAD / STATE (P2-5; P2R-4a container) are the single-session state boundary: SAVE writes the
// CURRENT sequence-0 state wrapped in a self-describing container (`STRATAST` magic, version, header and
// payload lengths, payload CRC32, model fingerprint, vocabulary, context, cursor), LOAD validates that
// container first and only then imports it, STATE prints one JSON line about the session.  The container
// payload itself is unchanged from P2-5: the raw bytes `llama_state_seq_get_data` produces (attention KV
// + GDN conv/SSM planes + cell positions); the container only describes and protects them.  A successful
// SAVE or LOAD arms session continuation: the next GEN appends its prompt at the
// current cursor instead of resetting first (`reused` reports the carried positions), which is how a
// restored session "continues".  Without SAVE/LOAD the per-request reset below is untouched.  LOAD checks
// the container in a fixed order (file size -> magic -> version -> header length -> vocabulary -> context
// -> pos -> payload CRC32 -> model fingerprint), answers a distinct `ERR` for every mismatch and leaves
// the live session untouched on any failure (including a legacy raw blob from a pre-P2R-4a build, which is
// refused with its own `ERR`); only after all checks passed is the payload imported
// into a TEMPORARY context and swapped only after the import fully succeeded, so a bad path/file leaves the
// live session untouched; SCORE's semantics (b) reset clears the session (documented in
// docs/backends/qwen35_llama_session_state.md - the full contract, boundaries and P2-6 plan).  P2R-3a adds
// `GEN ... cont=1` (see that doc): an EXPLICIT continuation - it refuses to run without an armed session
// and accepts an EMPTY id list (pure continuation, append nothing and decode from the cursor); a request
// without `cont=` keeps the exact behavior above.  `SCORE ... keep=1` arms/advances the same session
// machinery without any file (origin `score-keep`; SAVE/LOAD are neither required nor implied).
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
/// P2R-3a: `cont` is the explicit continuation key (`cont=1`; requires an armed session).
struct GenRequest {
    int64_t max_new = 0;
    std::vector<llama_token> ids;
    bool cont = false;          ///< P2R-3a: append at the armed session's cursor (empty id list allowed)
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

/// `GEN <max_new> [key=value ...] [cont=1] <id,id,...>`; the ids start at the first whitespace token without
/// '=' and run to the end of the line (same split rule as the official engine's parser).  Returns false with
/// `err` set.  P2R-3a: `cont` only accepts the literal `1` (anything else is refused, never ignored); a
/// repeated `cont=` key is refused.  With `cont=1` an empty id list is legal (pure continuation - append
/// nothing); without it the empty list stays the legacy `ERR bad request: no token ids`.
bool parse_gen_line(const std::string& line, GenRequest& req, std::string& err) {
    const char* p = line.c_str() + 4;   // past "GEN "
    while (*p == ' ') ++p;
    if (*p < '0' || *p > '9') { err = "bad request: max_new"; return false; }
    char* end = nullptr;
    errno = 0;
    const long long max_new = std::strtoll(p, &end, 10);
    if (errno == ERANGE || end == p || max_new < 1 || (*end != ' ' && *end != '\0')) {
        err = "bad request: max_new"; return false;
    }
    req.max_new = max_new;
    p = end;
    bool cont_seen = false;
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
        } else if (key == "cont") {
            // P2R-3a: only the literal `1`; a repeated key is refused (a dropped continuation flag would
            // silently turn the request into a different one).
            if (cont_seen) { err = "bad request: duplicate cont key"; return false; }
            cont_seen = true;
            if (val != "1") { err = "bad request: cont"; return false; }
            req.cont = true;
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
    if (ids_end == ids && req.cont) return true;   // P2R-3a: cont=1 with no ids is a pure continuation
    return parse_id_list(ids, ids_end, req.ids, err);
}

/// P2R-1: the hard cap of explicit `cand=` ids per SCORE (refused above it; `topk` covers wide dumps).
constexpr size_t kScoreMaxCand = 2048;

/// One parsed SCORE request (P2-1; `cand=` is P2R-1; `keep=` is P2R-3a): a prompt, the number of candidate
/// logits to report (default 5; 0 = answer only `cand`, no LOGPROBS line) and the optional explicit
/// candidate token ids.  `keep` (`keep=1` only) makes the score state-keeping: no reset, append at the
/// armed session's cursor, stay armed after the answer.
struct ScoreRequest {
    std::vector<llama_token> ids;
    int32_t topk = 5;
    bool cand_present = false;
    std::vector<llama_token> cand;
    bool keep = false;
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

/// `SCORE <id,id,...> [topk=N] [cand=id,id,...] [keep=1]`: the only non-key token is the id list (it may
/// come before or after the keys).  `topk`, `cand` and `keep` are the only keys; any other key is refused -
/// a silently ignored key typo would answer a different question than the one asked.  `topk` must be a
/// non-negative integer (the run clamps it to the vocabulary size); `topk=0` answers only the explicit
/// candidates and therefore needs a `cand=` list.  A repeated `topk` takes the last value (unchanged); a
/// repeated `cand=` key is refused - a silently dropped id list would be worse than an error.  `cand=` ids
/// must be decimal, non-negative, pairwise distinct (duplicates are refused, never deduplicated silently)
/// and at most 2048 many; whether they are inside the model's vocabulary is checked by the caller, exactly
/// like the prompt ids and with the same error text.  P2R-3a: `keep` accepts only the literal `1` (any
/// other value is refused - never a silent default); a repeated `keep=` key is refused.
bool parse_score_line(const std::string& line, ScoreRequest& req, std::string& err) {
    const char* p = line.c_str() + 5;   // past "SCORE"
    const char* ids = nullptr;
    const char* ids_end = nullptr;
    const bool has_cand = score_line_has_cand_key(line);
    bool cand_seen = false;
    bool keep_seen = false;
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
        if (key == "keep") {
            if (keep_seen) { err = "bad request: duplicate keep key"; return false; }
            keep_seen = true;
            if (val != "1") { err = "bad request: keep"; return false; }
            req.keep = true;
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

/// P2-5 session bookkeeping: `active` is set by a successful SAVE or LOAD (or a P2R-3a SCORE `keep=1`
/// that found no armed session) and makes GEN continue the session instead of resetting; it is cleared
/// whenever the state is reset for real (SCORE's semantics (b) reset, a failed/cancelled keep request).
/// `from_load` distinguishes a session that came out of a file (LOAD) from one that was merely checkpointed
/// (SAVE); `from_score_keep` (P2R-3a) marks a session armed by `SCORE keep=1` with no file; `path`/
/// `file_bytes` describe the last successful session file (P2R-4a: `file_bytes` is the CONTAINER file size,
/// `payload_bytes` the raw sequence-state bytes inside it - both null when there is no file).  STATE
/// reports these facts.  Continuing an armed session (GEN, `cont=1` or `keep=1` on top of one) never
/// re-arms it: origin and file fields stay.
struct SessionBook {
    bool active = false;
    bool from_load = false;
    bool from_score_keep = false;
    std::string path;
    size_t file_bytes = 0;
    size_t payload_bytes = 0;
};

/// The cursor a continuation starts at: last occupied position + 1 (0 when the sequence is empty).
int64_t state_cursor(llama_context* ctx) {
    const llama_pos last = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    return last < 0 ? 0 : (int64_t) last + 1;
}

/// The exact byte size of the raw sequence-0 state (llama's own size walk; the container payload size).
size_t state_blob_size(llama_context* ctx) {
    return llama_state_seq_get_size(ctx, 0);
}

/// P2R-4a: the session-file container - a fixed 96-byte little-endian header in front of the P2-5 raw
/// payload.  layout (offsets in bytes): 0 magic "STRATAST" (8); 8 version (u32, 1); 12 header_bytes (u32,
/// this header's total length, 96 for v1); 16 payload_bytes (u64, raw sequence-state bytes); 24 file_bytes
/// (u64, header_bytes + payload_bytes); 32 payload_crc32 (u32, CRC-32/ISO-HDLC over the payload); 36
/// n_vocab (u32); 40 max_context (u32); 44 pos (u64, sequence cursor at SAVE); 52 model_fp (32 bytes);
/// 84 flags (u32, 0); 88..96 reserved (zero).  The payload follows the header.  A writer may append fields
/// (header_bytes > 96, readers skip to header_bytes); version 1 readers require at least 96.
constexpr size_t kContainerMagicBytes = 8;
constexpr uint8_t kContainerMagic[kContainerMagicBytes] = {'S', 'T', 'R', 'A', 'T', 'A', 'S', 'T'};
constexpr uint32_t kContainerVersion = 1;
constexpr uint32_t kContainerHeaderBytes = 96;

/// The raw llama sequence-state magic a legacy (pre-P2R-4a) SAVE file begins with - used to tell "an old
/// session file" apart from "some other file" in LOAD's errors.
constexpr uint32_t kRawStateMagic = 0xaf143cd8u;

void put_u32_le(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t) (v & 0xffu);
    p[1] = (uint8_t) ((v >> 8) & 0xffu);
    p[2] = (uint8_t) ((v >> 16) & 0xffu);
    p[3] = (uint8_t) ((v >> 24) & 0xffu);
}

void put_u64_le(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t) ((v >> (8 * i)) & 0xffu);
}

uint32_t get_u32_le(const uint8_t* p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

uint64_t get_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t) p[i] << (8 * i);
    return v;
}

/// CRC-32/ISO-HDLC (the zlib/PKZIP one: reflected, poly 0xEDB88320, init 0xFFFFFFFF, final xor
/// 0xFFFFFFFF) - the container payload checksum.  A table built once on first use.
uint32_t crc32_ieee(const uint8_t* data, size_t n) {
    static const struct Table {
        uint32_t t[256];
        Table() {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
                t[i] = c;
            }
        }
    } table;
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) c = table.t[(c ^ data[i]) & 0xffu] ^ (c >> 8);
    return c ^ 0xffffffffu;
}

/// SHA-256 (FIPS 180-4), self-contained: the model fingerprint is its only user.  `final` is terminal
/// (the object is not reusable afterwards).
struct Sha256 {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint64_t len = 0;      // message bytes fed so far
    uint8_t block[64] = {0};
    size_t fill = 0;

    static uint32_t ror(uint32_t v, int r) { return (v >> r) | (v << (32 - r)); }

    void compress(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
            0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
            0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
            0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
            0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
            0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
            0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
            0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = ((uint32_t) p[i * 4] << 24) | ((uint32_t) p[i * 4 + 1] << 16) |
                   ((uint32_t) p[i * 4 + 2] << 8) | (uint32_t) p[i * 4 + 3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            const uint32_t s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const uint8_t* data, size_t n) {
        len += n;
        while (n > 0) {
            const size_t take = std::min<size_t>(64 - fill, n);
            std::memcpy(block + fill, data, take);
            fill += take;
            data += take;
            n -= take;
            if (fill == 64) { compress(block); fill = 0; }
        }
    }

    void final(uint8_t out[32]) {
        const uint64_t bits = len * 8;
        uint8_t tail[72];
        size_t tail_len = 0;
        tail[tail_len++] = 0x80;
        while ((fill + tail_len) % 64 != 56) tail[tail_len++] = 0;
        for (int i = 7; i >= 0; --i) tail[tail_len++] = (uint8_t) (bits >> (8 * i));
        update(tail, tail_len);
        for (int i = 0; i < 8; ++i) {
            out[i * 4 + 0] = (uint8_t) (h[i] >> 24);
            out[i * 4 + 1] = (uint8_t) (h[i] >> 16);
            out[i * 4 + 2] = (uint8_t) (h[i] >> 8);
            out[i * 4 + 3] = (uint8_t) h[i];
        }
    }
};

/// P2R-4a: the model fingerprint - SHA-256 over <u64 file size LE> || first min(1 MiB, size) bytes ||
/// last min(1 MiB, size) bytes of the --qwen35-model file: exactly 2 MiB + 8 bytes are ever read, never
/// the whole 13.6 GB.  It is computed ONCE, right after the model load, and cached for the process's
/// life: a same-path replacement of the file after startup does not change the weights the running engine
/// holds, so the startup value stays the correct compatibility reference for SAVE/LOAD.  This is a
/// COMPATIBILITY check (different model / quantization / file length), not a security measure - the
/// expert pack is deliberately not part of it (documented in docs/backends/qwen35_llama_session_state.md).
struct ModelFingerprint {
    bool ok = false;
    uint8_t bytes[32] = {0};
};

ModelFingerprint compute_model_fingerprint(const std::string& path) {
    ModelFingerprint fp;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return fp;
    int64_t size = -1;
#if defined(_WIN32)
    if (_fseeki64(f, 0, SEEK_END) == 0) size = (int64_t) _ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END) == 0) size = (int64_t) ftello(f);
#endif
    if (size <= 0) {
        std::fclose(f);
        return fp;
    }
    const int64_t chunk = std::min<int64_t>(size, (int64_t) 1 << 20);
    std::vector<uint8_t> buf((size_t) chunk);
    Sha256 sha;
    uint8_t size_le[8];
    put_u64_le(size_le, (uint64_t) size);
    sha.update(size_le, sizeof(size_le));
    bool io_ok = true;
#if defined(_WIN32)
    io_ok = io_ok && _fseeki64(f, 0, SEEK_SET) == 0;
#else
    io_ok = io_ok && fseeko(f, 0, SEEK_SET) == 0;
#endif
    if (io_ok) io_ok = std::fread(buf.data(), 1, (size_t) chunk, f) == (size_t) chunk;
    if (io_ok) sha.update(buf.data(), (size_t) chunk);
#if defined(_WIN32)
    if (io_ok) io_ok = _fseeki64(f, size - chunk, SEEK_SET) == 0;
#else
    if (io_ok) io_ok = fseeko(f, size - chunk, SEEK_SET) == 0;
#endif
    if (io_ok) io_ok = std::fread(buf.data(), 1, (size_t) chunk, f) == (size_t) chunk;
    if (io_ok) sha.update(buf.data(), (size_t) chunk);
    const int close_rc = std::fclose(f);
    if (!io_ok || close_rc != 0) return fp;
    sha.final(fp.bytes);
    fp.ok = true;
    return fp;
}

/// The 8-byte (16 hex chars) fingerprint prefix STATE and the error texts print.
std::string fp_prefix_hex(const uint8_t* fp) {
    static const char* hex = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 8; ++i) {
        s += hex[fp[i] >> 4];
        s += hex[fp[i] & 0x0f];
    }
    return s;
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

/// P2R-4a: fill the fixed 96-byte container header.  The payload CRC32 is left zero here - the caller
/// writes it after `llama_state_seq_get_data` has filled the payload region (offset header_bytes).
void write_container_header(uint8_t* h, uint64_t payload_size, uint32_t n_vocab, uint32_t max_context,
                            uint64_t pos, const ModelFingerprint& fp) {
    std::memcpy(h, kContainerMagic, kContainerMagicBytes);
    put_u32_le(h + 8, kContainerVersion);
    put_u32_le(h + 12, kContainerHeaderBytes);
    put_u64_le(h + 16, payload_size);
    put_u64_le(h + 24, (uint64_t) kContainerHeaderBytes + payload_size);
    put_u32_le(h + 32, 0);   // payload_crc32: filled in after the payload is written
    put_u32_le(h + 36, n_vocab);
    put_u32_le(h + 40, max_context);
    put_u64_le(h + 44, pos);
    std::memcpy(h + 52, fp.bytes, sizeof(fp.bytes));
    put_u32_le(h + 84, 0);   // flags, reserved
    std::memset(h + 88, 0, kContainerHeaderBytes - 88);   // reserved to header_bytes
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
                  "\"state_keep_score\":true,\"session_continue\":true,"
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
    // P2R-4a: the model fingerprint, computed ONCE at startup and cached (never per request, never a
    // full-file hash).  Failure to read the model file back is a corner case (the model just loaded from
    // it); the engine keeps serving ordinary requests but refuses SAVE/LOAD, which need the fingerprint.
    const ModelFingerprint model_fp = compute_model_fingerprint(o.model);
    const std::string model_fp_prefix = model_fp.ok ? fp_prefix_hex(model_fp.bytes) : std::string();
    if (!model_fp.ok) {
        std::fprintf(stderr, "strata qwen35 serve: warning: cannot compute the session model fingerprint from "
                             "%s; SAVE/LOAD will refuse\n", o.model.c_str());
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
            js += session.active ? (session.from_load ? "\"load\""
                                    : (session.from_score_keep ? "\"score-keep\"" : "\"save\"")) : "null";
            js += ",\"file_bytes\":";
            js += (session.active && session.file_bytes > 0) ? std::to_string(session.file_bytes) : "null";
            // P2R-4a: the raw payload size inside the last session file; file_bytes is the container size.
            js += ",\"payload_bytes\":";
            js += (session.active && session.payload_bytes > 0) ? std::to_string(session.payload_bytes) : "null";
            js += ",\"max_context\":";
            js += std::to_string(total_ctx);
            // P2R-4a: the container version this build writes/reads and the model fingerprint prefix
            // (16 hex chars of the startup-cached SHA-256); both are build-level facts, not session facts.
            js += ",\"container_version\":";
            js += std::to_string(kContainerVersion);
            js += ",\"model_fp_prefix\":";
            js += model_fp_prefix.empty() ? std::string("null") : json_quote(model_fp_prefix);
            js += ",\"not_covered\":[\"logits\",\"sampler_rng_state\",\"expert_cache\",\"pack\","
                  "\"bridge_counters\",\"rollback_history\"]}";
            std::printf("STATE %s\n", js.c_str());
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("SAVE ", 0) == 0) {
            // P2-5 / P2R-4a: export the CURRENT sequence-0 state as the raw llama_state_seq bytes wrapped
            // in the 96-byte container (magic/version/lengths/CRC/fingerprint/vocab/context/pos).  On
            // success the session is armed: the next GEN appends at the cursor (continuation) instead of
            // resetting.  `SAVED`'s byte count is the CONTAINER file size (the pre-P2R-4a build reported
            // the raw payload size; the difference is exactly the 96-byte header).
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
            if (!model_fp.ok) {
                std::printf("ERR session: the model fingerprint is unavailable (the --qwen35-model file could "
                            "not be re-read); refusing to write a session container\n");
                std::fflush(stdout);
                continue;
            }
            const size_t payload_size = state_blob_size(ctx.get());
            if (payload_size == 0) {
                std::printf("ERR session: the state size walk failed\n");
                std::fflush(stdout);
                continue;
            }
            std::vector<uint8_t> container((size_t) kContainerHeaderBytes + payload_size, 0);
            const int64_t save_pos = state_cursor(ctx.get());
            write_container_header(container.data(), (uint64_t) payload_size, (uint32_t) n_vocab,
                                   (uint32_t) total_ctx, (uint64_t) save_pos, model_fp);
            const size_t got = llama_state_seq_get_data(ctx.get(), container.data() + kContainerHeaderBytes,
                                                        payload_size, 0);
            if (got != payload_size) {
                std::printf("ERR session: the state export returned %zu of %zu bytes\n", got, payload_size);
                std::fflush(stdout);
                continue;
            }
            put_u32_le(container.data() + 32, crc32_ieee(container.data() + kContainerHeaderBytes,
                                                         payload_size));
            if (!write_file_atomic(path, container.data(), container.size(), &derr)) {
                std::printf("ERR %s\n", derr.c_str());
                std::fflush(stdout);
                continue;
            }
            session.active = true;
            session.from_load = false;
            session.from_score_keep = false;
            session.path = path;
            session.file_bytes = container.size();
            session.payload_bytes = payload_size;
            std::printf("SAVED %zu %lld %.1f\n", container.size(), (long long) save_pos, ms_since(save_begin));
            std::fflush(stdout);
            std::fprintf(stderr, "serve: session save '%s' bytes=%zu pos=%lld payload=%zu model_fp=%s\n",
                         path.c_str(), container.size(), (long long) save_pos, payload_size,
                         model_fp_prefix.c_str());
            continue;
        }
        if (line.rfind("LOAD ", 0) == 0) {
            // P2-5 / P2R-4a: import a session container.  The container is validated first (fixed order,
            // every mismatch its own ERR - a failure here never touches the live session, not even an
            // armed one); only then do the payload bytes go into a TEMPORARY context (same model and
            // context parameters) and only a fully successful import is swapped in, so a bad path, a
            // foreign file or a corrupt payload leaves the live session untouched - never a half-restore.
            // `LOADED`'s byte count is the CONTAINER file size, like `SAVED` (the raw payload size is the
            // container's payload field and STATE's payload_bytes).
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
            if (!model_fp.ok) {
                std::printf("ERR session: the model fingerprint is unavailable (the --qwen35-model file could "
                            "not be re-read); refusing to import a session container\n");
                std::fflush(stdout);
                continue;
            }
            // The fixed validation order: file size -> magic -> version -> header length -> n_vocab ->
            // max_context -> pos -> payload CRC32 -> model fingerprint (see the session-state doc).  The
            // legacy/foreign classification on a missing magic keeps "an old raw state file" and "some
            // other file" apart; the payload's own llama header (magic + sequence id) is validated by
            // llama's reader inside the import below.
            if (blob.size() < kContainerHeaderBytes) {
                if (blob.size() >= 4 && get_u32_le(blob.data()) == kRawStateMagic) {
                    std::printf("ERR session: '%s' is a legacy raw state blob (no STRATAST container); re-SAVE "
                                "it with this build to migrate\n", path.c_str());
                } else {
                    std::printf("ERR session: '%s' is too small to be a STRATAST session container (%zu bytes; "
                                "the fixed header is %zu)\n", path.c_str(), blob.size(),
                                (size_t) kContainerHeaderBytes);
                }
                std::fflush(stdout);
                continue;
            }
            const bool container_magic_ok =
                std::memcmp(blob.data(), kContainerMagic, kContainerMagicBytes) == 0;
            const bool legacy_shape = get_u32_le(blob.data()) == kRawStateMagic;
            const uint64_t declared_file_bytes = get_u64_le(blob.data() + 24);
            if ((uint64_t) blob.size() != declared_file_bytes) {
                if (!container_magic_ok) {
                    if (legacy_shape) {
                        std::printf("ERR session: '%s' is a legacy raw state blob (no STRATAST container); "
                                    "re-SAVE it with this build to migrate\n", path.c_str());
                    } else {
                        std::printf("ERR session: '%s' is not a STRATAST session container (bad magic)\n",
                                    path.c_str());
                    }
                } else {
                    std::printf("ERR session: '%s' size mismatch (the file is %zu bytes, the container declares "
                                "%llu)\n", path.c_str(), blob.size(), (unsigned long long) declared_file_bytes);
                }
                std::fflush(stdout);
                continue;
            }
            if (!container_magic_ok) {
                if (legacy_shape) {
                    std::printf("ERR session: '%s' is a legacy raw state blob (no STRATAST container); re-SAVE "
                                "it with this build to migrate\n", path.c_str());
                } else {
                    std::printf("ERR session: '%s' is not a STRATAST session container (bad magic)\n",
                                path.c_str());
                }
                std::fflush(stdout);
                continue;
            }
            const uint32_t container_version = get_u32_le(blob.data() + 8);
            if (container_version != kContainerVersion) {
                std::printf("ERR session: '%s' has container version %u; this build reads version %u\n",
                            path.c_str(), container_version, kContainerVersion);
                std::fflush(stdout);
                continue;
            }
            const uint32_t header_bytes = get_u32_le(blob.data() + 12);
            const uint64_t payload_bytes = get_u64_le(blob.data() + 16);
            if (header_bytes < kContainerHeaderBytes || (uint64_t) header_bytes > (uint64_t) blob.size() ||
                payload_bytes != (uint64_t) blob.size() - (uint64_t) header_bytes) {
                std::printf("ERR session: '%s' has an invalid container header length (header %u, payload %llu, "
                            "file %zu)\n", path.c_str(), header_bytes, (unsigned long long) payload_bytes,
                            blob.size());
                std::fflush(stdout);
                continue;
            }
            const uint32_t saved_vocab = get_u32_le(blob.data() + 36);
            if ((uint64_t) saved_vocab != (uint64_t) n_vocab) {
                std::printf("ERR session: '%s' was saved for vocab %u; this model has %lld\n", path.c_str(),
                            saved_vocab, (long long) n_vocab);
                std::fflush(stdout);
                continue;
            }
            const uint32_t saved_ctx = get_u32_le(blob.data() + 40);
            if ((int64_t) saved_ctx != total_ctx) {
                std::printf("ERR session: '%s' was saved under context %u; this engine runs context %lld\n",
                            path.c_str(), saved_ctx, (long long) total_ctx);
                std::fflush(stdout);
                continue;
            }
            const uint64_t saved_pos = get_u64_le(blob.data() + 44);
            if (saved_pos > (uint64_t) total_ctx) {
                std::printf("ERR session: '%s' declares position %llu beyond the context (%lld)\n", path.c_str(),
                            (unsigned long long) saved_pos, (long long) total_ctx);
                std::fflush(stdout);
                continue;
            }
            const uint8_t* payload = blob.data() + header_bytes;
            const uint32_t saved_crc = get_u32_le(blob.data() + 32);
            const uint32_t computed_crc = crc32_ieee(payload, (size_t) payload_bytes);
            if (saved_crc != computed_crc) {
                std::printf("ERR session: '%s' payload CRC32 mismatch (container %08x, computed %08x)\n",
                            path.c_str(), saved_crc, computed_crc);
                std::fflush(stdout);
                continue;
            }
            if (std::memcmp(blob.data() + 52, model_fp.bytes, sizeof(model_fp.bytes)) != 0) {
                std::printf("ERR session: '%s' model fingerprint mismatch (container %s..., this engine %s...)\n",
                            path.c_str(), fp_prefix_hex(blob.data() + 52).c_str(), model_fp_prefix.c_str());
                std::fflush(stdout);
                continue;
            }
            llama_context* tmp = llama_init_from_model(model.get(), cp_saved);
            if (tmp == nullptr) {
                std::printf("ERR session: the temporary import context could not be created\n");
                std::fflush(stdout);
                continue;
            }
            const size_t got = llama_state_seq_set_data(tmp, payload, (size_t) payload_bytes, 0);
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
            session.from_score_keep = false;
            session.path = path;
            session.file_bytes = blob.size();
            session.payload_bytes = (size_t) payload_bytes;
            std::printf("LOADED %zu %lld %.1f\n", blob.size(), (long long) state_cursor(ctx.get()),
                        ms_since(load_begin));
            std::fflush(stdout);
            std::fprintf(stderr, "serve: session load '%s' bytes=%zu pos=%lld payload=%zu model_fp=%s\n",
                         path.c_str(), blob.size(), (long long) state_cursor(ctx.get()),
                         (size_t) payload_bytes, model_fp_prefix.c_str());
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
            // P2R-3a: `keep=1` appends at an ALREADY armed session's cursor without resetting (that is the
            // state-keeping score); with no armed session - keep or not - the request starts from 0 after
            // the usual reset.  `score_base_pos` is what the DONE `reused` field reports (0 without keep).
            const bool score_session = sreq.keep && session.active;
            const int64_t score_base_pos = score_session ? state_cursor(ctx.get()) : 0;
            if (score_base_pos + (int64_t) sreq.ids.size() > total_ctx) {
                if (score_session) {
                    std::printf("ERR session (%lld) + prompt (%lld tokens) exceeds the context (%lld)\n",
                                (long long) score_base_pos, (long long) sreq.ids.size(), (long long) total_ctx);
                } else {
                    std::printf("ERR prompt (%lld tokens) exceeds the context (%lld)\n",
                                (long long) sreq.ids.size(), (long long) total_ctx);
                }
                std::fflush(stdout);
                continue;
            }
            stdin_shared->stop.store(false);   // a STOP between requests is stale (the GEN rule)
            if (!score_session) {
                // No continuation: the same reset the default SCORE always did; an armed session ends here
                // (semantics (b) - a fresh keep request re-arms with origin `score-keep` after its answer).
                reset_session(ctx.get());
                session = SessionBook{};
            }
            const int64_t total = (int64_t) sreq.ids.size();
            const PrefillRun pre =
                run_prefill(ctx.get(), batch.get(), sreq.ids, chunk_max, stdin_shared, score_base_pos);
            if (pre.failed) {
                std::printf("ERR prefill failed\n");
                std::fflush(stdout);
                // A failed keep append resets the state and clears the session - a partial append is never
                // left for a continuation to pick up (the caller can LOAD a checkpoint again).
                if (sreq.keep) {
                    reset_session(ctx.get());
                    session = SessionBook{};
                }
                continue;
            }
            if (!pre.loan_returned) {
                std::printf("ERR the prompt workspace loan was not returned\n");
                std::fflush(stdout);
                if (sreq.keep) {
                    reset_session(ctx.get());
                    session = SessionBook{};
                }
                continue;
            }
            if (pre.cancelled) {
                // A STOP ended the read part-way: there is no complete last-position row to score.  The
                // carried positions (if any) are still reported honestly; the state is reset as always.
                std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld\n",
                            0LL, (long long) total, pre.prompt_ms, 0.0, "cancel", 0LL, 0LL,
                            (long long) score_base_pos, 0LL, 0LL, 0LL, 0LL, 0.0, (long long) pre.read_n, 0LL);
                std::fflush(stdout);
                std::fprintf(stderr, "serve: req %lld finish=cancel produced=0 prompt_ms=%.1f decode_ms=0.0 "
                                     "score%s\n",
                             ++serve_reqs, pre.prompt_ms, sreq.keep ? " keep=1" : "");
                reset_session(ctx.get());
                if (sreq.keep) session = SessionBook{};
                continue;
            }
            const float* logits = llama_get_logits_ith(ctx.get(), -1);
            if (logits == nullptr) {
                std::printf("ERR score produced no logits\n");
                std::fflush(stdout);
                if (sreq.keep) {
                    reset_session(ctx.get());
                    session = SessionBook{};
                }
                continue;
            }
            if (logits_have_nan(logits, n_vocab)) {
                std::fprintf(stderr, "strata qwen35 serve: the head logits are not all finite; answering ERR "
                                     "(score)\n");
                std::printf("ERR score: the head logits are not all finite\n");
                std::fflush(stdout);
                reset_session(ctx.get());
                if (sreq.keep) session = SessionBook{};
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
                        0LL, (long long) total, pre.prompt_ms, 0.0, "score", 0LL, 0LL,
                        (long long) score_base_pos /* reused */, 0LL, 0LL, 0LL, 0LL, 0.0,
                        (long long) pre.read_n, 0LL);
            std::fflush(stdout);
            if (sreq.cand_present) {
                std::fprintf(stderr, "serve: req %lld finish=score produced=0 prompt_ms=%.1f decode_ms=0.0 "
                                     "topk=%lld cand=%lld%s\n",
                             ++serve_reqs, pre.prompt_ms, (long long) k, (long long) sreq.cand.size(),
                             sreq.keep ? " keep=1" : "");
            } else {
                std::fprintf(stderr, "serve: req %lld finish=score produced=0 prompt_ms=%.1f decode_ms=0.0 "
                                     "topk=%lld%s\n",
                             ++serve_reqs, pre.prompt_ms, (long long) k, sreq.keep ? " keep=1" : "");
            }
            if (sreq.keep) {
                // P2R-3a: a successful keep request does NOT reset - the session stays armed at the new
                // cursor.  It armed the session itself when none was active (origin `score-keep`); when it
                // continued an armed session, that session keeps its origin and file fields (like a
                // session GEN - continuing never re-arms).
                if (!score_session) {
                    session.active = true;
                    session.from_load = false;
                    session.from_score_keep = true;
                }
            } else {
                reset_session(ctx.get());
            }
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: CAPS | STATE | SAVE <path> | LOAD <path> | "
                        "SCORE <id,id,...> [topk=N] [cand=id,id,...] [keep=1] | "
                        "GEN <max_new> [key=value ...] [cont=1] <id,id,...>\n");
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
        if (req.cont && !session.active) {
            // P2R-3a: `cont=1` is an assertion that there IS a session to continue - a continuation
            // without one would be an isolated request, i.e. a different request than the caller meant.
            std::printf("ERR cont=1 needs an armed session\n");
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
        // P2-5: after a successful SAVE/LOAD (or a P2R-3a `SCORE keep=1`) the session is armed and this GEN
        // APPENDS its prompt at the cursor (the continuation); without a session the request is isolated
        // exactly as before - fresh state from position 0.  `base_pos` is also what the DONE `reused` field
        // reports.  P2R-3a `cont=1` does not change this path at all: it only asserted above that a session
        // IS armed and (in the parser) allowed an empty id list (pure continuation, `total` = 0).
        const bool session_gen = session.active;
        const int64_t base_pos = session_gen ? state_cursor(ctx.get()) : 0;
        // Subtract only after each operand has been bounded: adding LLONG_MAX to a
        // carried cursor would otherwise overflow before the context guard ran.
        if (base_pos < 0 || base_pos > total_ctx || req.ids.size() > (uint64_t) (total_ctx - base_pos) ||
            req.max_new > total_ctx - base_pos - (int64_t) req.ids.size()) {
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
                const bool is_eog = llama_vocab_is_eog(vocab, tok);
                // An armed session commits EVERY returned token, including the last
                // length token and EOG. The next request must read its successor logits,
                // not sample the previous row again. Isolated requests keep their old
                // evaluation count; SAVE/LOAD needs no new pending-token container field.
                if (session_gen || (!is_eog && i + 1 < req.max_new)) {
                    if (!fill_batch(batch.get(), &tok, 1, (int) (base_pos + total + i)) ||
                        llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                        std::printf("ERR decode failed at output %lld\n", produced);
                        std::fflush(stdout);
                        finish = nullptr;
                        break;
                    }
                }
                if (is_eog) { finish = "stop"; break; }
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
            std::fprintf(stderr, "serve: req %lld finish=%s produced=%lld prompt_ms=%.1f decode_ms=%.1f%s\n",
                         ++serve_reqs, finish, produced, pre.prompt_ms, decode_ms, req.cont ? " cont=1" : "");
        }
    }
    core::qwen35_bridge_shutdown();
    return 0;
}

}  // namespace strata::program
