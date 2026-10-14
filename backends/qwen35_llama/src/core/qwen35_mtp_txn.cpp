// src/core/qwen35_mtp_txn.cpp - task 257/259: greedy K=1 MTP transaction engine (default OFF diagnostic).
//
// Task 268 candidate (STRATA_QWEN35_MTP_SERIAL_VERIFY=1, default OFF): the target verify of a round becomes a
// serial of token-only T1 decodes instead of one T2 batch - T1#1 decodes the pending token alone, its t0/h0
// are read and copied immediately; only when draft_id == t0 (and the draft token is legal, i.e. not EOG) a
// second T1#2 decodes the draft token alone and its t1/h1 are copied immediately.  On reject no speculative
// target row was ever decoded, so the target sequence is already clean (nothing to seq_rm); the corrective is
// still t0/h0.  On accept the real draft token + hidden replay and every seed/bonus/EOS/max_new/context-tail
// rule are unchanged.  OFF = the previous T2 path, unchanged.
//
// Task 273 (default OFF, isolated copy under results/phase2_mtp/reject273/source): diagnostic fault
// injection.  STRATA_QWEN35_MTP_FAULT_INJECT=1/2/3 replaces the greedy draft token BEFORE the target
// verify (pred+1 modulo vocabulary, skipping EOG), so the speculative row, decide, rollback and replay all
// operate on the injected token - a reject is a real t0 != injected mismatch, never a comparison-value
// edit.  Every round logs inject/original_draft/injected_id/injected_mismatch; any active mode marks the
// run diagnostic (never product speed or accept rate).  0/unset = previous behavior untouched.
//
// Task 259 fixes (CODEX_REVIEW_256_257_258):
//   1. The accepted-prefix draft replay is token+hidden: the accepted token at pos+1 is decoded into the
//      draft context together with a deep copy of target verify row0 (hidden after the pending token).
//      Token-only replay would not provide h (the vendor graph_mtp always allocates an h input and its
//      set_input fills it only when a row carries embeddings).  The next pending hidden is verify row1.
//      Only the real accepted token is ever replayed; no future/fabricated draft token is.
//   2. One output owner: this engine emits every visible greedy token exactly once, in order - the
//      prompt-sampled seed, the accepted draft, the accepted bonus / rejected corrective pending, and the
//      context-tail serial emit.  The caller no longer emits anything for a txn round.
//   3. max_new is the visible-token budget; rounds/EOS are accounted separately.  EOS stops without
//      emitting the EOG token and without replaying it; pos >= n_ctx fails before any decode; only one
//      visible-token budget remaining does one legal final serial/round-start emit and stops.
//
// Task 309 (exact STRATA_QWEN35_MTP_VERIFY_PROFILE_RANGE=1, default OFF; source preparation only, runtime
// UNTESTED in this batch): one bounded CUDA profiler range around the designated round (index 3, zero-based)
// of the SECOND serial transaction invocation (the registered request ordinal; the caller runs one
// qwen35_mtp_txn_run per request with STRATA_QWEN35_MTP_TXN_REQUESTS=2).  cudaProfilerStart is called
// immediately before that round's actual target T2 decode of [pending, real greedy draft]; cudaProfilerStop
// after the existing target logits/nextn hidden readiness and both deep copies - before decide/printf/
// replay/rollback - so the stop lands on an EXISTING readiness instant and adds no device synchronization.
// The window can still contain upstream draft/replay GPU work in flight at entry: it is a host-inclusive
// TARGET window, never a pure kernel cost and never the whole model lifetime (spatially 40 layers x 2 rows =
// 80 layer-rows, temporally one bounded round).  The flag is fail-closed: any contract break (K != 1, serial
// verify, legacy TWO_REQUEST / T1 profile, fault/rollback/matched/accept oracle, SMALL_T_PHASES,
// COST_SPLIT/COST_SYNC, MTP_TXN_REQUESTS != 2) refuses before the first transaction step instead of
// silently switching.  One bounded JSON record (qwen35_mtp_txn_verify_profile_range) reports the invocation
// ordinal, round, K/T, actual pending/draft tokens and positions, the exact start/stop call state and
// errors, designated-stop and host-elapsed; a designated round that never runs is recorded as
// round_not_reached, never faked.  Flag OFF: no profiler call, no event, no clock read - a couple of local
// flag comparisons per T2 round remain (no zero-instruction claim).
//
// Task 317 (exact STRATA_QWEN35_MTP_DRAFT_STATE_ORACLE=1 plus STRATA_QWEN35_MTP_DRAFT_ORACLE_CASE=accept|reject,
// default OFF; diagnostic only): the ACTUAL drafter-context state oracle.  On the warm round INDEX 1 (zero
// based) of the request, BEFORE that round's real propose, it snapshots the DRAFT context's real serialized
// sequence state (public llama_state_seq_*; PRE) and deep-copies the pending token / position / pending_h.
// The round then runs its REAL propose -> target T2 -> decide exactly as usual.  The CASE must match the real
// branch (accept = natural accept; reject = real injected reject under FAULT_INJECT mode 3) or the run fails
// closed with an explicit NO_GO - no manufactured input, no forced token.  At the branch endpoint ONE real
// next-consumer propose decode of the actual next pending (t1@pos+2 with target h_row1 on accept, t0@pos+1
// with target h_row0 on reject) captures the ACT consumer logits/hidden and the serialized draft state ACT;
// then REF restores PRE on the SAME draft context and replays only the same committed inputs (pending@pos
// with the captured pending_h, the accepted token@pos+1 with target h_row0 on accept - logits unused exactly
// like the real replay - then the same consumer), capturing REF logits/hidden/state.  Gates: real head getter
// finite, greedy equal, head/hidden within strict 1e-5 abs AND rel, and raw serialized draft state
// BYTE_EQUAL (mandatory for scope PASS; a TYPED_COMPARED number is never a PASS).  The request stops with
// stop=draft_state_oracle_probe; the ACT state is NEVER restored and no continued request / product speed is
// claimed.  It runs only under its own exact env gate plus the case env, refuses every conflict (K != 1,
// serial verify, target rollback/matched/accept oracles, COST_SPLIT/COST_SYNC, SMALL_T_PHASES, T1/verify
// profiler, legacy TWO_REQUEST), requires STRATA_QWEN35_MTP_ORACLE_DIR and fails before any oversize
// allocation (128 MiB per snapshot, 384 MiB live).  This only tests the draft consumption/replay/state
// contract - not model semantics, the full context tail or K2.  Flag OFF allocates nothing and runs no state
// API; a few local flag comparisons per round remain (no zero-instruction claim).
//
// Task 320 (exact STRATA_QWEN35_MTP_T3_VERIFY_ORACLE=1, default OFF; diagnostic only): the single warm-round
// (INDEX 1) FULL-GRAPH T3 target verification oracle.  At that round, before its normal propose, the step is
// replaced by one isolated comparison: PRE = the target context's serialized state after the real K1 round 0
// (public llama_state_seq_get_data); REF = three REAL serial token-only T1 decodes of the natural greedy
// prefix - pending@pos, then s1=argmax@pos+1, then s2=argmax@pos+2 - with every row's logits and nextn hidden
// deep-copied and the final serialized state saved; then the SAME PRE is restored (exact returned size) and
// ACT = ONE real token-only T3 batch of the same [pending, s1, s2] (all three rows output logits), with its
// three rows and final state captured.  Gates: per-row greedy equal, finite, fixed 1e-5 max-abs AND L1-rel
// (row0/row1/row2 EACH - a global history aggregate never dilutes a new row), and the final ACT/REF
// serialized target state BYTE_EQUAL with the typed GDN/KV layout force-walked (a byte-equal state whose
// layout the parser cannot prove is an explicit PARTIAL, never a silent PASS).  The request stops with
// stop=t3_verify_oracle_probe; nothing is restored to fake a continued request and no product speed, real
// draft acceptance or K2 is claimed.  Fail-closed contract: K1, TXN_REQUESTS=1, TXN=1, SMALL_T_NATIVE=1,
// SMALL_T_MULTI=1, SLICE_GU_MULTI=1, COMPACT_SLOTS=1, FAULT mode 0, no other oracle/profiler/COST/PHASES/
// TWO_REQUEST, n_rs_seq>=2 on the target context, and STRATA_QWEN35_MTP_ORACLE_DIR for the three PRE/REF/ACT
// exports; 192 MiB per target snapshot / 512 MiB live host caps are checked before every allocation.  A
// parser the walk cannot prove reports an explicit PARTIAL verdict.  Flag OFF adds a couple of local flag
// comparisons only - no allocation, no state API, no diagnostic math.
//
// Task 323 (exact STRATA_QWEN35_MTP_T3_RESTORE_SERIAL_CONTROL=1, default OFF; valid ONLY together with
// T3_VERIFY_ORACLE=1, fail-closed inside the transaction otherwise): the restore-vs-live serial control for
// the actual 322 T3 numeric FAIL.  The whole PRE/REF chain is UNCHANGED (real PRE snapshot after the K1
// warm round 0, PRE exported, three real serial token-only T1 decodes of the natural greedy prefix as REF
// with their logits/hidden/state captured, PRE restored exactly).  With the flag ON the ACT capture is no
// longer one T3 batch: it is THREE REAL serial token-only T1 decodes of the IDENTICAL immutable
// [pending, s1, s2] at pos / pos+1 / pos+2 (no re-chaining by argmax), and every row's logits/nextn hidden
// are deep-copied IMMEDIATELY from its own call (ith 0) - never re-read from another ith row and never
// overwritten by a common batch extraction.  This isolates the one open question of 322: whether restoring
// PRE reproduces the same three serial T1 references (a control fail keeps the restoration/starting state
// suspect - the T3 kernel is never blamed first).  Gates unchanged (per-row greedy/finite/fixed 1e-5
// max-abs AND L1-rel, serialized state BYTE_EQUAL, proven parse, nonempty typed coverage with zero
// non-finite).  The PASS verdict is RESTORE_SERIAL_CONTROL_PASS; every fail/PARTIAL verdict is explicitly
// CONTROL_*.  The frame reports probe_kind=restore_serial_control, batch calls 3 / rows_per_call 1 /
// outputs_per_call 1 / captured_rows 3 and api_calls ref_t1_decodes 3 / act_t1_decodes 3 / act_t3_decodes
// 0 / state_saves 3 / state_restores 1.  The diagnostic stop is restore_serial_control; the final ACT
// state is NOT restored and no continued request, product speed, real draft acceptance or K2 is claimed.
// Flag OFF keeps the original one-T3-batch ACT decode and its three-row extraction untouched.
#include "strata/core/qwen35_mtp_txn.hpp"

#include "strata/core/qwen35_bridge.hpp"

#include "llama-ext.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Task 329 private diagnostic seam (defined in src/program/qwen35_model.cpp; no public header): the bounded
// five-tensor attention-slice probe's phase setter.  Declared here at namespace scope so the calls below
// resolve to the definition in the program translation unit.
namespace strata::core { void qwen35_t3_attn_probe_phase(int phase, int row); }

namespace strata::core {
namespace {

std::vector<float> copy_nextn_row(llama_context* tgt, int32_t row, bool* ok) {
    *ok = false;
    const int64_t n_embd_out = (int64_t) llama_model_n_embd_out(llama_get_model(tgt));
    const float* src = llama_get_embeddings_nextn_ith(tgt, row);
    if (src == nullptr || n_embd_out <= 0) return {};
    std::vector<float> out((size_t) n_embd_out);
    std::memcpy(out.data(), src, (size_t) n_embd_out * sizeof(float));
    for (float v : out) {
        if (!std::isfinite(v)) return {};
    }
    *ok = true;
    return out;
}

llama_token greedy_argmax(const float* logits, int64_t n_vocab, bool* ok) {
    *ok = false;
    if (logits == nullptr || n_vocab <= 0) return -1;
    int64_t best = -1;
    double best_v = 0.0;
    for (int64_t i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(logits[i])) return -1;
        if (best < 0 || logits[i] > best_v) { best = i; best_v = logits[i]; }
    }
    *ok = true;
    return (llama_token) best;
}

bool decode_rows(llama_context* ctx, const llama_token* ids, const std::vector<float>* embd_row, int64_t n_rows,
                 int64_t first_pos, bool all_outputs, int32_t* rc_out) {
    llama_batch_ext* b = llama_batch_ext_init(ctx);
    if (b == nullptr) return false;
    bool ok = true;
    for (int64_t r = 0; r < n_rows && ok; ++r) {
        const int32_t idx = llama_batch_ext_add_token(b, 0, ids[r]);
        const llama_pos pos = (llama_pos) (first_pos + r);
        ok = idx >= 0 && llama_batch_ext_set_pos(b, idx, &pos);
        if (ok && embd_row != nullptr && r == 0) {
            const llama_embd e = {embd_row->data(), 1, embd_row->size()};
            ok = llama_batch_ext_set_embd_token(b, idx, e);
        }
        if (ok && (all_outputs || r == n_rows - 1)) ok = llama_batch_ext_set_output_logits(b, idx, true);
    }
    int32_t rc = -1;
    if (ok) rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b);
    llama_batch_ext_free(b);
    if (rc_out != nullptr) *rc_out = rc;
    return ok;
}

void print_piece_hex(const llama_vocab* vocab, llama_token id, std::string* out) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    out->clear();
    for (int i = 0; i < (n > 0 ? n : 0); ++i) {
        char b[3];
        std::snprintf(b, sizeof(b), "%02x", (unsigned char) buf[i]);
        out->append(b);
    }
}

// Task 273: EOG predicate adapter so the pure injection mapping stays llama-call-free and unit-testable.
bool inject_eog_cb(void* user, llama_token id) {
    return llama_vocab_is_eog((const llama_vocab*) user, id);
}

// ================= Task 309: bounded profiler range owner (default OFF, source preparation only) ===========
// cudaProfilerStart/Stop have no visible declaration in this translation unit: the fixed CUDA 12.8.1 runtime
// headers available to the engine do not declare them (no cuda_profiler_api.h; the read-only header evidence
// is recorded with qwen35_model.cpp's task-183 declarations).  qwen35_model.cpp declares them as
// `extern "C" cudaError_t cudaProfilerStart(void)` / `cudaProfilerStop` and the exe links them from the
// shared CUDA runtime it already uses.  A cudaError_t is an int-sized enum, so `int` here is the same x64
// C ABI and the same exported symbol; 0 == success.  No new DLL/library/tool is installed for this.
extern "C" int cudaProfilerStart(void);
extern "C" int cudaProfilerStop(void);

// Exactly-once Start/Stop owner for ONE bounded range (the designated round's actual target T2).  States:
//   requested=false                      -> completely inert: no call, no event, no clock read.
//   requested, designated round not run  -> destructor seals; record status=round_not_reached (not faked).
//   requested, start failed              -> no Stop at all (a failed start must not expect one); the run
//                                           keeps its original behavior; record status=start_failed (NO_GO).
//   requested, started, early exit/error -> destructor issues the single safety Stop; early_exit_after_start.
//   requested, started, registered point -> Stop lands after the existing readiness/copies; range_complete.
// One bounded JSON record per registered run is emitted exactly once (at the registered close, or by the
// destructor otherwise); it reports the actual call/stop state, errors, designated-stop and host-elapsed,
// and labels the window host-inclusive (upstream draft/replay GPU work can still be in flight at entry -
// the record is never presented as pure kernel cost).  No Stop without a successful Start; never a double
// Stop; nothing here synchronizes or changes a decision, an output or the math.
struct VerifyProfileRange {
    bool requested = false;              // flag ON and this invocation is the registered ordinal (2)
    bool start_called = false;
    bool started = false;
    bool stop_called = false;
    bool stopped = false;
    bool designated_stop_reached = false;
    bool round_reached = false;
    bool event_emitted = false;
    int start_error = 0;                 // cudaError_t value of the start call (0 == success)
    int stop_error = 0;                  // cudaError_t value of the stop call  (0 == success)
    int64_t ordinal = 0;                 // per-process invocation ordinal when the flag was on
    int64_t designated_round = -1;
    int64_t round = -1;
    int64_t k = 0;
    int64_t t_rows = 0;
    int64_t pending = -1;
    int64_t draft = -1;
    int64_t pending_pos = -1;
    int64_t draft_pos = -1;
    double start_s = -1.0;               // engine-relative host steady-clock instants (same clock)
    double stop_s = -1.0;

    /// Call immediately before the designated round's actual target T2 decode.  No-op for every other run
    /// and round.
    void try_start(double now_s, int64_t round_idx, int64_t k_val, int64_t pos, llama_token pend,
                   llama_token dr) {
        if (!requested || start_called || round_idx != designated_round) return;
        start_called = true;
        round_reached = true;
        round = round_idx;
        k = k_val;
        t_rows = 2;                      // the actual T2 batch [pending, real greedy draft]
        pending = (int64_t) pend;
        draft = (int64_t) dr;
        pending_pos = pos;
        draft_pos = pos + 1;
        start_s = now_s;
        const int e = cudaProfilerStart();
        start_error = e;
        started = (e == 0);
    }

    /// The registered close point: after the existing logits/nextn readiness and both deep copies, before
    /// decide/emit/replay/rollback.  Idempotent; never issued after a failed start.
    void stop_designated(double now_s) {
        if (!started || stopped) return;
        stop_called = true;
        stop_s = now_s;
        stop_error = cudaProfilerStop();
        stopped = true;
        designated_stop_reached = (stop_error == 0);
    }

    const char* status() const {
        if (!requested) return "not_requested";
        if (!round_reached) return "round_not_reached";
        if (!started) return "start_failed";
        if (!designated_stop_reached) return "early_exit_after_start";
        return "range_complete";
    }

    void emit_event() {
        if (!requested || event_emitted) return;
        event_emitted = true;
        const double host_elapsed_ms =
            (start_s >= 0.0 && stop_s >= 0.0) ? (stop_s - start_s) * 1000.0 : -1.0;
        std::printf(
            "{\"event\":\"qwen35_mtp_txn_verify_profile_range\",\"process_invocation\":%lld,"
            "\"request_ordinal\":%lld,\"designated_round_index\":%lld,\"round_reached\":%s,"
            "\"round\":%lld,\"k\":%lld,\"t_rows\":%lld,\"start_called\":%s,\"started\":%s,"
            "\"start_error\":%d,\"stop_called\":%s,\"stopped\":%s,\"stop_error\":%d,"
            "\"designated_stop\":%s,\"host_elapsed_ms\":%.3f,\"pending\":%lld,\"draft\":%lld,"
            "\"pending_pos\":%lld,\"draft_pos\":%lld,\"status\":\"%s\","
            "\"note\":\"host-inclusive target T2 window; upstream draft/replay GPU work may be in flight "
            "at entry; not pure kernel cost\"}\n",
            (long long) ordinal, (long long) ordinal, (long long) designated_round,
            round_reached ? "true" : "false", (long long) round, (long long) k, (long long) t_rows,
            start_called ? "true" : "false", started ? "true" : "false", start_error,
            stop_called ? "true" : "false", stopped ? "true" : "false", stop_error,
            designated_stop_reached ? "true" : "false", host_elapsed_ms, (long long) pending,
            (long long) draft, (long long) pending_pos, (long long) draft_pos, status());
        std::fflush(stdout);
    }

    ~VerifyProfileRange() {
        if (started && !stopped) {       // single safety seal for every early return/error after a start
            stop_called = true;
            stop_error = cudaProfilerStop();
            stopped = true;
        }
        emit_event();
    }
};

// ================= Task 277: rollback state oracle helpers (default OFF, diagnostic only) =================
// The oracle is gated by the exact STRATA_QWEN35_MTP_ROLLBACK_ORACLE=1 and only runs inside the T2 verify
// path on a real injected reject round.  It changes no default-path decision: it captures the pre-verify
// logical sequence state (public llama_state_seq_*), runs the real seq_rm rollback and the real next
// corrective decode (ACT), restores the pre state and replays the same committed token prefix serially
// (REF), then compares logits, nextn hidden and the serialized sequence state numerically.  State bytes
// are walked with their own self-describing typed layout; raw padding/slot ordering is never by itself
// called pollution and a parse that cannot be proven is reported as PENDING, never as PASS.
const size_t ORACLE_SIMUL_CAP_BYTES  = (size_t) 512 * 1024 * 1024;   // two live host blocks max
const size_t ORACLE_EXPORT_CAP_BYTES = (size_t) 512 * 1024 * 1024;   // exported .bin total
const int    ORACLE_MAX_PROBES       = 2;                            // per process (one per request)
const uint32_t ORACLE_STATE_IO_MAGIC = 0xaf143cd8u;                  // llama_context::state_seq_get_data io_magic

// Persistent per-process oracle accounting (the engine runs once per transaction request).
int      g_oracle_probes_done = 0;
uint64_t g_oracle_export_bytes = 0;

// Task 317 draft-state oracle bounds (own constants; the 277/294 target-oracle numbers above are untouched).
// Per snapshot 128 MiB, live host snapshot total 384 MiB - both checked BEFORE any allocation.
const size_t DRAFT_ORACLE_STATE_CAP_BYTES = (size_t) 128 * 1024 * 1024;
const size_t DRAFT_ORACLE_LIVE_CAP_BYTES  = (size_t) 384 * 1024 * 1024;
int g_draft_oracle_probes_done = 0;

// Task 320 T3 verification-oracle bounds (own constants; the 277/294/317 numbers above are untouched).
// Per target snapshot 192 MiB, live host snapshot total 512 MiB - both checked BEFORE any allocation.
const size_t T3_ORACLE_STATE_CAP_BYTES = (size_t) 192 * 1024 * 1024;
const size_t T3_ORACLE_LIVE_CAP_BYTES  = (size_t) 512 * 1024 * 1024;
int g_t3_oracle_probes_done = 0;

struct OracleRowDiff {
    int64_t n = 0;          // elements compared
    int64_t neq = 0;        // element blocks whose raw bytes differ
    int64_t nonfinite = 0;  // elements where either side is not finite
    double max_abs = 0.0;   // max |a - b|
    double l1 = 0.0;        // sum |a - b|
    double sumsq = 0.0;     // sum (a - b)^2 (RMS at print time)
    double ref_l1 = 0.0;    // sum |b| (relative denominator)
};

bool oracle_float_type(int32_t t, size_t* esize, const char** name) {
    switch (t) {
        case 0:  *esize = 4; *name = "F32";  return true;
        case 1:  *esize = 2; *name = "F16";  return true;
        case 30: *esize = 2; *name = "BF16"; return true;
        default: *esize = 0; *name = "raw";  return false;
    }
}

float oracle_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31;
    const uint32_t exp  = (uint32_t) (h >> 10) & 0x1fu;
    const uint32_t man  = (uint32_t) h & 0x3ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = man;
            do { ++e; m <<= 1; } while ((m & 0x400u) == 0);
            bits = sign | ((uint32_t) (127 - 15 - e) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float oracle_elem_f32(const uint8_t* p, int32_t t) {
    if (t == 0) {
        float f;
        std::memcpy(&f, p, 4);
        return f;
    }
    uint16_t h;
    std::memcpy(&h, p, 2);
    if (t == 1) return oracle_f16_to_f32(h);
    const uint32_t bits = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

OracleRowDiff oracle_diff_elems(const uint8_t* a, const uint8_t* b, int64_t n, int32_t t, size_t esize) {
    OracleRowDiff d;
    d.n = n;
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t* pa = a + (size_t) i * esize;
        const uint8_t* pb = b + (size_t) i * esize;
        if (std::memcmp(pa, pb, esize) != 0) ++d.neq;
        const double fa = (double) oracle_elem_f32(pa, t);
        const double fb = (double) oracle_elem_f32(pb, t);
        if (!std::isfinite(fa) || !std::isfinite(fb)) ++d.nonfinite;
        const double diff = fa - fb;
        const double ad = diff < 0 ? -diff : diff;
        if (ad > d.max_abs) d.max_abs = ad;
        d.l1 += ad;
        d.sumsq += diff * diff;
        d.ref_l1 += fb < 0 ? -fb : fb;
    }
    return d;
}

OracleRowDiff oracle_diff_bytes(const uint8_t* a, const uint8_t* b, int64_t nbytes) {
    OracleRowDiff d;
    for (int64_t i = 0; i < nbytes; ++i) {
        ++d.n;
        if (a[i] != b[i]) ++d.neq;
    }
    d.max_abs = -1.0;   // not a float comparison
    return d;
}

void oracle_accum(OracleRowDiff* acc, const OracleRowDiff& d) {
    acc->n += d.n;
    acc->neq += d.neq;
    acc->nonfinite += d.nonfinite;
    if (d.max_abs > acc->max_abs) acc->max_abs = d.max_abs;
    acc->l1 += d.l1;
    acc->sumsq += d.sumsq;
    acc->ref_l1 += d.ref_l1;
}

struct OracleReader {
    const uint8_t* base = nullptr;
    size_t size = 0;
    size_t off = 0;
    bool bad = false;

    template <typename T>
    T rd() {
        T v{};
        if (bad || off + sizeof(T) > size) { bad = true; return v; }
        std::memcpy(&v, base + off, sizeof(T));
        off += sizeof(T);
        return v;
    }

    const uint8_t* ptr(size_t n) {
        if (bad || n > size || off + n > size) { bad = true; return nullptr; }
        const uint8_t* p = base + off;
        off += n;
        return p;
    }
};

struct OracleTensor {
    std::string name;
    const char* dtype = "raw";
    OracleRowDiff d;
};

struct OracleCellMeta {
    int32_t pos = 0;
    uint32_t nseq = 0;
    std::vector<int32_t> seqs;
    uint8_t ext[12] = {0};
};

bool oracle_read_cells(OracleReader& r, uint32_t count, bool with_ext, std::vector<OracleCellMeta>* out) {
    out->clear();
    out->reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        OracleCellMeta m;
        m.pos = r.rd<int32_t>();
        m.nseq = r.rd<uint32_t>();
        if (r.bad || m.nseq > 64) return false;
        if (with_ext) {
            const uint8_t* e = r.ptr(12);
            if (e == nullptr) return false;
            std::memcpy(m.ext, e, 12);
        }
        m.seqs.resize(m.nseq);
        for (uint32_t j = 0; j < m.nseq; ++j) m.seqs[j] = r.rd<int32_t>();
        if (r.bad) return false;
        out->push_back(std::move(m));
    }
    return true;
}

// Compare the per-cell metadata of both streams.  Returns: 0 = same order, 1 = same pos set but reordered,
// 2 = metadata mismatch (not equal), and fills idxb_for_a (index in B for each A index) when reordered.
int oracle_match_cells(const std::vector<OracleCellMeta>& ca, const std::vector<OracleCellMeta>& cb,
                       std::vector<size_t>* idxb_for_a, std::string* note) {
    const size_t n = ca.size();
    if (cb.size() != n) {
        *note = "cell count differs";
        return 2;
    }
    bool same_order = true;
    for (size_t i = 0; i < n && same_order; ++i) {
        same_order = ca[i].pos == cb[i].pos && ca[i].nseq == cb[i].nseq &&
                     ca[i].seqs == cb[i].seqs && std::memcmp(ca[i].ext, cb[i].ext, sizeof(ca[i].ext)) == 0;
    }
    if (same_order) {
        idxb_for_a->resize(n);
        for (size_t i = 0; i < n; ++i) (*idxb_for_a)[i] = i;
        return 0;
    }
    std::unordered_map<int32_t, size_t> posb;
    for (size_t i = 0; i < n; ++i) {
        if (!posb.emplace(cb[i].pos, i).second) {
            *note = "duplicate cell position in B";
            return 2;
        }
    }
    idxb_for_a->resize(n);
    for (size_t i = 0; i < n; ++i) {
        const auto it = posb.find(ca[i].pos);
        if (it == posb.end()) {
            *note = "cell position missing in B";
            return 2;
        }
        (*idxb_for_a)[i] = it->second;
        const OracleCellMeta& ma = ca[i];
        const OracleCellMeta& mb = cb[it->second];
        if (ma.nseq != mb.nseq || ma.seqs != mb.seqs || std::memcmp(ma.ext, mb.ext, sizeof(ma.ext)) != 0) {
            *note = "cell metadata differs at pos " + std::to_string(ma.pos);
            return 2;
        }
    }
    return 1;
}

// Structural walk + typed payload comparison of the two serialized sequence states, streamed in lockstep.
// The only ambiguous parts are the optional per-cell ext blob (global 2-way choice) and the null-layer
// pattern of the recurrent part (bounded DFS on both streams, end-exact).  Returns 0 = compared,
// 1 = structural divergence (note + where), 2 = could not prove a parse (PENDING decoder).
// Task 320 adds the OPTIONAL force_parse parameter (default false - every earlier caller is unchanged):
// with it, equal bytes still go through the full typed structural walk, so a byte-equal verdict carries a
// proven walk of the actual serialized layout (or comes back as 2/PENDING when the parser cannot prove it).
int oracle_compare_state_bytes(const std::vector<uint8_t>& A, const std::vector<uint8_t>& B,
                               std::vector<OracleTensor>* tensors, std::string* note,
                               int64_t* first_diff, int64_t* n_diff_bytes, bool* byte_equal,
                               bool force_parse = false) {
    *byte_equal = false;
    if (first_diff) *first_diff = -1;
    if (n_diff_bytes) *n_diff_bytes = 0;
    if (A.size() != B.size()) {
        *note = "size mismatch";
        return 2;
    }

    // Vendor io header (llama_context::state_seq_get_data): u32 io_magic 0xaf143cd8 then the i32 seq_id
    // (0 for the oracle's seq 0).  Both streams are verified before any payload interpretation - also on
    // the byte-equal path: equal bytes never skip the format check.  A wrong/missing header (or a
    // non-seq-0 id) stays PENDING (fail-closed) exactly like the offline 277 parser.
    if (A.size() < 8) {
        *note = "state shorter than io header";
        return 2;
    }
    {
        uint32_t mag_a = 0, mag_b = 0;
        int32_t seq_a = 0, seq_b = 0;
        std::memcpy(&mag_a, A.data(), 4);
        std::memcpy(&mag_b, B.data(), 4);
        std::memcpy(&seq_a, A.data() + 4, 4);
        std::memcpy(&seq_b, B.data() + 4, 4);
        if (mag_a != ORACLE_STATE_IO_MAGIC || mag_b != ORACLE_STATE_IO_MAGIC) {
            *note = "wrong sequence state magic";
            return 2;
        }
        if (seq_a != 0 || seq_b != 0) {
            *note = "unexpected state seq_id";
            return 2;
        }
    }

    const size_t S = A.size();
    int64_t nd = 0;
    int64_t first = -1;
    for (size_t i = 0; i < S; ++i) {
        if (A[i] != B[i]) {
            ++nd;
            if (first < 0) first = (int64_t) i;
        }
    }
    if (first_diff) *first_diff = first;
    if (n_diff_bytes) *n_diff_bytes = nd;
    if (nd == 0) {
        *byte_equal = true;
        // Task 320: force_parse continues into the typed structural walk even on equal bytes, so a T3
        // BYTE_EQUAL verdict is backed by a proven walk of the actual serialized layout; a layout the
        // parser cannot prove comes back as PENDING (2) instead of silently skipping the format check.
        // Default false: every earlier caller returns 0 exactly as before (byte_equal only, no walk).
        if (!force_parse) return 0;
    }

    // ext presence affects every attn cell meta; prove it by trying both and requiring a unique parse.
    const size_t ext_candidates[2] = {12, 0};
    int n_ok = 0;
    size_t ext_used = 0;
    std::vector<OracleTensor> ts_used;
    std::string note_used;
    for (size_t k = 0; k < 2; ++k) {
        std::vector<OracleTensor> ts;
        std::string n2;
        // Both streams carry the verified 8-byte io header above; the structural walk starts after it and
        // its first field is attn n_stream (llama_kv_cache::state_write order).
        OracleReader ra{A.data(), A.size()};
        OracleReader rb{B.data(), B.size()};
        ra.off = 8;
        rb.off = 8;
        bool ok = true;

        // Lockstep u32: both streams must carry the same value (a differing header is a real structural
        // divergence between ACT and REF, not padding).
        auto rd32 = [&](uint32_t* out, const char* what) -> bool {
            const uint32_t a = ra.rd<uint32_t>();
            const uint32_t b = rb.rd<uint32_t>();
            if (ra.bad || rb.bad || a != b) {
                n2 = std::string("u32 mismatch: ") + what;
                return false;
            }
            *out = a;
            return true;
        };

        // ---- attention KV part ----
        uint32_t n_stream = 0;
        if (!rd32(&n_stream, "attn n_stream") || n_stream == 0 || n_stream > 64) {
            ok = false;
        }
        for (uint32_t s = 0; ok && s < n_stream; ++s) {
            uint32_t cell_count = 0;
            if (!rd32(&cell_count, "attn cell_count")) { ok = false; break; }
            if (cell_count == 0) continue;
            if (cell_count > 1u << 24) { n2 = "attn cell_count too large"; ok = false; break; }
            std::vector<OracleCellMeta> ca, cb;
            if (!oracle_read_cells(ra, cell_count, ext_candidates[k] != 0, &ca) ||
                !oracle_read_cells(rb, cell_count, ext_candidates[k] != 0, &cb)) {
                n2 = "attn cell meta overrun";
                ok = false;
                break;
            }
            std::vector<size_t> idxb;
            const int mm = oracle_match_cells(ca, cb, &idxb, &n2);
            if (mm == 2) { ok = false; break; }

            uint32_t v_trans = 0, n_layer = 0, n_rot_k = 0, n_rot_v = 0;
            if (!rd32(&v_trans, "attn v_trans") || !rd32(&n_layer, "attn n_layer") ||
                !rd32(&n_rot_k, "attn n_rot_k") || !rd32(&n_rot_v, "attn n_rot_v") ||
                v_trans > 1 || n_layer == 0 || n_layer > 256) {
                ok = false;
                break;
            }
            for (uint32_t L = 0; ok && L < n_layer; ++L) {
                int32_t kta = ra.rd<int32_t>();
                int32_t ktb = rb.rd<int32_t>();
                uint64_t krowa = ra.rd<uint64_t>();
                uint64_t krowb = rb.rd<uint64_t>();
                if (ra.bad || rb.bad || kta != ktb || krowa != krowb || krowa == 0 || krowa > S) {
                    n2 = "attn K header mismatch";
                    ok = false;
                    break;
                }
                size_t esz = 0;
                const char* nm = "raw";
                const bool typed = oracle_float_type(kta, &esz, &nm);
                const int64_t elems_row = typed ? (int64_t) (krowa / esz) : 0;
                // collect rows in physical order, then compare keyed by cell position
                std::vector<const uint8_t*> rowa(cell_count), rowb(cell_count);
                for (uint32_t i = 0; i < cell_count && ok; ++i) {
                    rowa[i] = ra.ptr((size_t) krowa);
                    rowb[i] = rb.ptr((size_t) krowb);
                    if (rowa[i] == nullptr || rowb[i] == nullptr) { n2 = "attn K payload overrun"; ok = false; }
                }
                if (!ok) break;
                OracleTensor t;
                t.name = "attn.kv" + std::to_string(L) + ".K";
                t.dtype = typed ? nm : "raw";
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const OracleRowDiff d = typed
                        ? oracle_diff_elems(rowa[i], rowb[idxb[i]], elems_row, kta, esz)
                        : oracle_diff_bytes(rowa[i], rowb[idxb[i]], (int64_t) krowa);
                    oracle_accum(&t.d, d);
                }
                ts.push_back(std::move(t));
            }
            if (!ok) break;
            if (!v_trans) {
                for (uint32_t L = 0; ok && L < n_layer; ++L) {
                    int32_t vta = ra.rd<int32_t>();
                    int32_t vtb = rb.rd<int32_t>();
                    uint64_t vrowa = ra.rd<uint64_t>();
                    uint64_t vrowb = rb.rd<uint64_t>();
                    if (ra.bad || rb.bad || vta != vtb || vrowa != vrowb || vrowa == 0 || vrowa > S) {
                        n2 = "attn V header mismatch";
                        ok = false;
                        break;
                    }
                    size_t esz = 0;
                    const char* nm = "raw";
                    const bool typed = oracle_float_type(vta, &esz, &nm);
                    const int64_t elems_row = typed ? (int64_t) (vrowa / esz) : 0;
                    std::vector<const uint8_t*> rowa(cell_count), rowb(cell_count);
                    for (uint32_t i = 0; i < cell_count && ok; ++i) {
                        rowa[i] = ra.ptr((size_t) vrowa);
                        rowb[i] = rb.ptr((size_t) vrowb);
                        if (rowa[i] == nullptr || rowb[i] == nullptr) { n2 = "attn V payload overrun"; ok = false; }
                    }
                    if (!ok) break;
                    OracleTensor t;
                    t.name = "attn.kv" + std::to_string(L) + ".V";
                    t.dtype = typed ? nm : "raw";
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const OracleRowDiff d = typed
                            ? oracle_diff_elems(rowa[i], rowb[idxb[i]], elems_row, vta, esz)
                            : oracle_diff_bytes(rowa[i], rowb[idxb[i]], (int64_t) vrowa);
                        oracle_accum(&t.d, d);
                    }
                    ts.push_back(std::move(t));
                }
            } else {
                // Transposed V: the stride (kv_size) is not serialized, so the payload is compared in its
                // written order (j-major, then cells); a physical reorder makes this a byte-order note.
                for (uint32_t L = 0; ok && L < n_layer; ++L) {
                    int32_t vta = ra.rd<int32_t>();
                    int32_t vtb = rb.rd<int32_t>();
                    uint32_t vesa = ra.rd<uint32_t>();
                    uint32_t vesb = rb.rd<uint32_t>();
                    uint32_t nev = ra.rd<uint32_t>();
                    uint32_t nev2 = rb.rd<uint32_t>();
                    if (ra.bad || rb.bad || vta != vtb || vesa != vesb || nev != nev2 || vesa == 0 || nev == 0) {
                        n2 = "attn V trans header mismatch";
                        ok = false;
                        break;
                    }
                    // payload size = n_embd_v * cell_count * v_size_el (j-major, ranges in cell order)
                    const uint64_t plen = (uint64_t) nev * cell_count * vesa;
                    if (plen > S) { n2 = "attn V trans payload too large"; ok = false; break; }
                    if (mm == 0) {
                        const uint8_t* pa = ra.ptr((size_t) plen);
                        const uint8_t* pb = rb.ptr((size_t) plen);
                        if (pa == nullptr || pb == nullptr) { n2 = "attn V trans overrun"; ok = false; break; }
                        OracleTensor t;
                        t.name = "attn.kv" + std::to_string(L) + ".Vt";
                        t.dtype = "raw";
                        oracle_accum(&t.d, oracle_diff_bytes(pa, pb, (int64_t) plen));
                        ts.push_back(std::move(t));
                    } else {
                        // reordered cells: compare elementwise with per-j index mapping
                        std::vector<const uint8_t*> basea(nev), baseb(nev);
                        bool o = true;
                        for (uint32_t j = 0; j < nev && o; ++j) {
                            basea[j] = ra.ptr((size_t) cell_count * vesa);
                            baseb[j] = rb.ptr((size_t) cell_count * vesa);
                            o = basea[j] != nullptr && baseb[j] != nullptr;
                        }
                        if (!o) { n2 = "attn V trans overrun"; ok = false; break; }
                        OracleTensor t;
                        t.name = "attn.kv" + std::to_string(L) + ".Vt";
                        t.dtype = "raw";
                        OracleRowDiff d;
                        for (uint32_t j = 0; j < nev; ++j) {
                            for (uint32_t i = 0; i < cell_count; ++i) {
                                const uint8_t* pa = basea[j] + (size_t) i * vesa;
                                const uint8_t* pb = baseb[j] + idxb[i] * (size_t) vesa;
                                d.n += 1;
                                if (std::memcmp(pa, pb, vesa) != 0) ++d.neq;
                            }
                        }
                        d.max_abs = -1.0;
                        t.d = d;
                        ts.push_back(std::move(t));
                    }
                }
            }
        }
        if (!ok) {
            if (n_ok == 0) note_used = n2;
            continue;   // try the next ext hypothesis
        }

        // ---- recurrent part (GDN state + PLE conv history) ----
        uint32_t r_cells = 0;
        if (!rd32(&r_cells, "recr cell_count") || r_cells == 0 || r_cells > 1u << 10) {
            ok = false;
        }
        if (ok) {
            std::vector<OracleCellMeta> ca, cb;
            if (!oracle_read_cells(ra, r_cells, false, &ca) || !oracle_read_cells(rb, r_cells, false, &cb)) {
                n2 = "recr cell meta overrun";
                ok = false;
            } else {
                // a single cell per sequence: the pos-set match must be exact, reorder cannot occur
                std::vector<size_t> idxb;
                if (oracle_match_cells(ca, cb, &idxb, &n2) != 0) {
                    ok = false;
                }
            }
        }
        uint32_t s_trans = 0, r_n_layer = 0;
        if (ok && (!rd32(&s_trans, "recr s_trans") || !rd32(&r_n_layer, "recr n_layer") ||
                   s_trans > 1 || r_n_layer == 0 || r_n_layer > 256)) {
            ok = false;
        }

        // Bounded DFS over the null-layer pattern: at each layer choose R absent / R present (+P optional),
        // and in the S loop choose absent/present.  Exactly one pattern must parse both streams to their
        // exact end (the full buffer); ambiguity or exhaustion is reported as PENDING, never as equal.
        // Memoized failed states collapse the subset-sum-like re-visits (same layer + same offset is only
        // proven once), and the exploration order tries the header that is locally a real R row first.
        if (ok) {
            struct Dfs {
                OracleReader* ra;
                OracleReader* rb;
                size_t end_a;
                size_t end_b;
                uint32_t n_layer;
                uint32_t cells;
                uint32_t s_trans;
                int64_t steps = 0;
                int64_t cap = 4000000;
                std::vector<uint8_t> rc;   // 0 absent, 1 R, 2 R+P
                std::vector<uint8_t> sc;   // 0 absent, 1 present
                bool found = false;
                bool over = false;
                std::set<std::tuple<int, uint32_t, size_t, size_t>> failed;

                bool has_r_header(size_t oa, size_t ob, uint64_t* row_out) const {
                    if (oa + 12 > end_a || ob + 12 > end_b) return false;
                    int32_t ta, tb;
                    std::memcpy(&ta, ra->base + oa, 4);
                    std::memcpy(&tb, rb->base + ob, 4);
                    uint64_t rwa, rwb;
                    std::memcpy(&rwa, ra->base + oa + 4, 8);
                    std::memcpy(&rwb, rb->base + ob + 4, 8);
                    size_t esz = 0;
                    const char* nm = "raw";
                    if (ta != tb || !oracle_float_type(ta, &esz, &nm) || rwa != rwb || rwa == 0) return false;
                    if (rwa % esz != 0) return false;
                    const uint64_t pay = rwa * cells;
                    if (pay > end_a - (oa + 12) || pay > end_b - (ob + 12)) return false;
                    *row_out = rwa;
                    return true;
                }

                bool run(size_t oa, size_t ob, int phase, uint32_t il) {
                    if (found) return true;
                    if (++steps > cap) { over = true; return false; }
                    const auto key = std::make_tuple(phase, il, oa, ob);
                    if (!failed.insert(key).second) return false;
                    if (phase == 2) {
                        if (oa == end_a && ob == end_b) { found = true; return true; }
                        return false;
                    }
                    if (il == n_layer) return run(oa, ob, phase + 1, 0);
                    if (phase == 0) {
                        uint64_t rw = 0;
                        if (has_r_header(oa, ob, &rw)) {
                            const uint64_t pay = rw * cells;
                            const size_t pa = oa + 12 + (size_t) pay;
                            const size_t pb = ob + 12 + (size_t) pay;
                            if (run(pa, pb, 0, il + 1)) { rc[il] = 1; return true; }
                            if (pa + 8 <= end_a && pb + 8 <= end_b) {
                                uint64_t ppa, ppb;
                                std::memcpy(&ppa, ra->base + pa, 8);
                                std::memcpy(&ppb, rb->base + pb, 8);
                                if (ppa == ppb && ppa > 0) {
                                    const uint64_t ppay = ppa * cells;
                                    if (ppay <= end_a - (pa + 8) && ppay <= end_b - (pb + 8)) {
                                        if (run(pa + 8 + (size_t) ppay, pb + 8 + (size_t) ppay, 0, il + 1)) {
                                            rc[il] = 2;
                                            return true;
                                        }
                                    }
                                }
                            }
                        }
                        if (run(oa, ob, 0, il + 1)) { rc[il] = 0; return true; }
                        return false;
                    }
                    // phase 1: S tensor, ordered by the layer's R presence (the likely coupling)
                    auto try_s = [&]() -> bool {
                        if (oa + 12 > end_a || ob + 12 > end_b) return false;
                        int32_t ta, tb;
                        std::memcpy(&ta, ra->base + oa, 4);
                        std::memcpy(&tb, rb->base + ob, 4);
                        uint64_t pay = 0;
                        if (s_trans) {
                            uint32_t esa, esb, nea, neb;
                            std::memcpy(&esa, ra->base + oa + 4, 4);
                            std::memcpy(&esb, rb->base + ob + 4, 4);
                            std::memcpy(&nea, ra->base + oa + 8, 4);
                            std::memcpy(&neb, rb->base + ob + 8, 4);
                            size_t esz = 0;
                            const char* nm = "raw";
                            if (ta != tb || esa != esb || nea != neb || nea == 0 ||
                                !oracle_float_type(ta, &esz, &nm) || esa != esz) {
                                return false;
                            }
                            pay = (uint64_t) nea * cells * esa;
                        } else {
                            uint64_t rwa, rwb;
                            std::memcpy(&rwa, ra->base + oa + 4, 8);
                            std::memcpy(&rwb, rb->base + ob + 4, 8);
                            size_t esz = 0;
                            const char* nm = "raw";
                            if (ta != tb || rwa != rwb || rwa == 0 ||
                                !oracle_float_type(ta, &esz, &nm) || rwa % esz != 0) {
                                return false;
                            }
                            pay = rwa * cells;
                        }
                        if (pay > end_a - (oa + 12) || pay > end_b - (ob + 12)) return false;
                        if (run(oa + 12 + (size_t) pay, ob + 12 + (size_t) pay, 1, il + 1)) {
                            sc[il] = 1;
                            return true;
                        }
                        return false;
                    };
                    if (rc[il] != 0) {
                        if (try_s()) return true;
                        if (run(oa, ob, 1, il + 1)) { sc[il] = 0; return true; }
                    } else {
                        if (run(oa, ob, 1, il + 1)) { sc[il] = 0; return true; }
                        if (try_s()) return true;
                    }
                    return false;
                }
            } dfs;
            dfs.ra = &ra;
            dfs.rb = &rb;
            dfs.end_a = A.size();
            dfs.end_b = B.size();
            dfs.n_layer = r_n_layer;
            dfs.cells = r_cells;
            dfs.s_trans = s_trans;
            dfs.rc.assign(r_n_layer, 0);
            dfs.sc.assign(r_n_layer, 0);
            const size_t recr_start_a = ra.off;
            const size_t recr_start_b = rb.off;
            dfs.run(ra.off, rb.off, 0, 0);
            if (!dfs.found) {
                n2 = dfs.over ? "recr DFS cap exceeded (ambiguous)" : "no valid recr layer pattern";
                ok = false;
            } else {
                // re-walk the proven pattern and compare payloads
                OracleReader wa{A.data(), A.size()};
                OracleReader wb{B.data(), B.size()};
                wa.off = recr_start_a;
                wb.off = recr_start_b;
                for (uint32_t il = 0; il < r_n_layer && ok; ++il) {
                    if (dfs.rc[il] == 0) continue;
                    int32_t ta = wa.rd<int32_t>();
                    int32_t tb = wb.rd<int32_t>();
                    uint64_t rwa = wa.rd<uint64_t>();
                    uint64_t rwb = wb.rd<uint64_t>();
                    if (wa.bad || wb.bad || ta != tb || rwa != rwb) { n2 = "recr R header"; ok = false; break; }
                    size_t esz = 0;
                    const char* nm = "raw";
                    const bool typed = oracle_float_type(ta, &esz, &nm);
                    OracleTensor t;
                    t.name = "recr.R" + std::to_string(il);
                    t.dtype = typed ? nm : "raw";
                    for (uint32_t i = 0; i < r_cells; ++i) {
                        const uint8_t* pa = wa.ptr((size_t) rwa);
                        const uint8_t* pb = wb.ptr((size_t) rwb);
                        if (pa == nullptr || pb == nullptr) { n2 = "recr R payload"; ok = false; break; }
                        const OracleRowDiff d = typed
                            ? oracle_diff_elems(pa, pb, (int64_t) (rwa / esz), ta, esz)
                            : oracle_diff_bytes(pa, pb, (int64_t) rwa);
                        oracle_accum(&t.d, d);
                    }
                    if (!ok) break;
                    ts.push_back(std::move(t));
                    if (dfs.rc[il] == 2) {
                        uint64_t ppa = wa.rd<uint64_t>();
                        uint64_t ppb = wb.rd<uint64_t>();
                        if (wa.bad || wb.bad || ppa != ppb || ppa == 0) { n2 = "recr P header"; ok = false; break; }
                        OracleTensor tp;
                        tp.name = "recr.P" + std::to_string(il);
                        const bool pty = (ppa % 4) == 0;
                        tp.dtype = pty ? "F32" : "raw";
                        for (uint32_t i = 0; i < r_cells; ++i) {
                            const uint8_t* pa = wa.ptr((size_t) ppa);
                            const uint8_t* pb = wb.ptr((size_t) ppb);
                            if (pa == nullptr || pb == nullptr) { n2 = "recr P payload"; ok = false; break; }
                            const OracleRowDiff d = pty
                                ? oracle_diff_elems(pa, pb, (int64_t) (ppa / 4), 0, 4)
                                : oracle_diff_bytes(pa, pb, (int64_t) ppa);
                            oracle_accum(&tp.d, d);
                        }
                        if (!ok) break;
                        ts.push_back(std::move(tp));
                    }
                }
                for (uint32_t il = 0; il < r_n_layer && ok; ++il) {
                    if (dfs.sc[il] == 0) continue;
                    OracleTensor t;
                    t.name = "recr.S" + std::to_string(il);
                    if (s_trans) {
                        int32_t ta = wa.rd<int32_t>();
                        int32_t tb = wb.rd<int32_t>();
                        uint32_t esa = wa.rd<uint32_t>();
                        uint32_t esb = wb.rd<uint32_t>();
                        uint32_t nea = wa.rd<uint32_t>();
                        uint32_t neb = wb.rd<uint32_t>();
                        if (wa.bad || wb.bad || ta != tb || esa != esb || nea != neb) {
                            n2 = "recr S trans header";
                            ok = false;
                            break;
                        }
                        size_t esz = 0;
                        const char* nm = "raw";
                        const bool typed = oracle_float_type(ta, &esz, &nm);
                        t.dtype = typed ? nm : "raw";
                        for (uint32_t j = 0; j < nea; ++j) {
                            const uint8_t* pa = wa.ptr((size_t) r_cells * esa);
                            const uint8_t* pb = wb.ptr((size_t) r_cells * esa);
                            if (pa == nullptr || pb == nullptr) { n2 = "recr S trans payload"; ok = false; break; }
                            const OracleRowDiff d = typed
                                ? oracle_diff_elems(pa, pb, (int64_t) r_cells, ta, esz)
                                : oracle_diff_bytes(pa, pb, (int64_t) r_cells * esa);
                            oracle_accum(&t.d, d);
                        }
                    } else {
                        int32_t ta = wa.rd<int32_t>();
                        int32_t tb = wb.rd<int32_t>();
                        uint64_t rwa = wa.rd<uint64_t>();
                        uint64_t rwb = wb.rd<uint64_t>();
                        if (wa.bad || wb.bad || ta != tb || rwa != rwb) { n2 = "recr S header"; ok = false; break; }
                        size_t esz = 0;
                        const char* nm = "raw";
                        const bool typed = oracle_float_type(ta, &esz, &nm);
                        t.dtype = typed ? nm : "raw";
                        for (uint32_t i = 0; i < r_cells; ++i) {
                            const uint8_t* pa = wa.ptr((size_t) rwa);
                            const uint8_t* pb = wb.ptr((size_t) rwb);
                            if (pa == nullptr || pb == nullptr) { n2 = "recr S payload"; ok = false; break; }
                            const OracleRowDiff d = typed
                                ? oracle_diff_elems(pa, pb, (int64_t) (rwa / esz), ta, esz)
                                : oracle_diff_bytes(pa, pb, (int64_t) rwa);
                            oracle_accum(&t.d, d);
                        }
                    }
                    if (!ok) break;
                    ts.push_back(std::move(t));
                }
                if (ok && (wa.off != A.size() || wb.off != B.size())) {
                    n2 = "re-walk length mismatch";
                    ok = false;
                }
            }
        }

        if (ok) {
            ++n_ok;
            if (n_ok == 1) {
                ext_used = ext_candidates[k];
                ts_used = std::move(ts);
                note_used.clear();
            } else {
                note_used = "ambiguous ext parse (both hypotheses valid)";
            }
        }
    }
    if (n_ok == 0) {
        *note = note_used.empty() ? "state structure not provable" : note_used;
        return 2;
    }
    if (n_ok > 1) {
        *note = note_used;
        return 2;
    }
    *note = "ext=" + std::to_string(ext_used);
    *tensors = std::move(ts_used);
    return 1;
}

bool oracle_state_save(llama_context* tgt, llama_seq_id seq, std::vector<uint8_t>* buf, size_t* size_out,
                       std::string* err) {
    const size_t sz = llama_state_seq_get_size(tgt, seq);
    if (sz == 0) {
        *err = "llama_state_seq_get_size returned 0";
        return false;
    }
    buf->resize(sz);
    const size_t got = llama_state_seq_get_data(tgt, buf->data(), sz, seq);
    if (got != sz) {
        err->assign("llama_state_seq_get_data returned " + std::to_string(got) + " of " + std::to_string(sz));
        return false;
    }
    *size_out = sz;
    return true;
}

bool oracle_state_load(llama_context* tgt, llama_seq_id seq, const std::vector<uint8_t>& buf, size_t* ret_out) {
    const size_t ret = llama_state_seq_set_data(tgt, buf.data(), buf.size(), seq);
    if (ret_out != nullptr) *ret_out = ret;
    return ret != 0;
}

bool oracle_export(const std::string& dir, const std::string& name, const std::vector<uint8_t>& buf,
                   std::string* err) {
    if (g_oracle_export_bytes + buf.size() > ORACLE_EXPORT_CAP_BYTES) {
        *err = "export cap reached";
        return false;
    }
    const std::string path = dir + "/" + name;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        *err = "open failed: " + path;
        return false;
    }
    const size_t wr = std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (wr != buf.size()) {
        *err = "short write: " + path;
        return false;
    }
    g_oracle_export_bytes += buf.size();
    return true;
}

}  // namespace

bool qwen35_mtp_txn_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_TXN");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

int64_t qwen35_mtp_txn_k() {
    const char* v = std::getenv("STRATA_QWEN35_MTP_TXN_K");
    if (v == nullptr || v[0] == '\0') return 1;
    return std::strtoll(v, nullptr, 10);
}

// Task 273: default-OFF diagnostic fault injection.  Any unknown value is OFF - never guessed.
// Task 275 exact gate: only the single characters '0'..'3' with a trailing NUL are accepted (strtoll's
// prefix acceptance is not a valid gate - "1garbage" must stay OFF, not enable mode 1).
int64_t qwen35_mtp_txn_fault_inject_mode() {
    static const int64_t mode = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_FAULT_INJECT");
        if (v == nullptr || v[0] == '\0' || v[1] != '\0') return (int64_t) 0;
        if (v[0] < '0' || v[0] > '3') return (int64_t) 0;
        return (int64_t) (v[0] - '0');
    }();
    return mode;
}

bool qwen35_mtp_txn_inject_round(int64_t mode, int64_t round_idx, bool last_round_accepted) {
    switch (mode) {
        case 1: return round_idx == 0;
        case 2: return true;
        case 3: return round_idx > 0 && last_round_accepted;
        default: return false;
    }
}

llama_token qwen35_mtp_txn_inject_id(llama_token draft_id, int64_t n_vocab,
                                     bool (*is_eog)(void* user, llama_token id), void* user) {
    if (draft_id < 0 || n_vocab <= 1) return -1;
    for (int64_t step = 1; step < n_vocab; ++step) {
        const llama_token cand = (llama_token) ((((int64_t) draft_id) + step) % n_vocab);
        if (cand == draft_id) break;
        if (is_eog != nullptr && is_eog(user, cand)) continue;
        return cand;
    }
    return -1;
}

void qwen35_mtp_txn_decide(llama_token draft_id, llama_token t0, llama_token t1, llama_token* emit,
                           llama_token* next_pending, int64_t* next_pos_delta, int64_t* rollback_rows) {
    if (t0 == draft_id) {
        *emit = draft_id;
        *next_pending = t1;
        *next_pos_delta = 2;
        *rollback_rows = 0;
    } else {
        *emit = t0;
        *next_pending = t0;
        *next_pos_delta = 1;
        *rollback_rows = 1;
    }
}

int qwen35_mtp_txn_run(llama_model* model, llama_context* tgt, const llama_vocab* vocab, int64_t first_pos,
                       llama_token first_token, int64_t max_new, std::vector<llama_token>* emitted_out,
                       Qwen35MtpTxnTiming* timing_out, std::string* err) {
    auto fail = [&](const std::string& what) {
        if (err != nullptr) *err = what;
        std::fprintf(stderr, "qwen35 mtp-txn: fail-closed: %s\n", what.c_str());
        std::fflush(stderr);
        return 1;
    };
    // Task 261/264: per-token and end-to-end wall observation inside the engine (additive event fields
    // only).  All values are relative to the engine entry (txn_begin); the caller composes them with its own
    // request anchor - they are never presented as the request E2E by themselves.
    const auto txn_begin = std::chrono::steady_clock::now();
    auto t_s = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - txn_begin).count();
    };
    Qwen35MtpTxnTiming* timing = timing_out;
    if (timing != nullptr) *timing = Qwen35MtpTxnTiming{};
    llama_token eog_token = -1;

    // Task 267: default-OFF stage cost accounting.  OFF => every phase cost stays 0 and no extra clock
    // read or synchronize is added, so the control path is the previous behavior (the two extra steady
    // reads per phase are inside the OFF guard).  ON adds host wall clocks only (plus the optional
    // registered sync), never a math/KV/accept/output change.
    static const bool cost_split_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_COST_SPLIT");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    static const bool cost_sync_on = cost_split_on && [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_COST_SYNC");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    // Task 268: exact STRATA_QWEN35_MTP_SERIAL_VERIFY=1 gate for the serial target verifier candidate.
    // OFF keeps the T2 target path (and with it every accept/reject/output decision) exactly as before; the
    // flag changes the verify decode structure only, never the greedy result, the emission or the risk data.
    static const bool serial_verify_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_SERIAL_VERIFY");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    // Task 273: default-OFF diagnostic fault injection (see qwen35_mtp_txn_fault_inject_mode).  It only
    // replaces the token that is REALLY fed to the target verify - the target decode, accept/reject decision,
    // rollback and emissions all run on the injected token, so a reject is a real mismatch, never a
    // post-decode comparison edit.  A process with any mode active is diagnostic (no product speed/accept
    // labeling).  Mode 0 adds no work to the control path.
    static const int64_t fault_inject_mode = qwen35_mtp_txn_fault_inject_mode();
    if (fault_inject_mode != 0) {
        std::fprintf(stderr, "qwen35 mtp-txn fault-injection ACTIVE mode=%lld (diagnostic only)\n",
                     (long long) fault_inject_mode);
        std::fflush(stderr);
    }
    // Task 277: default-OFF rollback state oracle gate (exact STRATA_QWEN35_MTP_ROLLBACK_ORACLE=1).  It
    // requires the T2 verify path (the serial path has no speculative row to roll back and is fail-closed
    // here) and stays a pure diagnostic: no default-path decision, output or timing changes when off.
    static const bool rollback_oracle_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_ROLLBACK_ORACLE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    const char* oracle_dir = rollback_oracle_on ? std::getenv("STRATA_QWEN35_MTP_ORACLE_DIR") : nullptr;
    if (rollback_oracle_on) {
        if (serial_verify_on) return fail("rollback oracle requires the T2 verify (SERIAL_VERIFY=1 is set)");
        std::fprintf(stderr, "qwen35 mtp-txn rollback-oracle ACTIVE (diagnostic only, no product speed)\n");
        std::fflush(stderr);
    }
    // Task 289 (default OFF; exact STRATA_QWEN35_MTP_ORACLE_MATCHED_T2=1, diagnostic-prepare): switches the
    // oracle's REF side to a shape-matched CONTROL trajectory - restore the same PRE, a legal target
    // token-only T2 [pending, corrective t0] (the same batch shape as the ACT wrong-draft T2; only the
    // future speculative row input differs), capture its row0, then the same real seq_rm(pos+1) the ACT side
    // performed and the real corrective T1@pos+1 consuming the control rs_idx.  Fail-closed: it can only run
    // with ROLLBACK_ORACLE=1 and FAULT mode 2 (a real injected reject) and K1 and the T2 verify; an unmet
    // condition fails the run instead of silently enabling or silently dropping the flag.  With the flag
    // absent/off the original serial REF path is untouched.
    static const bool matched_t2_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_ORACLE_MATCHED_T2");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (matched_t2_on) {
        if (!rollback_oracle_on) return fail("MATCHED_T2 requires ROLLBACK_ORACLE=1");
        if (fault_inject_mode != 2) return fail("MATCHED_T2 requires FAULT_INJECT mode 2 (real injected reject)");
        if (qwen35_mtp_txn_k() != 1) return fail("MATCHED_T2 requires the greedy K1 engine");
        if (serial_verify_on) return fail("MATCHED_T2 requires the T2 verify (SERIAL_VERIFY must be 0)");
        std::fprintf(stderr, "qwen35 mtp-txn matched-T2 CONTROL ACTIVE (diagnostic only, no product speed)\n");
        std::fflush(stderr);
    }
    // Task 294 (default OFF; exact STRATA_QWEN35_MTP_ACCEPT_ORACLE=1): diagnostic oracle for the ACCEPTED
    // branch.  It probes only the first real target verify round (round 0) when the actual greedy draft is
    // naturally accepted, captures the real PRE/T2 target state and compares it against a real serial T1
    // replay of the same accepted prefix.  Fail-closed: an unmet gate (serial verify, rollback oracle,
    // matched-T2 control, fault injection, K != 1, missing oracle dir) ends the run instead of silently
    // mixing diagnostics.  With the flag absent/off the loop below is byte-identical to the previous
    // behavior.
    static const bool accept_oracle_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_ACCEPT_ORACLE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    const char* accept_oracle_dir = nullptr;
    if (accept_oracle_on) {
        if (serial_verify_on) return fail("accept oracle requires the T2 verify (SERIAL_VERIFY=1 is set)");
        if (rollback_oracle_on) return fail("accept oracle conflicts with ROLLBACK_ORACLE=1");
        if (matched_t2_on) return fail("accept oracle conflicts with MATCHED_T2=1");
        if (fault_inject_mode != 0) return fail("accept oracle requires FAULT_INJECT mode 0");
        if (qwen35_mtp_txn_k() != 1) return fail("accept oracle requires the greedy K1 engine");
        accept_oracle_dir = std::getenv("STRATA_QWEN35_MTP_ORACLE_DIR");
        if (accept_oracle_dir == nullptr || accept_oracle_dir[0] == '\0')
            return fail("accept oracle requires STRATA_QWEN35_MTP_ORACLE_DIR (export directory)");
        std::fprintf(stderr, "qwen35 mtp-txn accept-oracle ACTIVE (diagnostic only, no product speed)\n");
        std::fflush(stderr);
    }
    // Task 309 (exact STRATA_QWEN35_MTP_VERIFY_PROFILE_RANGE=1, default OFF; see the file header): one
    // bounded CUDA profiler range around the designated round (index 3, zero-based) of the registered
    // request (invocation ordinal 2).  The contract is checked HERE, before the first transaction step, and
    // every unmet condition refuses the run (fail-closed, never a silent switch): K != 1; serial verify; any
    // fault/rollback/matched/accept-oracle diagnostic; COST_SPLIT/COST_SYNC; SMALL_T_PHASES; the legacy
    // TWO_REQUEST path; the T1 profiler range; and STRATA_QWEN35_MTP_TXN_REQUESTS not exactly 2 (the caller
    // must run the two serial transaction requests this ordinal contract assumes).  What this translation
    // unit cannot check - model load/preload/caller-side request ids - is NOT claimed here; the checks are
    // env/caller-contract visible only.
    static const bool verify_profile_range_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_VERIFY_PROFILE_RANGE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (verify_profile_range_on) {
        const auto exact1 = [](const char* name) {
            const char* v = std::getenv(name);
            return v != nullptr && v[0] == '1' && v[1] == '\0';
        };
        if (qwen35_mtp_txn_k() != 1) return fail("verify profile range requires the greedy K1 engine");
        if (serial_verify_on) return fail("verify profile range requires the T2 verify (SERIAL_VERIFY=1 is set)");
        if (fault_inject_mode != 0) return fail("verify profile range conflicts with FAULT_INJECT active");
        if (rollback_oracle_on) return fail("verify profile range conflicts with ROLLBACK_ORACLE=1");
        if (matched_t2_on) return fail("verify profile range conflicts with MATCHED_T2=1");
        if (accept_oracle_on) return fail("verify profile range conflicts with ACCEPT_ORACLE=1");
        if (cost_split_on) return fail("verify profile range conflicts with COST_SPLIT=1");
        if (cost_sync_on) return fail("verify profile range conflicts with COST_SYNC=1");
        if (exact1("STRATA_QWEN35_SMALL_T_PHASES"))
            return fail("verify profile range conflicts with SMALL_T_PHASES=1");
        if (exact1("STRATA_QWEN35_T1_PROFILE_RANGE"))
            return fail("verify profile range conflicts with T1_PROFILE_RANGE=1");
        if (exact1("STRATA_QWEN35_TWO_REQUEST"))
            return fail("verify profile range conflicts with the legacy TWO_REQUEST path");
        const char* nr = std::getenv("STRATA_QWEN35_MTP_TXN_REQUESTS");
        if (nr == nullptr || std::strcmp(nr, "2") != 0)
            return fail("verify profile range requires STRATA_QWEN35_MTP_TXN_REQUESTS=2");
    }
    // Task 317 (exact STRATA_QWEN35_MTP_DRAFT_STATE_ORACLE=1 plus STRATA_QWEN35_MTP_DRAFT_ORACLE_CASE=accept|
    // reject, default OFF): the real draft-context state oracle contract, checked HERE before the first
    // transaction step.  Every unmet condition refuses the run (fail-closed, never a silent switch): a
    // missing/unknown case; K != 1; serial verify; any other oracle (rollback / matched / accept); the
    // profiler ranges; COST_SPLIT/COST_SYNC; SMALL_T_PHASES; the legacy TWO_REQUEST path; CASE=accept without
    // FAULT_INJECT mode 0 (the accept must be the natural greedy one) or CASE=reject without FAULT_INJECT
    // mode 3 (the reject must be a real injected one); and a missing export directory.  OFF adds a couple of
    // local flag comparisons only - no allocation, no state API, no diagnostic math.
    static const bool draft_state_oracle_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_DRAFT_STATE_ORACLE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    const char* draft_oracle_case =
        draft_state_oracle_on ? std::getenv("STRATA_QWEN35_MTP_DRAFT_ORACLE_CASE") : nullptr;
    const bool draft_oracle_accept =
        draft_oracle_case != nullptr && std::strcmp(draft_oracle_case, "accept") == 0;
    const bool draft_oracle_reject =
        draft_oracle_case != nullptr && std::strcmp(draft_oracle_case, "reject") == 0;
    const char* draft_oracle_dir = nullptr;
    if (draft_state_oracle_on) {
        const auto draft_exact1 = [](const char* name) {
            const char* v = std::getenv(name);
            return v != nullptr && v[0] == '1' && v[1] == '\0';
        };
        if (!draft_oracle_accept && !draft_oracle_reject)
            return fail("draft state oracle requires STRATA_QWEN35_MTP_DRAFT_ORACLE_CASE=accept|reject");
        if (qwen35_mtp_txn_k() != 1) return fail("draft state oracle requires the greedy K1 engine");
        if (serial_verify_on) return fail("draft state oracle requires the T2 verify (SERIAL_VERIFY=1 is set)");
        if (rollback_oracle_on) return fail("draft state oracle conflicts with ROLLBACK_ORACLE=1");
        if (matched_t2_on) return fail("draft state oracle conflicts with MATCHED_T2=1");
        if (accept_oracle_on) return fail("draft state oracle conflicts with ACCEPT_ORACLE=1");
        if (verify_profile_range_on) return fail("draft state oracle conflicts with VERIFY_PROFILE_RANGE=1");
        if (cost_split_on || cost_sync_on)
            return fail("draft state oracle conflicts with COST_SPLIT/COST_SYNC=1");
        if (draft_exact1("STRATA_QWEN35_SMALL_T_PHASES"))
            return fail("draft state oracle conflicts with SMALL_T_PHASES=1");
        if (draft_exact1("STRATA_QWEN35_T1_PROFILE_RANGE"))
            return fail("draft state oracle conflicts with T1_PROFILE_RANGE=1");
        if (draft_exact1("STRATA_QWEN35_TWO_REQUEST"))
            return fail("draft state oracle conflicts with the legacy TWO_REQUEST path");
        if (draft_oracle_accept && fault_inject_mode != 0)
            return fail("draft state oracle CASE=accept requires FAULT_INJECT mode 0 (natural accept)");
        if (draft_oracle_reject && fault_inject_mode != 3)
            return fail("draft state oracle CASE=reject requires FAULT_INJECT mode 3 (real injected reject)");
        draft_oracle_dir = std::getenv("STRATA_QWEN35_MTP_ORACLE_DIR");
        if (draft_oracle_dir == nullptr || draft_oracle_dir[0] == '\0')
            return fail("draft state oracle requires STRATA_QWEN35_MTP_ORACLE_DIR (export directory)");
        std::fprintf(stderr, "qwen35 mtp-txn draft-state-oracle ACTIVE case=%s (diagnostic only, no product "
                             "speed)\n", draft_oracle_case);
        std::fflush(stderr);
    }
    // Task 320 (exact STRATA_QWEN35_MTP_T3_VERIFY_ORACLE=1, default OFF; see the file header): the full-graph
    // T3 target verification oracle contract, checked HERE before the first transaction step.  Every unmet
    // condition refuses the run (fail-closed, never a silent switch): K != 1; serial verify; every other
    // oracle (rollback / matched / accept / draft-state) or the profiler range; COST_SPLIT/COST_SYNC;
    // SMALL_T_PHASES; the T1 profiler range; the legacy TWO_REQUEST path; any fault injection; a missing
    // transaction/caller contract (TXN, TXN_REQUESTS != 1); a run configuration other than the registered
    // one (SMALL_T_NATIVE/SMALL_T_MULTI/SLICE_GU_MULTI/COMPACT_SLOTS != 1); a missing export directory.
    static const bool t3_verify_oracle_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_T3_VERIFY_ORACLE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    const char* t3_oracle_dir = nullptr;
    if (t3_verify_oracle_on) {
        const auto t3_exact1 = [](const char* name) {
            const char* v = std::getenv(name);
            return v != nullptr && v[0] == '1' && v[1] == '\0';
        };
        if (qwen35_mtp_txn_k() != 1) return fail("t3 verify oracle requires the greedy K1 engine");
        if (serial_verify_on) return fail("t3 verify oracle requires the T2 verify (SERIAL_VERIFY=1 is set)");
        if (rollback_oracle_on) return fail("t3 verify oracle conflicts with ROLLBACK_ORACLE=1");
        if (matched_t2_on) return fail("t3 verify oracle conflicts with MATCHED_T2=1");
        if (accept_oracle_on) return fail("t3 verify oracle conflicts with ACCEPT_ORACLE=1");
        if (draft_state_oracle_on) return fail("t3 verify oracle conflicts with DRAFT_STATE_ORACLE=1");
        if (verify_profile_range_on) return fail("t3 verify oracle conflicts with VERIFY_PROFILE_RANGE=1");
        if (cost_split_on || cost_sync_on)
            return fail("t3 verify oracle conflicts with COST_SPLIT/COST_SYNC=1");
        if (fault_inject_mode != 0) return fail("t3 verify oracle requires FAULT_INJECT mode 0");
        if (t3_exact1("STRATA_QWEN35_SMALL_T_PHASES"))
            return fail("t3 verify oracle conflicts with SMALL_T_PHASES=1");
        if (t3_exact1("STRATA_QWEN35_T1_PROFILE_RANGE"))
            return fail("t3 verify oracle conflicts with T1_PROFILE_RANGE=1");
        if (t3_exact1("STRATA_QWEN35_TWO_REQUEST"))
            return fail("t3 verify oracle conflicts with the legacy TWO_REQUEST path");
        if (!qwen35_mtp_txn_enabled()) return fail("t3 verify oracle requires the transaction engine (TXN=1)");
        const char* t3_nr = std::getenv("STRATA_QWEN35_MTP_TXN_REQUESTS");
        if (t3_nr == nullptr || std::strcmp(t3_nr, "1") != 0)
            return fail("t3 verify oracle requires STRATA_QWEN35_MTP_TXN_REQUESTS=1");
        if (!t3_exact1("STRATA_QWEN35_SMALL_T_NATIVE"))
            return fail("t3 verify oracle requires STRATA_QWEN35_SMALL_T_NATIVE=1");
        if (!t3_exact1("STRATA_QWEN35_SMALL_T_MULTI"))
            return fail("t3 verify oracle requires STRATA_QWEN35_SMALL_T_MULTI=1");
        if (!t3_exact1("STRATA_QWEN35_SLICE_GU_MULTI"))
            return fail("t3 verify oracle requires STRATA_QWEN35_SLICE_GU_MULTI=1");
        if (!t3_exact1("STRATA_QWEN35_COMPACT_SLOTS"))
            return fail("t3 verify oracle requires STRATA_QWEN35_COMPACT_SLOTS=1");
        t3_oracle_dir = std::getenv("STRATA_QWEN35_MTP_ORACLE_DIR");
        if (t3_oracle_dir == nullptr || t3_oracle_dir[0] == '\0')
            return fail("t3 verify oracle requires STRATA_QWEN35_MTP_ORACLE_DIR (export directory)");
        std::fprintf(stderr, "qwen35 mtp-txn t3-verify-oracle ACTIVE (diagnostic only, no product speed; "
                             "K2 business is not authorized)\n");
        std::fflush(stderr);
    }
    // Task 323 (exact STRATA_QWEN35_MTP_T3_RESTORE_SERIAL_CONTROL=1, default OFF; see the file header):
    // valid ONLY together with the T3 verify oracle - every strict gate above (K1, REQUESTS=1, GU/native/
    // multi/COMPACT, no other oracle / fault / profile / COST / TWO_REQUEST) already holds when the T3
    // flag is on.  The flag alone fails CLOSED inside the transaction; it never silently runs anything.
    static const bool t3_restore_serial_control_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_T3_RESTORE_SERIAL_CONTROL");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (t3_restore_serial_control_on && !t3_verify_oracle_on)
        return fail("t3 restore-serial control requires the exact T3 verify oracle (T3_VERIFY_ORACLE=1)");
    if (t3_restore_serial_control_on) {
        std::fprintf(stderr, "qwen35 mtp-txn t3 restore-serial control ACTIVE (ACT = three serial T1 "
                             "decodes of the identical immutable prefix; diagnostic only, no product speed, "
                             "never an actual T3)\n");
        std::fflush(stderr);
    }
    // Task 325 (exact STRATA_QWEN35_MTP_T3_MOE_SLICE=1, default OFF; diagnostic only): the bounded first-
    // divergence capture at the fixed physical MoE layers 3/4 on the native service seam.  Valid ONLY inside
    // the exact T3 verify-oracle true-T3 mode: with the oracle itself off, or with the restore-serial control
    // (whose ACT is three serial T1 decodes, so no actual T3 exists to capture), the flag fails CLOSED here -
    // never a silent switch.  The bridge phases are armed ONLY around this warm round's REF T1 decodes and
    // the one ACT T3 below (each disarmed immediately after its decode); nothing is armed anywhere else.
    static const bool t3_moe_slice_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_T3_MOE_SLICE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (t3_moe_slice_on && !t3_verify_oracle_on)
        return fail("t3 moe slice probe requires the exact T3 verify oracle (T3_VERIFY_ORACLE=1)");
    if (t3_moe_slice_on && t3_restore_serial_control_on)
        return fail("t3 moe slice probe is invalid with the restore-serial control (no actual T3 exists)");
    if (t3_moe_slice_on) {
        std::fprintf(stderr, "qwen35 mtp-txn t3 moe-slice probe ACTIVE (bounded layer-3/4 service-boundary "
                             "captures in the warm round 1; diagnostic only, no product speed)\n");
        std::fflush(stderr);
    }
    // Task 329 (exact STRATA_QWEN35_MTP_T3_ATTN_SLICE=1, default OFF; diagnostic only): the bounded five
    // fixed-tensor capture through the target-context scheduler eval callback.  Valid ONLY inside the exact
    // T3 verify-oracle true-T3 mode (which already excludes every other oracle/fault/profile/COST/PHASES flag,
    // TWO_REQUEST, and the non-registered run configurations) and never together with the restore-serial
    // control (no actual T3 exists) or the 325/327 moe-slice probe / prefix mode (one bounded probe per run).
    // The phases are armed ONLY around this warm round's REF T1 decodes and the one ACT T3 below, each
    // disarmed immediately after its decode; nothing is armed anywhere else.
    static const bool t3_attn_slice_on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_T3_ATTN_SLICE");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (t3_attn_slice_on && !t3_verify_oracle_on)
        return fail("t3 attn slice probe requires the exact T3 verify oracle (T3_VERIFY_ORACLE=1)");
    if (t3_attn_slice_on && t3_restore_serial_control_on)
        return fail("t3 attn slice probe is invalid with the restore-serial control (no actual T3 exists)");
    if (t3_attn_slice_on && t3_moe_slice_on)
        return fail("t3 attn slice probe conflicts with the moe slice probe (one bounded probe per run)");
    if (t3_attn_slice_on) {
        const char* pv = std::getenv("STRATA_QWEN35_MTP_T3_MOE_SLICE_PREFIX");
        if (pv != nullptr && pv[0] == '1' && pv[1] == '\0')
            return fail("t3 attn slice probe conflicts with the moe slice prefix mode");
        std::fprintf(stderr, "qwen35 mtp-txn t3 attn-slice probe ACTIVE (bounded l_out-2 / Qcur_full-3 / "
                             "Qcur-3 / attn_pregate-3 / attn_output-3 captures in the warm round 1; diagnostic "
                             "only, no product speed)\n");
        std::fflush(stderr);
    }
    // The registered request ordinal: the SECOND serial qwen35_mtp_txn_run invocation in this process (the
    // caller runs one transaction per request, request 1 first).  The counter advances ONLY when the flag is
    // on and the contract held; the designation is this fixed ordinal, never an inherited request id.  The
    // engine is single-threaded per process, so the static counter needs no lock.
    static int64_t g_verify_profile_invocations = 0;
    int64_t verify_profile_ordinal = 0;
    if (verify_profile_range_on) verify_profile_ordinal = ++g_verify_profile_invocations;
    const bool verify_profile_armed = verify_profile_range_on && verify_profile_ordinal == 2;
    const int64_t verify_profile_round = 3;   // fixed designated round index (zero-based)
    VerifyProfileRange prof_range;
    prof_range.requested = verify_profile_armed;
    prof_range.ordinal = verify_profile_ordinal;
    prof_range.designated_round = verify_profile_round;
    if (verify_profile_range_on) {
        std::fprintf(stderr, "qwen35 mtp-txn verify-profile-range %s (invocation %lld; diagnostic, no "
                             "product speed)\n",
                     verify_profile_armed ? "ARMED for this invocation" : "idle for this invocation",
                     (long long) verify_profile_ordinal);
        std::fflush(stderr);
    }
    if (timing != nullptr) {
        timing->cost_split_on = cost_split_on;
        timing->cost_sync_on = cost_sync_on;
    }
    // Host-wall accumulators (ms) + per-round reporting state.  INDEX: 0 init, 1 draft, 2 verify,
    // 3 hcopy, 4 replay, 5 rollback, 6 emit/host.  The clocks themselves (cost_phase / cost_write /
    // print_cost_line) are defined after the draft context exists - they capture it by reference.
    double ph[7] = {0, 0, 0, 0, 0, 0, 0};
    double prev[7] = {0, 0, 0, 0, 0, 0, 0};
    double prev_sync = 0.0;
    int64_t cost_sync_calls = 0;
    double cost_sync_wait_s = 0.0;
    int64_t round_idx = 0;
    double init_start_s = 0.0;

    if (emitted_out != nullptr) emitted_out->clear();
    if (model == nullptr || tgt == nullptr || vocab == nullptr) return fail("null model/ctx/vocab");
    if (max_new <= 0) return fail("max_new must be >= 1 visible token");
    if (qwen35_mtp_txn_k() != 1) return fail("K>1 not implemented in the greedy K=1 engine");
    if (llama_n_rs_seq(tgt) < 1) return fail("target context has n_rs_seq=0: rollback not provable");
    if (t3_verify_oracle_on && llama_n_rs_seq(tgt) < 2)
        return fail("t3 verify oracle requires n_rs_seq>=2 on the target context (caller contract)");
    if (first_pos <= 0) return fail("no prompt token before the pending seed (first_pos <= 0)");
    llama_memory_t tmem = llama_get_memory(tgt);
    if (tmem == nullptr) return fail("target memory is null");

    const int64_t n_ctx = (int64_t) llama_n_ctx(tgt);
    const int64_t n_vocab = llama_vocab_n_tokens(vocab);
    const int64_t n_embd_out = (int64_t) llama_model_n_embd_out(model);
    if (first_pos >= n_ctx) return fail("first pending position >= n_ctx: refusing before any decode");

    bool ok = false;
    std::vector<float> pending_h = copy_nextn_row(tgt, (int32_t) (first_pos - 1), &ok);
    if (!ok) return fail("no finite tail nextn row for the last prompt token");
    std::fprintf(stderr, "qwen35 mtp-txn stage: pending_h row=%lld n_embd_out=%lld bytes=%zu\n",
                 (long long) (first_pos - 1), (long long) n_embd_out, pending_h.size() * sizeof(float));
    std::fflush(stderr);

    if (cost_split_on) init_start_s = t_s();
    auto dp = llama_context_default_params();
    dp.n_ctx = (uint32_t) n_ctx;
    dp.n_batch = 2;
    dp.n_ubatch = 2;
    dp.n_threads = 8;
    dp.n_threads_batch = 8;
    dp.op_offload = true;
    dp.n_seq_max = 1;
    dp.n_outputs_max = 1;
    dp.no_perf = true;
    dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    dp.ctx_other = tgt;
    dp.n_rs_seq = 0;
    std::unique_ptr<llama_context, decltype(&llama_free)> draft(llama_init_from_model(model, dp), llama_free);
    if (draft == nullptr) return fail("draft MTP context init failed");
    llama_set_embeddings_nextn(draft.get(), true, true);

    // Task 267 clocks (defined here: cost_phase captures `draft`).  Every span is a control-flow
    // containment: a displaced readback or a diagnostic synchronize is INSIDE the phase span, not added
    // on top.  want_sync selects the diagnostic sync target: 0 none, +1 tgt, -1 draft.
    auto cost_phase = [&](int i, int8_t want_sync) {
        const double t = cost_split_on ? t_s() : 0.0;
        return [&, i, t, want_sync]() {
            if (!cost_split_on) return;
            if (cost_sync_on && want_sync != 0) {
                llama_context* c = want_sync > 0 ? tgt : draft.get();
                const double s0 = t_s();
                llama_synchronize(c);
                cost_sync_wait_s += t_s() - s0;
                ++cost_sync_calls;
            }
            ph[i] += (t_s() - t) * 1000.0;
        };
    };
    auto cost_write = [&]() {   // emit/host-write span
        const double t = cost_split_on ? t_s() : 0.0;
        return [&, t]() { if (cost_split_on) ph[6] += (t_s() - t) * 1000.0; };
    };
    auto print_cost_line = [&](const char* tag, int64_t round_no) {
        if (!cost_split_on) return;
        std::fprintf(stderr,
                     "qwen35 mtp-txn cost %s round=%lld d_init=%.3f d_draft=%.3f d_verify=%.3f d_hcopy=%.3f "
                     "d_replay=%.3f d_rollback=%.3f d_emit=%.3f c_init=%.3f c_draft=%.3f c_verify=%.3f "
                     "c_hcopy=%.3f c_replay=%.3f c_rollback=%.3f c_emit=%.3f sync_calls=%lld sync_wait_ms=%.3f\n",
                     tag, (long long) round_no, ph[0] - prev[0], ph[1] - prev[1], ph[2] - prev[2],
                     ph[3] - prev[3], ph[4] - prev[4], ph[5] - prev[5], ph[6] - prev[6], ph[0], ph[1],
                     ph[2], ph[3], ph[4], ph[5], ph[6], (long long) cost_sync_calls,
                     (cost_sync_wait_s - prev_sync) * 1000.0);
        std::fflush(stderr);
        for (int i = 0; i < 7; ++i) prev[i] = ph[i];
        prev_sync = cost_sync_wait_s;
    };
    {
        const double t = cost_split_on ? init_start_s : 0.0;
        if (cost_sync_on) {
            const double s0 = t_s();
            llama_synchronize(tgt);
            cost_sync_wait_s += t_s() - s0;
            ++cost_sync_calls;
        }
        if (cost_split_on) ph[0] += (t_s() - t) * 1000.0;
    }

    llama_token pending = first_token;
    const char* pending_role = "seed";   // seed | bonus | corrective
    int64_t pos = first_pos;
    int64_t rounds = 0;
    int64_t emitted = 0;
    int64_t eos_count = 0;
    bool last_round_accepted = false;   // task 273: drives the mode-3 accept-then-reject pattern
    const char* stop = "max_new";
    std::vector<llama_token> emits;

    // Task 277 oracle buffers (allocated only under the gate; at most two live host blocks, 512 MiB total).
    std::vector<uint8_t> oracle_pre, oracle_act, oracle_ref;
    std::vector<float> oracle_act_logits, oracle_ref_logits, oracle_act_h, oracle_ref_h;
    std::vector<float> oracle_act_row0, oracle_ref_row0;
    std::vector<float> oracle_act_h0, oracle_ref_h0;   // task 294: nextn hidden after the pending row (row0)
    size_t oracle_state_size = 0;
    bool oracle_no_stateproof_reported = false;

    // Task 317 draft-state probe state (nothing here is allocated or touched unless the flag is on; the
    // vectors stay empty under the OFF path).
    std::vector<uint8_t> ds_pre, ds_act, ds_ref;
    std::vector<float> ds_act_propose_logits, ds_ref_propose_logits;
    std::vector<float> ds_act_logits, ds_ref_logits, ds_act_h, ds_ref_h;
    std::vector<float> ds_pending_h0;              // deep copy of pending_h before the round-1 propose
    llama_token ds_pending0 = -1;
    int64_t ds_pos0 = -1;
    bool ds_armed = false;
    bool ds_completed = false;
    const int64_t ds_probe_round = 1;              // the single warm round (zero-based)
    auto ds_no_go = [&](const char* reason) -> int {
        std::printf("{\"event\":\"qwen35_mtp_txn_draft_state_oracle_no_go\",\"round\":%lld,\"case\":\"%s\","
                    "\"reason\":\"%s\",\"armed\":%s,\"fail_closed\":true}\n",
                    (long long) round_idx, draft_oracle_case != nullptr ? draft_oracle_case : "unset", reason,
                    ds_armed ? "true" : "false");
        std::fflush(stdout);
        return fail(std::string("draft state oracle NO_GO: ") + reason);
    };
    auto ds_all_zero = [](const std::vector<float>& v) {
        for (float x : v) {
            if (x != 0.0f) return false;
        }
        return true;
    };
    // Task 318: the one bounded DRAFT-only state save used by EVERY PRE/ACT/REF draft-state capture.  The
    // ACTUAL serialized size is queried first and zero / over the 128 MiB per-snapshot cap fails before any
    // allocation; the existing ds_pre/ds_act/ds_ref host buffers plus the requested allocation are accounted
    // with overflow-safe adds against the 384 MiB live cap; afterwards the returned byte count and a
    // post-save size re-query must both be exact (same context, single thread - fail-closed if not).  The
    // 277/294 target-oracle helper and caps stay untouched; under the OFF flag this lambda is never called.
    auto ds_state_save = [&](std::vector<uint8_t>* dst, size_t* size_out, std::string* err) -> bool {
        const size_t sz = llama_state_seq_get_size(draft.get(), 0);
        if (sz == 0) {
            *err = "draft_state_size_zero";
            return false;
        }
        if (sz > DRAFT_ORACLE_STATE_CAP_BYTES) {
            *err = "draft_state_over_per_snapshot_cap";
            return false;
        }
        size_t live = 0;
        const std::vector<uint8_t>* held[3] = { &ds_pre, &ds_act, &ds_ref };
        for (int i = 0; i < 3; ++i) {
            const size_t c = held[i]->capacity();
            if (c > DRAFT_ORACLE_LIVE_CAP_BYTES || live > DRAFT_ORACLE_LIVE_CAP_BYTES - c) {
                *err = "draft_state_live_cap_account_overflow";
                return false;
            }
            live += c;
        }
        if (sz > DRAFT_ORACLE_LIVE_CAP_BYTES || live > DRAFT_ORACLE_LIVE_CAP_BYTES - sz) {
            *err = "draft_state_over_live_cap";
            return false;
        }
        dst->resize(sz);
        const size_t got = llama_state_seq_get_data(draft.get(), dst->data(), sz, 0);
        if (got != sz || dst->size() != sz) {
            *err = "llama_state_seq_get_data returned " + std::to_string(got) + " of " + std::to_string(sz) +
                   " (buffer " + std::to_string(dst->size()) + ")";
            return false;
        }
        if (llama_state_seq_get_size(draft.get(), 0) != sz) {
            *err = "draft_state_size_changed_across_save";
            return false;
        }
        *size_out = sz;
        return true;
    };

    // Task 320 T3-oracle probe state (allocated/touched only under the exact flag; OFF leaves it empty).
    // t3_pre/t3_ref/t3_act are the three unique serialized target snapshots; the per-row vectors are the
    // REF (3 serial T1) and ACT (one T3 batch) logits/hidden deep copies and the natural reference tokens.
    std::vector<uint8_t> t3_pre, t3_ref, t3_act;
    std::vector<float> t3_ref_logits[3], t3_act_logits[3], t3_ref_h[3], t3_act_h[3];
    llama_token t3_nat[3] = {-1, -1, -1};
    bool t3_armed = false;
    bool t3_done = false;
    const int64_t t3_probe_round = 1;   // the single warm round (zero-based), before its normal propose
    auto t3_no_go = [&](const char* reason) -> int {
        std::printf("{\"event\":\"qwen35_mtp_txn_t3_verify_oracle_no_go\",\"round\":%lld,\"reason\":\"%s\","
                    "\"armed\":%s,\"fail_closed\":true}\n",
                    (long long) round_idx, reason, t3_armed ? "true" : "false");
        std::fflush(stdout);
        return fail(std::string("t3 verify oracle NO_GO: ") + reason);
    };
    // Task 320: the one bounded TARGET state save used by EVERY PRE/REF/ACT capture.  The ACTUAL serialized
    // size is queried first; zero / over the 192 MiB per-snapshot cap fails before any allocation.  The
    // three T3 host buffers plus the requested allocation are accounted with overflow-safe adds against the
    // 512 MiB live cap; afterwards the returned byte count and a post-save size re-query must both be exact
    // (same context, single thread - fail-closed if not).  Under the OFF flag this lambda is never called.
    auto t3_state_save = [&](std::vector<uint8_t>* dst, size_t* size_out, std::string* err) -> bool {
        const size_t sz = llama_state_seq_get_size(tgt, 0);
        if (sz == 0) {
            *err = "t3_target_state_size_zero";
            return false;
        }
        if (sz > T3_ORACLE_STATE_CAP_BYTES) {
            *err = "t3_target_state_over_per_snapshot_cap";
            return false;
        }
        size_t live = 0;
        const std::vector<uint8_t>* held[3] = { &t3_pre, &t3_ref, &t3_act };
        for (int i = 0; i < 3; ++i) {
            const size_t c = held[i]->capacity();
            if (c > T3_ORACLE_LIVE_CAP_BYTES || live > T3_ORACLE_LIVE_CAP_BYTES - c) {
                *err = "t3_target_state_live_cap_account_overflow";
                return false;
            }
            live += c;
        }
        if (sz > T3_ORACLE_LIVE_CAP_BYTES || live > T3_ORACLE_LIVE_CAP_BYTES - sz) {
            *err = "t3_target_state_over_live_cap";
            return false;
        }
        dst->resize(sz);
        const size_t got = llama_state_seq_get_data(tgt, dst->data(), sz, 0);
        if (got != sz || dst->size() != sz) {
            *err = "llama_state_seq_get_data returned " + std::to_string(got) + " of " + std::to_string(sz) +
                   " (buffer " + std::to_string(dst->size()) + ")";
            return false;
        }
        if (llama_state_seq_get_size(tgt, 0) != sz) {
            *err = "t3_target_state_size_changed_across_save";
            return false;
        }
        *size_out = sz;
        return true;
    };

    auto emit_one = [&](llama_token id, int64_t at_pos, const char* role, int64_t round_no) {
        std::string piece;
        print_piece_hex(vocab, id, &piece);
        // ready = the token is decided, before its write.  Task 267 field correction: the old event
        // "flush_s" was evaluated as a printf ARGUMENT, i.e. BEFORE the write - it was never a flush
        // instant.  The raw line now carries only the honest pre-write t_s (plus round), and the real
        // post-fflush arrays travel in the typed emit_flush_s vector to the caller's request timing.
        // Old raw logs are untouched; nothing here re-measures or rewrites history.
        const double ready = t_s();
        if (timing != nullptr && timing->first_ready_s < 0) timing->first_ready_s = ready;
        const auto w_done = cost_write();
        std::printf("{\"event\":\"qwen35_mtp_txn_emit\",\"index\":%lld,\"round\":%lld,\"id\":%d,\"pos\":%lld,"
                    "\"role\":\"%s\",\"piece_hex\":\"%s\",\"t_s\":%.9g}\n",
                    (long long) emitted, (long long) round_no, (int) id, (long long) at_pos, role, piece.c_str(),
                    ready);
        std::fflush(stdout);
        const double flushed = t_s();
        w_done();
        if (timing != nullptr) {
            timing->emit_ready_s.push_back(ready);
            timing->emit_flush_s.push_back(flushed);
            if (timing->first_flush_s < 0) timing->first_flush_s = flushed;
        }
        emits.push_back(id);
        ++emitted;
    };

    while (true) {
        if (emitted >= max_new) { stop = "max_new"; break; }
        if (llama_vocab_is_eog(vocab, pending)) {
            stop = (std::strcmp(pending_role, "seed") == 0) ? "seed_eog" : "pending_eog";
            eog_token = pending;
            ++eos_count;
            break;
        }
        if (pos >= n_ctx) return fail("pending position >= n_ctx before decode");
        // The committed pending token is visible now: emit it exactly once (seed / accepted bonus /
        // corrective).  On reject the corrective is pending first and emitted here, not at reject time.
        emit_one(pending, pos, pending_role, round_idx);
        if (emitted >= max_new) { stop = "max_new"; break; }

        if (pos + 1 >= n_ctx) {
            // Context tail: a two-row verify would place the draft at n_ctx.  Legal final serial emit: one
            // one-row decode of the pending at pos (< n_ctx), emit its greedy successor, stop.
            int32_t rc = -1;
            const llama_token tail = pending;
            if (!decode_rows(tgt, &tail, nullptr, 1, pos, true, &rc) || rc != 0)
                return fail("ctx-tail serial decode failed");
            const float* lg = llama_get_logits_ith(tgt, -1);
            const llama_token t = greedy_argmax(lg, n_vocab, &ok);
            if (!ok) return fail("ctx-tail logits not finite");
            if (llama_vocab_is_eog(vocab, t)) { stop = "ctx_tail_eog"; eog_token = t; ++eos_count; break; }
            emit_one(t, pos + 1, "ctx_tail", round_idx);
            stop = "ctx_tail_serial";
            break;
        }

        // Task 320 T3 verification oracle (exact flag, default OFF; diagnostic).  At the single warm round
        // INDEX 1, before its normal propose, the whole round is replaced by one isolated full-graph
        // comparison; the regular MTP logic of this round is skipped and the request stops right after.
        // PRE = the actual target state after the real K1 round 0 (public snapshot; the same logical-RS
        // reader as the 277/294/317 oracles); REF = three real serial token-only T1 decodes of the natural
        // greedy prefix; ACT = one real token-only T3 batch of the same [pending, s1, s2] after the exact
        // PRE restore.  Nothing here manufactures a draft or claims a real MTP acceptance.
        if (t3_verify_oracle_on && round_idx == t3_probe_round && !t3_done) {
            if (g_t3_oracle_probes_done > 0) return t3_no_go("single_probe_per_process_only");
            // EOG/context/end guards BEFORE any REF/ACT decode: the pending is already checked legal by the
            // loop top; the three prefix rows must fit the ACTUAL context (llama_n_ctx, never the CLI value).
            if (pos + 2 >= n_ctx) return t3_no_go("prefix_rows_past_actual_n_ctx");
            t3_armed = true;
            const bool t3_control = t3_restore_serial_control_on;
            std::string t3err;
            size_t t3_pre_sz = 0;
            if (!t3_state_save(&t3_pre, &t3_pre_sz, &t3err))
                return fail("t3 verify oracle PRE state save failed: " + t3err);
            const llama_pos t3_pre_pmin = tmem != nullptr ? llama_memory_seq_pos_min(tmem, 0) : -1;
            const llama_pos t3_pre_pmax = tmem != nullptr ? llama_memory_seq_pos_max(tmem, 0) : -1;
            std::printf("{\"event\":\"qwen35_mtp_txn_t3_verify_oracle_begin\",\"round\":%lld,\"pos\":%lld,"
                        "\"pending\":%d,\"pre_state_size\":%zu,\"probe_kind\":\"%s\","
                        "\"per_snapshot_cap_bytes\":%zu,\"live_cap_bytes\":%zu,"
                        "\"business_acceptance\":false,\"k2_generation\":false}\n",
                        (long long) round_idx, (long long) pos, (int) pending, t3_pre_sz,
                        t3_control ? "restore_serial_control" : "target_t3_vs_serial_t1_fullgraph",
                        T3_ORACLE_STATE_CAP_BYTES, T3_ORACLE_LIVE_CAP_BYTES);
            std::fflush(stdout);
            // REF: three real serial token-only T1 decodes on the LIVE target context, chained by the real
            // natural greedy argmax (input0 = pending@pos; input1 = output0@pos+1; input2 = output1@pos+2).
            // Every row's logits and nextn hidden are deep-copied immediately after its decode.  A prefix
            // token (s1/s2) that is EOG ends the probe explicitly - no forced stream, no fake id.
            llama_token t3_feed = pending;
            for (int r = 0; r < 3; ++r) {
                int32_t t3_row_rc = -1;
                if (t3_moe_slice_on) qwen35_bridge_t3_probe_phase(1, r);
                if (t3_attn_slice_on) qwen35_t3_attn_probe_phase(1, r);
                if (!decode_rows(tgt, &t3_feed, nullptr, 1, pos + r, true, &t3_row_rc) || t3_row_rc != 0)
                    return fail("t3 verify oracle REF T1 decode failed");
                if (t3_moe_slice_on) qwen35_bridge_t3_probe_phase(0, -1);
                if (t3_attn_slice_on) qwen35_t3_attn_probe_phase(0, -1);
                const float* t3_rl = llama_get_logits_ith(tgt, 0);
                if (t3_rl == nullptr) return fail("t3 verify oracle REF row logits null");
                t3_ref_logits[r].assign(t3_rl, t3_rl + (size_t) n_vocab);
                t3_ref_h[r] = copy_nextn_row(tgt, 0, &ok);
                if (!ok) return fail("t3 verify oracle REF row nextn hidden not finite");
                const llama_token t3_rn = greedy_argmax(t3_ref_logits[r].data(), n_vocab, &ok);
                if (!ok) return fail("t3 verify oracle REF row logits not finite");
                t3_nat[r] = t3_rn;
                if (r < 2) {
                    if (llama_vocab_is_eog(vocab, t3_rn))
                        return t3_no_go(r == 0 ? "ref_prefix_token0_eog" : "ref_prefix_token1_eog");
                    t3_feed = t3_rn;
                }
            }
            size_t t3_ref_sz = 0;
            if (!t3_state_save(&t3_ref, &t3_ref_sz, &t3err))
                return fail("t3 verify oracle REF state save failed: " + t3err);
            const llama_pos t3_ref_pmin = tmem != nullptr ? llama_memory_seq_pos_min(tmem, 0) : -1;
            const llama_pos t3_ref_pmax = tmem != nullptr ? llama_memory_seq_pos_max(tmem, 0) : -1;
            const int64_t t3_probe_no = g_t3_oracle_probes_done + 1;
            const std::string t3_odir = std::string(t3_oracle_dir);
            auto t3_export_one = [&](const char* what, const std::vector<uint8_t>& buf) -> bool {
                std::string ex_err;
                const bool wrote = oracle_export(t3_odir, std::string("t3-oracle-p") +
                                                                     std::to_string(t3_probe_no) + "-" + what +
                                                                     "-seq0.bin",
                                                 buf, &ex_err);
                std::printf("{\"event\":\"qwen35_mtp_txn_t3_verify_oracle_export\",\"probe\":%lld,"
                            "\"what\":\"%s\",\"bytes\":%zu,\"ok\":%s,\"note\":\"%s\"}\n",
                            (long long) t3_probe_no, what, buf.size(), wrote ? "true" : "false",
                            ex_err.c_str());
                std::fflush(stdout);
                return wrote;
            };
            if (!t3_export_one("pre", t3_pre)) return fail("t3 verify oracle PRE export failed");
            if (!t3_export_one("ref", t3_ref)) return fail("t3 verify oracle REF export failed");
            // Restore the SAME PRE snapshot on the SAME target context (real llama_state_seq_set_data;
            // exact returned size required), then free PRE immediately - before the ACT capture - so only
            // REF + ACT are live (<= 512 MiB).  No second context and no extra GPU graph/buffers exist.
            size_t t3_setret = 0;
            if (!oracle_state_load(tgt, 0, t3_pre, &t3_setret))
                return fail("t3 verify oracle PRE restore (llama_state_seq_set_data) failed");
            if (t3_setret != t3_pre.size())
                return fail("t3 verify oracle PRE restore size mismatch (return size not exact)");
            const size_t t3_pre_sz_kept = t3_pre.size();
            std::vector<uint8_t>().swap(t3_pre);
            // Task 323 control: with STRATA_QWEN35_MTP_T3_RESTORE_SERIAL_CONTROL=1 the ACT capture is
            // THREE REAL serial token-only T1 decodes of the IDENTICAL immutable [pending, s1, s2] (no
            // re-chaining by argmax), each row's logits/nextn hidden deep-copied IMMEDIATELY from its own
            // call (ith 0) - never re-read from another ith row and never overwritten by a common batch
            // extraction.  Flag OFF keeps the original ONE real T3 batch and its three-row extraction.
            const llama_token t3_ids[3] = { pending, t3_nat[0], t3_nat[1] };
            if (t3_control) {
                for (int r = 0; r < 3; ++r) {
                    int32_t t3_act_row_rc = -1;
                    const llama_token t3_act_feed = t3_ids[r];
                    if (!decode_rows(tgt, &t3_act_feed, nullptr, 1, pos + r, true, &t3_act_row_rc) ||
                        t3_act_row_rc != 0)
                        return fail("t3 restore-serial control ACT T1 decode failed");
                    const float* t3_al = llama_get_logits_ith(tgt, 0);
                    if (t3_al == nullptr)
                        return fail("t3 restore-serial control ACT row logits null");
                    t3_act_logits[r].assign(t3_al, t3_al + (size_t) n_vocab);
                    t3_act_h[r] = copy_nextn_row(tgt, 0, &ok);
                    if (!ok) return fail("t3 restore-serial control ACT row nextn hidden not finite");
                }
            } else {
                // ACT: ONE real token-only T3 batch of the same immutable [pending, s1, s2], all three
                // rows with output logits (the caller's context reserves three output rows for this
                // flag).  The actual corpus batch is this single call; single-vs-split ubatch is reported
                // from the actual n_ubatch, never assumed fused.
                int32_t t3_act_rc = -1;
                if (t3_moe_slice_on) qwen35_bridge_t3_probe_phase(2, -1);
                if (t3_attn_slice_on) qwen35_t3_attn_probe_phase(2, -1);
                if (!decode_rows(tgt, t3_ids, nullptr, 3, pos, true, &t3_act_rc) || t3_act_rc != 0)
                    return fail("t3 verify oracle ACT T3 decode failed");
                if (t3_moe_slice_on) qwen35_bridge_t3_probe_phase(0, -1);
                if (t3_attn_slice_on) qwen35_t3_attn_probe_phase(0, -1);
                for (int r = 0; r < 3; ++r) {
                    const float* t3_al = llama_get_logits_ith(tgt, r);
                    if (t3_al == nullptr)
                        return fail("t3 verify oracle ACT row logits null (outputs_max contract)");
                    t3_act_logits[r].assign(t3_al, t3_al + (size_t) n_vocab);
                }
                for (int r = 0; r < 3; ++r) {
                    t3_act_h[r] = copy_nextn_row(tgt, r, &ok);
                    if (!ok) return fail("t3 verify oracle ACT row nextn hidden not finite");
                }
            }
            size_t t3_act_sz = 0;
            if (!t3_state_save(&t3_act, &t3_act_sz, &t3err))
                return fail("t3 verify oracle ACT state save failed: " + t3err);
            const llama_pos t3_act_pmin = tmem != nullptr ? llama_memory_seq_pos_min(tmem, 0) : -1;
            const llama_pos t3_act_pmax = tmem != nullptr ? llama_memory_seq_pos_max(tmem, 0) : -1;
            if (!t3_export_one("act", t3_act)) return fail("t3 verify oracle ACT export failed");
            // Per-row full-graph compare: EACH row (after pending, after s1, after s2) gates on its own
            // greedy token, finiteness, fixed 1e-5 max-abs AND L1-rel.  A global history aggregate is
            // reported alongside but never dilutes a new row error.
            auto t3_rms = [](const OracleRowDiff& d) {
                return d.n > 0 ? std::sqrt(d.sumsq / (double) d.n) : 0.0;
            };
            auto t3_l1rel = [](const OracleRowDiff& d) {
                return d.l1 / (d.ref_l1 > 0.0 ? d.ref_l1 : 1.0);
            };
            bool t3_rows_gate = true;
            for (int r = 0; r < 3; ++r) {
                const OracleRowDiff t3_dl = oracle_diff_elems((const uint8_t*) t3_act_logits[r].data(),
                                                              (const uint8_t*) t3_ref_logits[r].data(),
                                                              n_vocab, 0, 4);
                const OracleRowDiff t3_dh = oracle_diff_elems((const uint8_t*) t3_act_h[r].data(),
                                                              (const uint8_t*) t3_ref_h[r].data(),
                                                              n_embd_out, 0, 4);
                const llama_token t3_act_next = greedy_argmax(t3_act_logits[r].data(), n_vocab, &ok);
                if (!ok) return fail("t3 verify oracle ACT row logits not finite");
                const bool t3_greedy_ok = (t3_act_next == t3_nat[r]);
                const bool t3_finite_ok = (t3_dl.nonfinite == 0 && t3_dh.nonfinite == 0);
                const bool t3_abs_ok = (t3_dl.max_abs <= 1e-5) && (t3_dh.max_abs <= 1e-5);
                const bool t3_rel_ok = (t3_l1rel(t3_dl) <= 1e-5) && (t3_l1rel(t3_dh) <= 1e-5);
                t3_rows_gate = t3_rows_gate && t3_greedy_ok && t3_finite_ok && t3_abs_ok && t3_rel_ok;
                std::printf("{\"event\":\"qwen35_mtp_txn_t3_verify_oracle_row\",\"probe\":%lld,\"row\":%d,"
                            "\"input_pos\":%lld,\"act_next\":%d,\"ref_next\":%d,"
                            "\"gates\":{\"greedy_equal\":%s,\"finite\":%s,\"max_abs_1e5\":%s,"
                            "\"l1_rel_1e5\":%s,\"bitwise_equal\":%s},"
                            "\"logits\":{\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g,"
                            "\"l1_rel\":%.9g},\"hidden\":{\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,"
                            "\"l1\":%.9g,\"rms\":%.9g,\"l1_rel\":%.9g}}\n",
                            (long long) t3_probe_no, r, (long long) (pos + r), (int) t3_act_next,
                            (int) t3_nat[r], t3_greedy_ok ? "true" : "false", t3_finite_ok ? "true" : "false",
                            t3_abs_ok ? "true" : "false", t3_rel_ok ? "true" : "false",
                            (t3_dl.neq == 0 && t3_dh.neq == 0) ? "true" : "false", (long long) t3_dl.n,
                            (long long) t3_dl.neq, t3_dl.max_abs, t3_dl.l1, t3_rms(t3_dl), t3_l1rel(t3_dl),
                            (long long) t3_dh.n, (long long) t3_dh.neq, t3_dh.max_abs, t3_dh.l1, t3_rms(t3_dh),
                            t3_l1rel(t3_dh));
                std::fflush(stdout);
            }
            // Final serialized target state: ACT (one T3 batch) vs REF (three serial T1s) from the same PRE.
            // BYTE_EQUAL is mandatory for the scope verdict; the typed GDN/KV layout is force-walked even on
            // equal bytes, and a layout the parser cannot prove is an explicit PARTIAL - never a byte-equal
            // silent PASS.  The logical state is tracked as the typed tensors (attn cells incl. the new
            // pos..pos+2 rows; recr GDN R/S/P per layer), not as rollback-plane bookkeeping.
            const bool t3_size_ok = (t3_act_sz == t3_ref_sz);
            std::vector<OracleTensor> t3_tensors;
            std::string t3_note;
            int64_t t3_first = -1;
            int64_t t3_ndiff = 0;
            bool t3_byte_equal = false;
            int t3_st_rc = 2;
            if (t3_size_ok) {
                t3_st_rc = oracle_compare_state_bytes(t3_act, t3_ref, &t3_tensors, &t3_note, &t3_first,
                                                      &t3_ndiff, &t3_byte_equal, true);
            } else {
                t3_note = "act/ref serialized size mismatch";
            }
            OracleRowDiff t3_st_acc;
            int64_t t3_typed_tensors = 0;
            int64_t t3_raw_tensors = 0;
            int64_t t3_typed_elems = 0;
            for (const auto& t : t3_tensors) {
                if (std::strcmp(t.dtype, "raw") == 0) {
                    ++t3_raw_tensors;
                } else {
                    ++t3_typed_tensors;
                    t3_typed_elems += t.d.n;
                }
                oracle_accum(&t3_st_acc, t.d);
            }
            const char* t3_state_status =
                (t3_st_rc == 1 && t3_byte_equal) ? "BYTE_EQUAL_PARSED"
                : (t3_byte_equal ? "BYTE_EQUAL_PARSER_UNKNOWN"
                                 : (t3_st_rc == 1 ? "TYPED_COMPARED" : "PENDING_DECODER"));
            const bool t3_state_ok = (t3_size_ok && t3_byte_equal);
            const bool t3_parser_proven = (t3_st_rc == 1);
            // Task 321 typed-coverage/finite gate: a byte-equal, provably-parsed walk alone is not a PASS.
            // It must carry NONEMPTY typed floating payloads (every walked tensor a proved float dtype, at
            // least one element) and the elementwise comparison must have seen ZERO non-finite values, both
            // per typed tensor and accumulated.  A raw/unknown/unprovable payload is an explicit PARTIAL and
            // never a PASS; structural integer metadata is never counted as float.
            const bool t3_typed_covered = (t3_parser_proven && t3_typed_tensors > 0 &&
                                           t3_raw_tensors == 0 && t3_typed_elems > 0);
            const bool t3_typed_finite_ok = (t3_typed_covered && t3_st_acc.nonfinite == 0);
            const bool t3_scope_pass = (t3_rows_gate && t3_state_ok && t3_parser_proven && t3_typed_finite_ok);
            // Task 323: the control run never uses the T3 PASS/fail verdict names - its own explicit
            // RESTORE_SERIAL_CONTROL_PASS / CONTROL_* vocabulary is reported instead; the frame carries
            // the explicit restore_serial_control probe_kind and the actual call/batch counts.
            const char* t3_verdict =
                t3_scope_pass ? (t3_control ? "RESTORE_SERIAL_CONTROL_PASS" : "T3_SCOPE_PASS")
                : (!t3_state_ok ? (t3_control ? "CONTROL_STATE_NOT_BYTE_EQUAL" : "STATE_NOT_BYTE_EQUAL")
                                : (!t3_parser_proven
                                       ? (t3_control ? "CONTROL_PARTIAL_PARSER_UNKNOWN"
                                                     : "PARTIAL_PARSER_UNKNOWN")
                                       : (!t3_typed_covered
                                              ? (t3_control ? "CONTROL_PARTIAL_TYPED_COVERAGE"
                                                            : "PARTIAL_TYPED_COVERAGE")
                                              : (!t3_typed_finite_ok
                                                     ? (t3_control ? "CONTROL_TYPED_NONFINITE_FAIL"
                                                                   : "TYPED_NONFINITE_FAIL")
                                                     : (t3_control ? "CONTROL_ROW_GATE_FAIL"
                                                                   : "ROW_GATE_FAIL")))));
            const char* t3_probe_kind =
                t3_control ? "restore_serial_control" : "target_t3_vs_serial_t1_fullgraph";
            const char* t3_api_calls =
                t3_control ? "{\"state_saves\":3,\"state_restores\":1,\"ref_t1_decodes\":3,"
                             "\"act_t1_decodes\":3,\"act_t3_decodes\":0}"
                           : "{\"state_saves\":3,\"state_restores\":1,\"ref_t1_decodes\":3,"
                             "\"act_t3_decodes\":1}";
            char t3_batch_frag[160];
            if (t3_control) {
                std::snprintf(t3_batch_frag, sizeof(t3_batch_frag),
                              "{\"calls\":3,\"rows_per_call\":1,\"outputs_per_call\":1,\"captured_rows\":3,"
                              "\"ctx_n_ubatch\":%u}", (unsigned) llama_n_ubatch(tgt));
            } else {
                std::snprintf(t3_batch_frag, sizeof(t3_batch_frag),
                              "{\"rows\":3,\"n_outputs_used\":3,\"ctx_n_ubatch\":%u,\"ubatch_covers_t3\":%s}",
                              (unsigned) llama_n_ubatch(tgt), llama_n_ubatch(tgt) >= 3 ? "true" : "false");
            }
            const char* t3_note_msg =
                t3_control
                    ? "target restore-vs-live serial T1 control (same immutable prefix from the same PRE); "
                      "not a real draft acceptance, not product speed; ACT state not restored; diagnostic stop"
                    : "teacher-forced target verification diagnostic; not a real draft acceptance, not "
                      "product speed; ACT state not restored; diagnostic stop";
            std::printf("{\"event\":\"qwen35_mtp_txn_t3_verify_oracle_compare\",\"probe\":%lld,\"round\":%lld,"
                        "\"pos\":%lld,\"pending\":%d,\"s1\":%d,\"s2\":%d,\"s3\":%d,\"ref_s3_eog\":%s,"
                        "\"probe_kind\":\"%s\",\"teacher_forced_act_prefix\":true,"
                        "\"business_acceptance\":false,\"k2_generation\":false,"
                        "\"api_calls\":%s,\"batch\":%s,"
                        "\"gates\":{\"rows_all\":%s,\"state_size_equal\":%s,\"state_byte_equal\":%s,"
                        "\"state_parser_proven\":%s,\"state_typed_covered\":%s,\"state_typed_finite\":%s,"
                        "\"scope_pass\":%s},"
                        "\"state\":{\"status\":\"%s\",\"pre_size\":%zu,\"act_size\":%zu,\"ref_size\":%zu,"
                        "\"tensors\":%zu,\"typed_tensors\":%lld,\"raw_tensors\":%lld,\"typed_elems\":%lld,"
                        "\"elems\":%lld,\"neq\":%lld,\"nonfinite\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,"
                        "\"rms\":%.9g,\"first_diff\":%lld,\"diff_bytes\":%lld,\"note\":\"%s\"},"
                        "\"seq_range\":{\"pre_min\":%lld,\"pre_max\":%lld,\"ref_min\":%lld,\"ref_max\":%lld,"
                        "\"act_min\":%lld,\"act_max\":%lld},\"verdict\":\"%s\","
                        "\"note\":\"%s\"}\n",
                        (long long) t3_probe_no, (long long) round_idx, (long long) pos, (int) pending,
                        (int) t3_nat[0], (int) t3_nat[1], (int) t3_nat[2],
                        llama_vocab_is_eog(vocab, t3_nat[2]) ? "true" : "false",
                        t3_probe_kind,
                        t3_api_calls, t3_batch_frag,
                        t3_rows_gate ? "true" : "false", t3_size_ok ? "true" : "false",
                        t3_byte_equal ? "true" : "false", t3_parser_proven ? "true" : "false",
                        t3_typed_covered ? "true" : "false", t3_typed_finite_ok ? "true" : "false",
                        t3_scope_pass ? "true" : "false", t3_state_status, t3_pre_sz_kept, t3_act_sz,
                        t3_ref_sz, t3_tensors.size(), (long long) t3_typed_tensors,
                        (long long) t3_raw_tensors, (long long) t3_typed_elems, (long long) t3_st_acc.n,
                        (long long) t3_st_acc.neq, (long long) t3_st_acc.nonfinite, t3_st_acc.max_abs,
                        t3_st_acc.l1, t3_rms(t3_st_acc), (long long) t3_first,
                        (long long) t3_ndiff, t3_note.c_str(), (long long) t3_pre_pmin,
                        (long long) t3_pre_pmax, (long long) t3_ref_pmin, (long long) t3_ref_pmax,
                        (long long) t3_act_pmin, (long long) t3_act_pmax, t3_verdict, t3_note_msg);
            std::fflush(stdout);
            if (t3_st_rc == 1) {
                for (const auto& t : t3_tensors) {
                    const bool t_typed = (std::strcmp(t.dtype, "raw") != 0);
                    std::printf("{\"event\":\"qwen35_mtp_txn_t3_verify_oracle_state_tensor\",\"probe\":%lld,"
                                "\"name\":\"%s\",\"dtype\":\"%s\",\"typed\":%s,\"n\":%lld,\"neq\":%lld,"
                                "\"nonfinite\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g}\n",
                                (long long) t3_probe_no, t.name.c_str(), t.dtype,
                                t_typed ? "true" : "false", (long long) t.d.n,
                                (long long) t.d.neq, (long long) t.d.nonfinite, t.d.max_abs, t.d.l1,
                                t3_rms(t.d));
                }
                std::fflush(stdout);
            }
            if (!t3_rows_gate)
                return fail(t3_control
                                ? "t3 restore-serial control row numeric/greedy gate failed (control not PASS)"
                                : "t3 verify oracle row numeric/greedy gate failed (scope not PASS)");
            if (!t3_state_ok)
                return fail(t3_control
                                ? "t3 restore-serial control ACT/REF serialized state not BYTE_EQUAL (control "
                                  "not PASS)"
                                : "t3 verify oracle ACT/REF serialized target state not BYTE_EQUAL (scope not "
                                  "PASS)");
            // Task 321: a proven parse whose typed coverage is empty/raw or whose typed comparison saw a
            // non-finite value is an explicit PARTIAL/fail - it can never continue into a normal request.
            if (t3_parser_proven && !t3_typed_finite_ok)
                return fail(t3_control
                                ? (t3_typed_covered
                                       ? "t3 restore-serial control typed state comparison found non-finite "
                                         "values (control not PASS)"
                                       : "t3 restore-serial control typed state coverage not fully proven "
                                         "(raw/unprovable payload; control not PASS)")
                                : (t3_typed_covered
                                       ? "t3 verify oracle typed state comparison found non-finite values "
                                         "(scope not PASS)"
                                       : "t3 verify oracle typed state coverage not fully proven "
                                         "(raw/unprovable payload; scope not PASS)"));
            ++g_t3_oracle_probes_done;
            t3_done = true;
            // Diagnostic prefix stop: the ACT state is NOT restored and no continued request / product
            // speed is claimed; a later request re-prefills explicitly.  A byte-equal state whose layout
            // the parser could not prove keeps the explicit PARTIAL verdict above and never becomes a
            // silent G1 PASS.  Task 323: the control run uses its own stop tag and never claims T3 scope.
            stop = t3_control ? "restore_serial_control" : "t3_verify_oracle_probe";
            break;
        }

        // Task 317: BEFORE the round-1 propose, snapshot the DRAFT context's real serialized sequence state
        // (PRE) and deep-copy the pending token / position / pending_h.  Only when the exact flag is on and
        // this is the single warm probe round; every failure is an explicit NO_GO before any oversize
        // allocation.  Task 318: PRE/ACT/REF all go through the one bounded ds_state_save helper (actual
        // size first, 128 MiB per snapshot and 384 MiB live checked before allocation).  OFF: the condition
        // short-circuits, nothing is called.
        if (draft_state_oracle_on && round_idx == ds_probe_round && !ds_armed && !ds_completed) {
            std::string dserr;
            size_t ds_sz = 0;
            if (!ds_state_save(&ds_pre, &ds_sz, &dserr))
                return ds_no_go(dserr.c_str());
            ds_pending0 = pending;
            ds_pos0 = pos;
            ds_pending_h0 = pending_h;
            ds_armed = true;
            std::printf("{\"event\":\"qwen35_mtp_txn_draft_state_oracle_begin\",\"round\":%lld,\"case\":\"%s\","
                        "\"pos\":%lld,\"pending\":%d,\"draft_state_size\":%zu,\"pending_h_bytes\":%zu,"
                        "\"probe_kind\":\"draft_ctx_warm_round1\",\"per_snapshot_cap_bytes\":%zu,"
                        "\"live_cap_bytes\":%zu}\n",
                        (long long) round_idx, draft_oracle_case, (long long) pos, (int) pending, ds_sz,
                        ds_pending_h0.size() * sizeof(float), DRAFT_ORACLE_STATE_CAP_BYTES,
                        DRAFT_ORACLE_LIVE_CAP_BYTES);
            std::fflush(stdout);
        }
        int32_t rc = -1;
        const auto draft_phase_done = cost_phase(1, -1);
        if (!decode_rows(draft.get(), &pending, &pending_h, 1, pos, true, &rc) || rc != 0)
            return fail("draft round decode failed");
        llama_token draft_id = greedy_argmax(llama_get_logits_ith(draft.get(), -1), n_vocab, &ok);
        if (!ok) return fail("draft logits not finite");
        // Task 317: the ACT propose head (read-only copy of the real propose logits for the REF repro compare).
        if (draft_state_oracle_on && ds_armed && round_idx == ds_probe_round) {
            const float* ds_prop_lg = llama_get_logits_ith(draft.get(), -1);
            if (ds_prop_lg != nullptr) ds_act_propose_logits.assign(ds_prop_lg, ds_prop_lg + (size_t) n_vocab);
        }
        draft_phase_done();

        // Task 273 (default OFF): replace the draft token that every later consumer uses.  The original
        // greedy draft is kept for logging only; `draft_id` is reassigned BEFORE the target verify, so the
        // speculative row, decide, rollback and any replay all operate on the injected token.
        const llama_token original_draft = draft_id;
        const bool injected_this_round =
            fault_inject_mode != 0 && qwen35_mtp_txn_inject_round(fault_inject_mode, round_idx, last_round_accepted);
        if (injected_this_round) {
            const llama_token inj = qwen35_mtp_txn_inject_id(original_draft, n_vocab, &inject_eog_cb, (void*) vocab);
            if (inj < 0) return fail("fault injection found no distinct legal replacement token");
            draft_id = inj;
            std::fprintf(stderr, "qwen35 mtp-txn inject round=%lld original_draft=%d injected_id=%d\n",
                         (long long) round_idx, (int) original_draft, (int) draft_id);
            std::fflush(stderr);
        }

        // Task 277 oracle: capture the target sequence's pre-verifier LOGICAL state (public API; with a
        // pending rs_idx the writer serializes the rollback snapshot plane, i.e. the logical current state).
        // Only injected rounds are probed (the first real injected reject is the designed target); at most
        // one probe per request and ORACLE_MAX_PROBES per process.
        bool oracle_pre_saved = false;
        if (rollback_oracle_on && injected_this_round && g_oracle_probes_done < ORACLE_MAX_PROBES) {
            if (oracle_state_size == 0) oracle_state_size = llama_state_seq_get_size(tgt, 0);
            if (oracle_state_size == 0 || 2 * oracle_state_size > ORACLE_SIMUL_CAP_BYTES) {
                if (!oracle_no_stateproof_reported) {
                    oracle_no_stateproof_reported = true;
                    std::printf("{\"event\":\"qwen35_mtp_txn_oracle_no_stateproof\",\"round\":%lld,"
                                "\"state_size\":%zu,\"simul_cap_bytes\":%zu,\"reason\":\"%s\"}\n",
                                (long long) round_idx, oracle_state_size, ORACLE_SIMUL_CAP_BYTES,
                                oracle_state_size == 0 ? "size_zero" : "simul_cap");
                    std::fflush(stdout);
                }
            } else {
                std::string oserr;
                if (!oracle_state_save(tgt, 0, &oracle_pre, &oracle_state_size, &oserr))
                    return fail("oracle PRE state save failed: " + oserr);
                oracle_pre_saved = true;
                std::printf("{\"event\":\"qwen35_mtp_txn_oracle_begin\",\"probe\":%lld,\"round\":%lld,"
                            "\"pos\":%lld,\"pending\":%d,\"draft\":%d,\"state_size\":%zu}\n",
                            (long long) (g_oracle_probes_done + 1), (long long) round_idx, (long long) pos,
                            (int) pending, (int) draft_id, oracle_state_size);
                std::fflush(stdout);
            }
        }

        // Task 294 accept oracle PRE capture: before the real target T2 of the first round.  Whether that
        // round is actually a natural accept is only known after the verify below; a reject ends the run
        // explicitly (NO_ACCEPT_PROBE) instead of probing a manufactured input.
        bool accept_probe_armed = false;
        if (accept_oracle_on && round_idx == 0 && g_oracle_probes_done < ORACLE_MAX_PROBES) {
            if (oracle_state_size == 0) oracle_state_size = llama_state_seq_get_size(tgt, 0);
            if (oracle_state_size == 0 || 2 * oracle_state_size > ORACLE_SIMUL_CAP_BYTES)
                return fail("accept oracle PRE state block exceeds the two-live-block simul cap");
            std::string oserr;
            if (!oracle_state_save(tgt, 0, &oracle_pre, &oracle_state_size, &oserr))
                return fail("accept oracle PRE state save failed: " + oserr);
            accept_probe_armed = true;
            std::printf("{\"event\":\"qwen35_mtp_txn_oracle_begin\",\"probe\":%lld,\"round\":%lld,\"pos\":%lld,"
                        "\"pending\":%d,\"draft\":%d,\"state_size\":%zu,\"accept_oracle\":true}\n",
                        (long long) (g_oracle_probes_done + 1), (long long) round_idx, (long long) pos,
                        (int) pending, (int) draft_id, oracle_state_size);
            std::fflush(stdout);
        }

        llama_token t0 = -1;
        llama_token t1 = -1;   // serial: present only when T1#2 actually decoded the draft row (else -1)
        std::vector<float> h_row0;
        std::vector<float> h_row1;
        bool accepted = false;
        int64_t rollback = 0;   // rows actually removed from the target sequence this round

        if (!serial_verify_on) {
            // T2 control: one token-only target batch [pending, draft_id].  Every phase closes before the
            // next one opens (task 268 cost-window fix), so no span contains another span's work.
            const auto verify_phase_done = cost_phase(2, 1);
            const llama_token vids[2] = {pending, draft_id};
            // Task 309: the designated round's real T2 opens the bounded profiler range immediately before
            // this decode.  The clock read and the owner call happen only when armed (OFF: two local flag
            // comparisons per T2 round, no clock read, no profiler call, no event).
            if (verify_profile_armed && round_idx == verify_profile_round)
                prof_range.try_start(t_s(), round_idx, qwen35_mtp_txn_k(), pos, pending, draft_id);
            if (!decode_rows(tgt, vids, nullptr, 2, pos, true, &rc) || rc != 0)
                return fail("target verify decode failed");
            const float* vrow0 = llama_get_logits_ith(tgt, 0);
            if (oracle_pre_saved && vrow0 != nullptr) oracle_act_row0.assign(vrow0, vrow0 + (size_t) n_vocab);
            t0 = greedy_argmax(vrow0, n_vocab, &ok);
            if (!ok) return fail("verify row0 logits not finite");
            t1 = greedy_argmax(llama_get_logits_ith(tgt, 1), n_vocab, &ok);
            if (!ok) return fail("verify row1 logits not finite");
            verify_phase_done();
            // Deep-copy both verify rows BEFORE any further decode can touch the target's embedding buffers:
            // row0 = hidden after pending@pos (the h input for the accepted token replayed at pos+1), row1 =
            // hidden after draft@pos+1 (the next pending hidden after an accept).  The copy phase holds the
            // copies and their finiteness scans only - it is closed here, before any emit work.
            const auto hcopy_phase_done = cost_phase(3, 0);
            h_row0 = copy_nextn_row(tgt, 0, &ok);
            if (!ok) return fail("no finite nextn row0 for the accepted-prefix draft sync");
            h_row1 = copy_nextn_row(tgt, 1, &ok);
            if (!ok) return fail("no finite nextn row1 for the accepted bonus pending");
            hcopy_phase_done();
            // Task 309: the registered close point - after the existing target logits/nextn readiness and
            // both deep copies, before decide/printf/replay/rollback.  Stop is a capture toggle at an
            // EXISTING readiness instant, never a new synchronization; no-op when never started (e.g. a
            // failed start, or a non-designated round).
            if (verify_profile_armed) prof_range.stop_designated(t_s());
        } else {
            // Serial T1 verify (task 268, default OFF): T1#1 decodes the pending token alone; t0 and h0 are
            // read and copied immediately (before any later decode can refill the target buffers).  Only on
            // draft_id == t0 with a legal (non-EOG) draft token does T1#2 decode the draft token alone for
            // t1/h1.  On reject the target never decoded the speculative row, so nothing is removed.
            const auto verify1_phase_done = cost_phase(2, 1);
            const llama_token v0 = pending;
            if (!decode_rows(tgt, &v0, nullptr, 1, pos, true, &rc) || rc != 0)
                return fail("serial verify T1#1 (pending) decode failed");
            t0 = greedy_argmax(llama_get_logits_ith(tgt, -1), n_vocab, &ok);
            if (!ok) return fail("serial verify T1#1 logits not finite");
            verify1_phase_done();

            const auto hcopy0_phase_done = cost_phase(3, 0);
            h_row0 = copy_nextn_row(tgt, 0, &ok);
            if (!ok) return fail("serial verify: no finite h0 for the pending row");
            hcopy0_phase_done();

            accepted = (draft_id == t0);
            if (accepted && !llama_vocab_is_eog(vocab, draft_id)) {
                const auto verify2_phase_done = cost_phase(2, 1);
                const llama_token v1 = draft_id;
                if (!decode_rows(tgt, &v1, nullptr, 1, pos + 1, true, &rc) || rc != 0)
                    return fail("serial verify T1#2 (draft) decode failed");
                t1 = greedy_argmax(llama_get_logits_ith(tgt, -1), n_vocab, &ok);
                if (!ok) return fail("serial verify T1#2 logits not finite");
                verify2_phase_done();

                const auto hcopy1_phase_done = cost_phase(3, 0);
                h_row1 = copy_nextn_row(tgt, 0, &ok);
                if (!ok) return fail("serial verify: no finite h1 for the draft row");
                hcopy1_phase_done();
            }
        }

        llama_token emit = -1, next_pending = -1;
        int64_t dpos = 0;
        if (!serial_verify_on) {
            qwen35_mtp_txn_decide(draft_id, t0, t1, &emit, &next_pending, &dpos, &rollback);
            accepted = (rollback == 0);
        } else if (accepted) {
            emit = draft_id;
            next_pending = t1;   // -1 only in the accepted-EOG stop below; never emitted
            dpos = 2;
        } else {
            emit = t0;
            next_pending = t0;
            dpos = 1;
        }
        const int64_t accept_report = accepted ? 1 : 0;
        // Task 273: injected_mismatch = the injection gate was active this round AND the target verify
        // really disagreed (t0 != injected_id); it is NOT a comparison-value edit - the mismatch token above
        // is the row the target actually decoded.  inject=0 rounds: original_draft == draft_id and 0.
        const int64_t injected_mismatch = (injected_this_round && accept_report == 0) ? 1 : 0;

        std::string pe, pp;
        print_piece_hex(vocab, emit, &pe);
        if (next_pending >= 0) print_piece_hex(vocab, next_pending, &pp);   // task 273: -1 (serial accepted-EOG) must not be piece-converted
        const auto round_write_done = cost_write();
        std::printf("{\"event\":\"qwen35_mtp_txn_round\",\"round\":%lld,\"pos\":%lld,\"seed\":%d,"
                    "\"draft\":%d,\"t0\":%d,\"t1\":%d,\"accept\":%lld,\"rollback\":%lld,\"emit\":%d,"
                    "\"inject\":%lld,\"original_draft\":%d,\"injected_id\":%d,\"injected_mismatch\":%lld,"
                    "\"piece_hex\":\"%s\",\"next_pending\":%d,\"next_piece_hex\":\"%s\","
                    "\"h_row0_bytes\":%zu,\"h_row1_bytes\":%zu,\"emitted_after\":%lld,\"t_s\":%.9g}\n",
                    (long long) rounds, (long long) pos, (int) pending, (int) draft_id, (int) t0, (int) t1,
                    (long long) accept_report, (long long) rollback, (int) emit,
                    (long long) (injected_this_round ? 1 : 0), (int) original_draft, (int) draft_id,
                    (long long) injected_mismatch, pe.c_str(),
                    (int) next_pending, pp.c_str(), h_row0.size() * sizeof(float), h_row1.size() * sizeof(float),
                    (long long) emitted, t_s());
        std::fflush(stdout);
        round_write_done();

        if (accept_probe_armed && !accepted) {
            std::printf("{\"event\":\"qwen35_mtp_txn_accept_oracle_no_probe\",\"round\":%lld,\"pos\":%lld,"
                        "\"pending\":%d,\"draft\":%d,\"t0\":%d,\"reason\":\"round0_not_accepted\","
                        "\"fail_closed\":true}\n",
                        (long long) round_idx, (long long) pos, (int) pending, (int) draft_id, (int) t0);
            std::fflush(stdout);
            std::vector<uint8_t>().swap(oracle_pre);
            return fail("accept oracle NO_ACCEPT_PROBE: first verify round was not a natural accept");
        }

        if (accepted) {
            // Task 277: an injected round that accepted cannot host the reject oracle; free the pre block
            // and let the engine continue (with FAULT modes the next injected round is probed instead).
            if (oracle_pre_saved) {
                std::printf("{\"event\":\"qwen35_mtp_txn_oracle_skip\",\"round\":%lld,"
                            "\"reason\":\"injected_round_accepted\"}\n", (long long) round_idx);
                std::fflush(stdout);
                std::vector<uint8_t>().swap(oracle_pre);
            }
            // Task 294 accept oracle (diagnostic, flag-gated): the first round's real T2 accepted
            // [pending, draft] at pos..pos+1.  Capture the actual ACT outputs - both logits rows and both
            // nextn hidden rows (deep copies, taken before any later target decode can refill the buffers)
            // and the actual target logical state - then restore the PRE snapshot (exact returned size
            // required) and replay the same accepted prefix serially with REAL T1 decodes: pending@pos
            // (row0 head+hidden captured before the second decode) then the same accepted draft@pos+1
            // (final head+hidden+state).  REF has no speculative row, so there is no rm/rollback on the
            // reference side.  The request stops with stop=accepted_oracle_probe; the accepted pair is
            // never replayed against the draft context (drafter replay not executed) and the ACT snapshot
            // is never restored to fake an ongoing business generation.
            if (accept_probe_armed) {
                const float* arow0 = llama_get_logits_ith(tgt, 0);
                const float* arow1 = llama_get_logits_ith(tgt, 1);
                if (arow0 == nullptr || arow1 == nullptr) return fail("accept oracle ACT logits row null");
                oracle_act_row0.assign(arow0, arow0 + (size_t) n_vocab);
                oracle_act_logits.assign(arow1, arow1 + (size_t) n_vocab);
                oracle_act_h0 = h_row0;
                oracle_act_h = h_row1;
                std::string oserr;
                size_t act_size = 0;
                if (!oracle_state_save(tgt, 0, &oracle_act, &act_size, &oserr))
                    return fail("accept oracle ACT state save failed: " + oserr);
                const int64_t probe_no = g_oracle_probes_done + 1;
                const std::string odir = accept_oracle_dir != nullptr ? std::string(accept_oracle_dir)
                                                                      : std::string();
                auto accept_export_one = [&](const char* what, const std::vector<uint8_t>& buf) {
                    std::string ex_err = odir.empty() ? "STRATA_QWEN35_MTP_ORACLE_DIR unset" : "";
                    bool wrote = false;
                    if (!odir.empty()) {
                        wrote = oracle_export(odir, std::string("oracle-p") + std::to_string(probe_no) + "-" +
                                              what + "-seq0.bin", buf, &ex_err);
                    }
                    std::printf("{\"event\":\"qwen35_mtp_txn_oracle_export\",\"probe\":%lld,\"what\":\"%s\","
                                "\"bytes\":%zu,\"ok\":%s,\"note\":\"%s\"}\n",
                                (long long) probe_no, what, buf.size(), wrote ? "true" : "false",
                                ex_err.c_str());
                    std::fflush(stdout);
                };
                accept_export_one("pre", oracle_pre);
                accept_export_one("act", oracle_act);
                // REF: restore the SAME PRE snapshot (state_read resets rs_idx to 0); free PRE before the
                // REF capture so at most two state blocks are live.
                size_t setret = 0;
                if (!oracle_state_load(tgt, 0, oracle_pre, &setret))
                    return fail("accept oracle PRE restore (llama_state_seq_set_data) failed");
                const bool restore_size_ok = (setret == oracle_pre.size());
                std::vector<uint8_t>().swap(oracle_pre);
                const llama_token ref_pending = pending;
                const llama_token ref_draft = draft_id;
                if (!decode_rows(tgt, &ref_pending, nullptr, 1, pos, true, &rc) || rc != 0)
                    return fail("accept oracle REF serial T1 pending decode failed");
                const float* ref0_lg = llama_get_logits_ith(tgt, 0);
                if (ref0_lg == nullptr) return fail("accept oracle REF pending logits null");
                oracle_ref_row0.assign(ref0_lg, ref0_lg + (size_t) n_vocab);
                const llama_token ref_t0 = greedy_argmax(oracle_ref_row0.data(), n_vocab, &ok);
                if (!ok) return fail("accept oracle REF pending logits not finite");
                oracle_ref_h0 = copy_nextn_row(tgt, 0, &ok);
                if (!ok) return fail("accept oracle REF pending nextn hidden not finite");
                if (!decode_rows(tgt, &ref_draft, nullptr, 1, pos + 1, true, &rc) || rc != 0)
                    return fail("accept oracle REF serial T1 draft decode failed");
                const float* ref_lg = llama_get_logits_ith(tgt, 0);
                if (ref_lg == nullptr) return fail("accept oracle REF logits null");
                oracle_ref_logits.assign(ref_lg, ref_lg + (size_t) n_vocab);
                const llama_token ref_next = greedy_argmax(oracle_ref_logits.data(), n_vocab, &ok);
                if (!ok) return fail("accept oracle REF logits not finite");
                oracle_ref_h = copy_nextn_row(tgt, 0, &ok);
                if (!ok) return fail("accept oracle REF nextn hidden not finite");
                size_t ref_size = 0;
                if (!oracle_state_save(tgt, 0, &oracle_ref, &ref_size, &oserr))
                    return fail("accept oracle REF state save failed: " + oserr);
                accept_export_one("ref", oracle_ref);

                const llama_token act_next = greedy_argmax(oracle_act_logits.data(), n_vocab, &ok);
                if (!ok) return fail("accept oracle ACT row1 logits not finite");
                const OracleRowDiff d_next = oracle_diff_elems((const uint8_t*) oracle_act_logits.data(),
                                                               (const uint8_t*) oracle_ref_logits.data(),
                                                               n_vocab, 0, 4);
                const OracleRowDiff d_row0 = oracle_diff_elems((const uint8_t*) oracle_act_row0.data(),
                                                               (const uint8_t*) oracle_ref_row0.data(),
                                                               n_vocab, 0, 4);
                const OracleRowDiff d_hid = oracle_diff_elems((const uint8_t*) oracle_act_h.data(),
                                                              (const uint8_t*) oracle_ref_h.data(),
                                                              n_embd_out, 0, 4);
                const OracleRowDiff d_hid0 = oracle_diff_elems((const uint8_t*) oracle_act_h0.data(),
                                                               (const uint8_t*) oracle_ref_h0.data(),
                                                               n_embd_out, 0, 4);
                const bool size_ok = (act_size == ref_size);
                std::vector<OracleTensor> st_tensors;
                std::string st_note;
                int64_t st_first = -1;
                int64_t st_ndiff = 0;
                bool st_byte_equal = false;
                int st_rc = 2;
                if (size_ok) {
                    st_rc = oracle_compare_state_bytes(oracle_act, oracle_ref, &st_tensors, &st_note, &st_first,
                                                       &st_ndiff, &st_byte_equal);
                } else {
                    st_note = "act/ref serialized size mismatch";
                }
                OracleRowDiff st_acc;
                for (const auto& t : st_tensors) oracle_accum(&st_acc, t.d);
                const char* st_status = (st_rc == 0 || st_rc == 1)
                                            ? (st_byte_equal ? "BYTE_EQUAL" : "TYPED_COMPARED")
                                            : "PENDING_DECODER";
                const bool greedy_next_ok = (act_next == t1) && (ref_next == t1);
                const bool ref_t0_ok = (ref_t0 == t0);
                const bool finite_ok = (d_next.nonfinite == 0 && d_hid.nonfinite == 0 &&
                                        d_row0.nonfinite == 0 && d_hid0.nonfinite == 0 && st_acc.nonfinite == 0);
                auto rms = [](const OracleRowDiff& d) {
                    return d.n > 0 ? std::sqrt(d.sumsq / (double) d.n) : 0.0;
                };
                auto l1rel = [](const OracleRowDiff& d) {
                    return d.l1 / (d.ref_l1 > 0.0 ? d.ref_l1 : 1.0);
                };
                std::printf("{\"event\":\"qwen35_mtp_txn_oracle_compare\",\"probe\":%lld,\"round\":%lld,"
                            "\"pos\":%lld,\"accept\":1,\"pending\":%d,\"draft\":%d,\"t0\":%d,\"t1\":%d,"
                            "\"act_next\":%d,\"ref_next\":%d,\"ref_t0\":%d,\"state_size\":%zu,"
                            "\"act_state_size\":%zu,\"ref_state_size\":%zu,\"restore_ret\":%zu,"
                            "\"restore_size_ok\":%s,\"probe_kind\":\"accepted_t2_vs_serial_t1_target_only\","
                            "\"drafter_replay_executed\":false,"
                            "\"gates\":{\"greedy_next_equal\":%s,\"ref_t0_equal\":%s,\"finite\":%s},"
                            "\"logits_next\":{\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,"
                            "\"rms\":%.9g,\"l1_rel\":%.9g},\"hidden_nextn\":{\"n\":%lld,\"neq\":%lld,"
                            "\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g,\"l1_rel\":%.9g},"
                            "\"row0\":{\"present\":true,\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,"
                            "\"rms\":%.9g,\"l1_rel\":%.9g},\"hidden_row0\":{\"n\":%lld,\"neq\":%lld,"
                            "\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g,\"l1_rel\":%.9g},"
                            "\"state\":{\"status\":\"%s\",\"byte_equal\":%s,\"tensors\":%zu,\"elems\":%lld,"
                            "\"neq\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g,\"first_diff\":%lld,"
                            "\"diff_bytes\":%lld,\"note\":\"%s\"},"
                            "\"matched\":{\"on\":false,\"pair\":\"serial_ref\","
                            "\"row0_pair\":\"act_t2_vs_ref_t1_pending\"}}\n",
                            (long long) probe_no, (long long) round_idx, (long long) pos, (int) pending,
                            (int) draft_id, (int) t0, (int) t1, (int) act_next, (int) ref_next, (int) ref_t0,
                            oracle_state_size, act_size, ref_size, setret,
                            restore_size_ok ? "true" : "false", greedy_next_ok ? "true" : "false",
                            ref_t0_ok ? "true" : "false", finite_ok ? "true" : "false",
                            (long long) d_next.n, (long long) d_next.neq, d_next.max_abs, d_next.l1,
                            rms(d_next), l1rel(d_next), (long long) d_hid.n, (long long) d_hid.neq,
                            d_hid.max_abs, d_hid.l1, rms(d_hid), l1rel(d_hid), (long long) d_row0.n,
                            (long long) d_row0.neq, d_row0.max_abs, d_row0.l1, rms(d_row0), l1rel(d_row0),
                            (long long) d_hid0.n, (long long) d_hid0.neq, d_hid0.max_abs, d_hid0.l1,
                            rms(d_hid0), l1rel(d_hid0), st_status, st_byte_equal ? "true" : "false",
                            st_tensors.size(), (long long) st_acc.n, (long long) st_acc.neq, st_acc.max_abs,
                            st_acc.l1, rms(st_acc), (long long) st_first, (long long) st_ndiff,
                            st_note.c_str());
                std::fflush(stdout);
                if (st_rc == 1) {
                    for (const auto& t : st_tensors) {
                        std::printf("{\"event\":\"qwen35_mtp_txn_oracle_state_tensor\",\"probe\":%lld,"
                                    "\"name\":\"%s\",\"dtype\":\"%s\",\"n\":%lld,\"neq\":%lld,"
                                    "\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g}\n",
                                    (long long) probe_no, t.name.c_str(), t.dtype, (long long) t.d.n,
                                    (long long) t.d.neq, t.d.max_abs, t.d.l1, rms(t.d));
                    }
                    std::fflush(stdout);
                }
                if (!restore_size_ok) return fail("accept oracle PRE restore size mismatch (returnsize not exact)");
                if (!ref_t0_ok) return fail("accept oracle REF pending greedy token mismatch (restore/replay)");
                if (!greedy_next_ok) return fail("accept oracle ACT/REF greedy next token mismatch");
                if (!finite_ok) return fail("accept oracle ACT/REF non-finite numeric difference");
                // Both trajectories agree on the committed prefix; the live target state is REF's (the same
                // accepted prefix decoded serially).  Stop here with the diagnostic stop - the accepted
                // pair is NOT replayed against the draft context and the ACT snapshot is never restored to
                // fake an ongoing business generation.
                if (!llama_vocab_is_eog(vocab, draft_id)) {
                    emit_one(draft_id, pos + 1, "draft", round_idx);
                }
                ++g_oracle_probes_done;
                stop = "accepted_oracle_probe";
                break;
            }
            // Accept: draft_id == t0 is the real greedy token at pos+1.  EOS stops without emitting it and
            // without any further replay; the target KV already holds the accepted prefix.
            if (llama_vocab_is_eog(vocab, draft_id)) {
                if (draft_state_oracle_on && ds_armed && round_idx == ds_probe_round)
                    return ds_no_go("accepted_eog_before_endpoint");
                stop = "accepted_eog"; eog_token = draft_id; ++eos_count; break;
            }
            emit_one(draft_id, pos + 1, "draft", round_idx);
            if (emitted >= max_new) {
                if (draft_state_oracle_on && ds_armed && round_idx == ds_probe_round)
                    return ds_no_go("max_new_before_endpoint");
                stop = "max_new"; break;
            }
            // Draft KV sync of the accepted prefix: decode the accepted token at pos+1 WITH the deep-copied
            // target hidden after pending@pos (token+hidden, never token-only).  Its own logits are unused.
            const llama_token replay = draft_id;
            const auto replay_phase_done = cost_phase(4, -1);
            if (!decode_rows(draft.get(), &replay, &h_row0, 1, pos + 1, false, &rc) || rc != 0)
                return fail("draft KV sync of the accepted prefix failed");
            replay_phase_done();
            pending = t1;
            pending_h = std::move(h_row1);
            pending_role = "bonus";
            pos += dpos;   // 2
        } else {
            // Reject: the corrective t0 becomes the pending token and is emitted at the next round start,
            // exactly once.  T2 removed the speculative draft row it had decoded; the serial path never
            // decoded one (rollback = 0), so there is nothing to remove and seq_rm is skipped.
            if (rollback > 0) {
                const auto rollback_phase_done = cost_phase(5, 0);
                if (!llama_memory_seq_rm(tmem, 0, pos + 1, -1)) return fail("target rollback refused");
                rollback_phase_done();
            }

            // Task 277 rollback state oracle (diagnostic prefix probe, default OFF).  ACT = the real
            // rollback above followed by the real next corrective decode (T1 at pos+1), which consumes the
            // pending recurrent rollback index exactly like the original transaction's next target decode.
            // REF = restore the pre-verifier state and replay the same committed prefix serially (T1
            // pending@pos, then T1 the same corrective@pos+1).  It emits evidence and stops generation
            // (stop=rollback_oracle_probe); it never restores ACT to fake a continuation.
            if (oracle_pre_saved) {
                const llama_token oracle_corr = t0;
                const llama_token oracle_pending = pending;
                if (!decode_rows(tgt, &oracle_corr, nullptr, 1, pos + 1, true, &rc) || rc != 0)
                    return fail("oracle ACT corrective T1 decode failed");
                const float* act_lg = llama_get_logits_ith(tgt, 0);
                if (act_lg == nullptr) return fail("oracle ACT logits null");
                oracle_act_logits.assign(act_lg, act_lg + (size_t) n_vocab);
                const llama_token act_next = greedy_argmax(oracle_act_logits.data(), n_vocab, &ok);
                if (!ok) return fail("oracle ACT logits not finite");
                oracle_act_h = copy_nextn_row(tgt, 0, &ok);
                if (!ok) return fail("oracle ACT nextn hidden not finite");
                std::string oserr;
                size_t act_size = 0;
                if (!oracle_state_save(tgt, 0, &oracle_act, &act_size, &oserr))
                    return fail("oracle ACT state save failed: " + oserr);

                const int64_t probe_no = g_oracle_probes_done + 1;
                const std::string odir = oracle_dir != nullptr ? std::string(oracle_dir) : std::string();
                auto oracle_export_one = [&](const char* what, const std::vector<uint8_t>& buf) {
                    std::string ex_err = odir.empty() ? "STRATA_QWEN35_MTP_ORACLE_DIR unset" : "";
                    bool wrote = false;
                    if (!odir.empty()) {
                        wrote = oracle_export(odir, std::string("oracle-p") + std::to_string(probe_no) + "-" +
                                              what + "-seq0.bin", buf, &ex_err);
                    }
                    std::printf("{\"event\":\"qwen35_mtp_txn_oracle_export\",\"probe\":%lld,\"what\":\"%s\","
                                "\"bytes\":%zu,\"ok\":%s,\"note\":\"%s\"}\n",
                                (long long) probe_no, what, buf.size(), wrote ? "true" : "false", ex_err.c_str());
                    std::fflush(stdout);
                };
                oracle_export_one("pre", oracle_pre);
                oracle_export_one("act", oracle_act);

                // REF / CONTROL: restore the pre-verifier state (state_read resets rs_idx to 0).
                //  - matched_t2_on: the shape-matched CONTROL trajectory.  A legal target token-only T2
                //    [pending, corrective t0] - the same batch shape as the ACT wrong-draft T2; only the
                //    future speculative row input differs (corrective vs the injected wrong draft).  Row0 of
                //    this T2 is the matched counterpart of the ACT T2 row0.  Then the same real seq_rm(pos+1)
                //    the ACT side performed removes the just-decoded corrective row, and the real T1
                //    corrective@pos+1 consumes the control rs_idx.  CONTROL also contains the rm, so this is
                //    a falsifiable supplement on the speculative input isolation - it is NOT independent
                //    ground truth and does not replace the serial REF path or the full state G1 gate.
                //  - otherwise: the original serial REF (T1 pending@pos then T1 the SAME corrective@pos+1).
                size_t setret = 0;
                if (!oracle_state_load(tgt, 0, oracle_pre, &setret))
                    return fail("oracle PRE restore (llama_state_seq_set_data) failed");
                const bool restore_size_ok = (setret == oracle_pre.size());
                std::vector<uint8_t>().swap(oracle_pre);   // free before the REF capture (two live blocks)

                const llama_token ref_corr = oracle_corr;
                llama_token ref_t0 = -1;
                llama_token ref_next = -1;
                bool ctl_seq_rm_ok = false;
                if (matched_t2_on) {
                    const llama_token ctl_ids[2] = {oracle_pending, oracle_corr};
                    if (!decode_rows(tgt, ctl_ids, nullptr, 2, pos, true, &rc) || rc != 0)
                        return fail("oracle CONTROL corrective T2 decode failed");
                    const float* ctl0_lg = llama_get_logits_ith(tgt, 0);
                    if (ctl0_lg == nullptr) return fail("oracle CONTROL T2 row0 logits null");
                    oracle_ref_row0.assign(ctl0_lg, ctl0_lg + (size_t) n_vocab);
                    ref_t0 = greedy_argmax(oracle_ref_row0.data(), n_vocab, &ok);
                    if (!ok) return fail("oracle CONTROL T2 row0 logits not finite");
                    // The same real removal the ACT side performed (ACT removed its speculative draft row;
                    // this removes the just-verified corrective row at the same position, pos+1).
                    ctl_seq_rm_ok = llama_memory_seq_rm(tmem, 0, pos + 1, -1);
                    if (!ctl_seq_rm_ok) return fail("oracle CONTROL seq_rm(pos+1) refused");
                    if (!decode_rows(tgt, &ref_corr, nullptr, 1, pos + 1, true, &rc) || rc != 0)
                        return fail("oracle CONTROL corrective T1 decode failed");
                    const float* ctl_lg = llama_get_logits_ith(tgt, 0);
                    if (ctl_lg == nullptr) return fail("oracle CONTROL logits null");
                    oracle_ref_logits.assign(ctl_lg, ctl_lg + (size_t) n_vocab);
                    ref_next = greedy_argmax(oracle_ref_logits.data(), n_vocab, &ok);
                    if (!ok) return fail("oracle CONTROL logits not finite");
                    oracle_ref_h = copy_nextn_row(tgt, 0, &ok);
                    if (!ok) return fail("oracle CONTROL nextn hidden not finite");
                } else {
                    if (!decode_rows(tgt, &oracle_pending, nullptr, 1, pos, true, &rc) || rc != 0)
                        return fail("oracle REF serial T1 pending decode failed");
                    const float* ref0_lg = llama_get_logits_ith(tgt, 0);
                    if (ref0_lg == nullptr) return fail("oracle REF pending logits null");
                    oracle_ref_row0.assign(ref0_lg, ref0_lg + (size_t) n_vocab);
                    ref_t0 = greedy_argmax(oracle_ref_row0.data(), n_vocab, &ok);
                    if (!ok) return fail("oracle REF pending logits not finite");
                    if (!decode_rows(tgt, &ref_corr, nullptr, 1, pos + 1, true, &rc) || rc != 0)
                        return fail("oracle REF corrective T1 decode failed");
                    const float* ref_lg = llama_get_logits_ith(tgt, 0);
                    if (ref_lg == nullptr) return fail("oracle REF logits null");
                    oracle_ref_logits.assign(ref_lg, ref_lg + (size_t) n_vocab);
                    ref_next = greedy_argmax(oracle_ref_logits.data(), n_vocab, &ok);
                    if (!ok) return fail("oracle REF logits not finite");
                    oracle_ref_h = copy_nextn_row(tgt, 0, &ok);
                    if (!ok) return fail("oracle REF nextn hidden not finite");
                }
                size_t ref_size = 0;
                if (!oracle_state_save(tgt, 0, &oracle_ref, &ref_size, &oserr))
                    return fail("oracle REF state save failed: " + oserr);
                oracle_export_one("ref", oracle_ref);

                // Numeric compare: final next logits + nextn hidden (both are T1@pos+1 on the same prefix
                // and dtype), the intermediate row0 (ACT T2 row0 vs REF T1 pending) and the serialized
                // sequence state walked in lockstep by its own typed layout.
                const OracleRowDiff d_next = oracle_diff_elems((const uint8_t*) oracle_act_logits.data(),
                                                               (const uint8_t*) oracle_ref_logits.data(),
                                                               n_vocab, 0, 4);
                const OracleRowDiff d_hid = oracle_diff_elems((const uint8_t*) oracle_act_h.data(),
                                                              (const uint8_t*) oracle_ref_h.data(),
                                                              n_embd_out, 0, 4);
                OracleRowDiff d_row0;
                bool row0_ok = false;
                if ((int64_t) oracle_act_row0.size() == n_vocab && (int64_t) oracle_ref_row0.size() == n_vocab) {
                    d_row0 = oracle_diff_elems((const uint8_t*) oracle_act_row0.data(),
                                               (const uint8_t*) oracle_ref_row0.data(), n_vocab, 0, 4);
                    row0_ok = true;
                }
                // Task 283: ACT and REF are the compared pair and must match each other; PRE is an
                // independent pre-verify snapshot and is only logged (task 277: PRE legitimately holds
                // fewer attn cells - 3896 vs 3898 - so it cannot gate the ACT/REF comparison).
                const bool size_ok = (act_size == ref_size);
                std::vector<OracleTensor> st_tensors;
                std::string st_note;
                int64_t st_first = -1;
                int64_t st_ndiff = 0;
                bool st_byte_equal = false;
                int st_rc = 2;
                if (size_ok) {
                    st_rc = oracle_compare_state_bytes(oracle_act, oracle_ref, &st_tensors, &st_note, &st_first,
                                                       &st_ndiff, &st_byte_equal);
                } else {
                    st_note = "act/ref serialized size mismatch";
                }
                OracleRowDiff st_acc;
                for (const auto& t : st_tensors) oracle_accum(&st_acc, t.d);
                const char* st_status = (st_rc == 0 || st_rc == 1) ? (st_byte_equal ? "BYTE_EQUAL" : "TYPED_COMPARED")
                                                                   : "PENDING_DECODER";
                const bool greedy_next_ok = (act_next == ref_next);
                const bool ref_t0_ok = (ref_t0 == t0);
                const bool finite_ok = (d_next.nonfinite == 0 && d_hid.nonfinite == 0 &&
                                        (!row0_ok || d_row0.nonfinite == 0) && st_acc.nonfinite == 0);
                auto rms = [](const OracleRowDiff& d) {
                    return d.n > 0 ? std::sqrt(d.sumsq / (double) d.n) : 0.0;
                };
                auto l1rel = [](const OracleRowDiff& d) {
                    return d.l1 / (d.ref_l1 > 0.0 ? d.ref_l1 : 1.0);
                };
                std::printf("{\"event\":\"qwen35_mtp_txn_oracle_compare\",\"probe\":%lld,\"round\":%lld,"
                            "\"pos\":%lld,\"original_draft\":%d,\"injected_id\":%d,\"t0\":%d,\"act_next\":%d,"
                            "\"ref_next\":%d,\"ref_t0\":%d,\"state_size\":%zu,\"act_state_size\":%zu,"
                            "\"ref_state_size\":%zu,\"restore_ret\":%zu,\"restore_size_ok\":%s,"
                            "\"gates\":{\"greedy_next_equal\":%s,\"ref_t0_equal\":%s,\"finite\":%s},"
                            "\"logits_next\":{\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g,"
                            "\"l1_rel\":%.9g},\"hidden_nextn\":{\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,"
                            "\"l1\":%.9g,\"rms\":%.9g,\"l1_rel\":%.9g},\"row0\":{\"present\":%s,\"n\":%lld,"
                            "\"neq\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g,\"l1_rel\":%.9g},"
                            "\"state\":{\"status\":\"%s\",\"byte_equal\":%s,\"tensors\":%zu,\"elems\":%lld,"
                            "\"neq\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g,\"first_diff\":%lld,"
                            "\"diff_bytes\":%lld,\"note\":\"%s\"},"
                            "\"matched\":{\"on\":%s,\"pair\":\"%s\",\"row0_pair\":\"%s\","
                            "\"act_t2_inputs\":[%d,%d],\"act_seq_rm_rows\":%lld,\"act_t1_input\":%d,"
                            "\"act_t1_pos\":%lld,\"ctl_t2_inputs\":[%d,%d],\"ctl_t2_pos\":%lld,"
                            "\"ctl_seq_rm\":%s,\"ctl_rm_pos\":%lld,\"ctl_t1_input\":%d,"
                            "\"ctl_t1_pos\":%lld}}\n",
                            (long long) probe_no, (long long) round_idx, (long long) pos, (int) original_draft,
                            (int) draft_id, (int) t0, (int) act_next, (int) ref_next, (int) ref_t0,
                            oracle_state_size, act_size, ref_size, setret, restore_size_ok ? "true" : "false",
                            greedy_next_ok ? "true" : "false", ref_t0_ok ? "true" : "false",
                            finite_ok ? "true" : "false", (long long) d_next.n, (long long) d_next.neq,
                            d_next.max_abs, d_next.l1, rms(d_next), l1rel(d_next), (long long) d_hid.n,
                            (long long) d_hid.neq, d_hid.max_abs, d_hid.l1, rms(d_hid), l1rel(d_hid),
                            row0_ok ? "true" : "false", (long long) d_row0.n, (long long) d_row0.neq,
                            d_row0.max_abs, d_row0.l1, rms(d_row0), l1rel(d_row0), st_status,
                            st_byte_equal ? "true" : "false", st_tensors.size(), (long long) st_acc.n,
                            (long long) st_acc.neq, st_acc.max_abs, st_acc.l1, rms(st_acc), (long long) st_first,
                            (long long) st_ndiff, st_note.c_str(),
                            matched_t2_on ? "true" : "false",
                            matched_t2_on ? "matched_t2_ctl" : "serial_ref",
                            matched_t2_on ? "act_t2_vs_ctl_t2" : "act_t2_vs_ref_t1_pending",
                            (int) oracle_pending, (int) draft_id, (long long) rollback, (int) oracle_corr,
                            (long long) (pos + 1), (int) oracle_pending, (int) oracle_corr, (long long) pos,
                            ctl_seq_rm_ok ? "true" : "false", (long long) (pos + 1), (int) oracle_corr,
                            (long long) (pos + 1));
                std::fflush(stdout);
                if (st_rc == 1) {
                    for (const auto& t : st_tensors) {
                        std::printf("{\"event\":\"qwen35_mtp_txn_oracle_state_tensor\",\"probe\":%lld,"
                                    "\"name\":\"%s\",\"dtype\":\"%s\",\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,"
                                    "\"l1\":%.9g,\"rms\":%.9g}\n",
                                    (long long) probe_no, t.name.c_str(), t.dtype, (long long) t.d.n,
                                    (long long) t.d.neq, t.d.max_abs, t.d.l1, rms(t.d));
                    }
                    std::fflush(stdout);
                }
                if (!greedy_next_ok || !ref_t0_ok) {
                    return fail(ref_t0_ok ? "oracle ACT/REF greedy next token mismatch"
                                          : "oracle REF pending greedy token mismatch (restore/replay)");
                }
                if (!finite_ok) return fail("oracle ACT/REF non-finite numeric difference");
                // The committed corrective is visible now (both trajectories agree on the prefix); stop the
                // request here with the diagnostic stop - the live state is REF's, and the next request
                // re-prefills explicitly, so nothing pretends to continue from ACT.
                emit_one(oracle_corr, pos + 1, "corrective", round_idx);
                ++g_oracle_probes_done;
                stop = "rollback_oracle_probe";
                break;
            }

            pending = t0;
            pending_h = std::move(h_row0);
            pending_role = "corrective";
            pos += dpos;   // 1
        }

        // Task 317 draft-state oracle endpoint (flag on + armed + this probe round).  The CASE must match the
        // REAL branch (accept: natural accept; reject: real injected reject) and the actual next-consumer
        // call path must exist (not EOG, not the context tail, budget left) - otherwise an explicit NO_GO,
        // never a manufactured input.  ACT consumer -> capture; REF restore PRE on the same draft ctx and
        // replay only the same committed inputs -> capture; compare -> diagnostic stop.
        if (draft_state_oracle_on && ds_armed && round_idx == ds_probe_round && !ds_completed) {
            const bool ds_case_ok = draft_oracle_accept ? accepted : (injected_this_round && !accepted);
            if (!ds_case_ok)
                return ds_no_go(draft_oracle_accept ? "round1_not_natural_accept"
                                                    : "round1_not_real_injected_reject");
            if (llama_vocab_is_eog(vocab, pending)) return ds_no_go("next_pending_eog_no_consumer");
            if (pos >= n_ctx) return ds_no_go("next_pos_at_ctx_limit");
            if (pos + 1 >= n_ctx) return ds_no_go("next_round_is_ctx_tail_no_propose");
            if (emitted >= max_new) return ds_no_go("max_new_budget_no_consumer");
            const llama_token ds_next_tok = pending;          // t1 (accept) / t0 (reject)
            const std::vector<float> ds_next_h = pending_h;   // h_row1 (accept) / h_row0 (reject) deep copy
            const int64_t ds_next_pos = pos;                  // pos+2 (accept) / pos+1 (reject)
            const std::vector<float> ds_h0 = h_row0;          // target row0 hidden (accept REF replay input)
            // ACT: the ONE actual next-consumer proposal decode on the draft ctx - exactly the call the next
            // round would run with these inputs.
            int32_t ds_rc = -1;
            if (!decode_rows(draft.get(), &ds_next_tok, &ds_next_h, 1, ds_next_pos, true, &ds_rc) || ds_rc != 0)
                return fail("draft state oracle ACT consumer decode failed");
            const float* ds_act_lg = llama_get_logits_ith(draft.get(), -1);
            if (ds_act_lg == nullptr) return fail("draft state oracle ACT consumer logits null");
            ds_act_logits.assign(ds_act_lg, ds_act_lg + (size_t) n_vocab);
            const llama_token ds_act_next = greedy_argmax(ds_act_logits.data(), n_vocab, &ok);
            if (!ok) return fail("draft state oracle ACT consumer head not finite");
            ds_act_h = copy_nextn_row(draft.get(), 0, &ok);
            if (!ok) return fail("draft state oracle ACT consumer hidden not finite");
            if ((int64_t) ds_act_h.size() != n_embd_out || ds_all_zero(ds_act_h))
                return ds_no_go("act_hidden_unavailable_or_all_zero");
            std::string ds_err;
            size_t ds_act_sz = 0;
            if (!ds_state_save(&ds_act, &ds_act_sz, &ds_err))
                return fail("draft state oracle ACT state save failed: " + ds_err);
            llama_memory_t ds_dmem = llama_get_memory(draft.get());
            const llama_pos ds_act_pmin = ds_dmem != nullptr ? llama_memory_seq_pos_min(ds_dmem, 0) : -1;
            const llama_pos ds_act_pmax = ds_dmem != nullptr ? llama_memory_seq_pos_max(ds_dmem, 0) : -1;
            const int64_t ds_probe_no = g_draft_oracle_probes_done + 1;
            const std::string ds_odir = std::string(draft_oracle_dir);
            auto ds_export_one = [&](const char* what, const std::vector<uint8_t>& buf) -> bool {
                std::string ex_err;
                const bool wrote = oracle_export(ds_odir, std::string("draft-oracle-p") +
                                                                     std::to_string(ds_probe_no) + "-" + what +
                                                                     "-seq0.bin",
                                                 buf, &ex_err);
                std::printf("{\"event\":\"qwen35_mtp_txn_draft_state_oracle_export\",\"probe\":%lld,"
                            "\"what\":\"%s\",\"bytes\":%zu,\"ok\":%s,\"note\":\"%s\"}\n",
                            (long long) ds_probe_no, what, buf.size(), wrote ? "true" : "false",
                            ex_err.c_str());
                std::fflush(stdout);
                return wrote;
            };
            if (!ds_export_one("pre", ds_pre)) return fail("draft state oracle PRE export failed");
            if (!ds_export_one("act", ds_act)) return fail("draft state oracle ACT export failed");
            // REF: restore the SAME PRE snapshot on the SAME draft context (real llama_state_seq_set_data;
            // exact returned size required), free PRE before the REF capture, then replay only the same
            // committed token/hidden inputs - pending@pos with the captured pending_h; the accepted token
            // @pos+1 with target h_row0 on accept (logits unused, exactly like the real replay); finally the
            // same consumer decode.
            size_t ds_setret = 0;
            if (!oracle_state_load(draft.get(), 0, ds_pre, &ds_setret))
                return fail("draft state oracle PRE restore (llama_state_seq_set_data) failed");
            if (ds_setret != ds_pre.size())
                return fail("draft state oracle PRE restore size mismatch (return size not exact)");
            const size_t ds_pre_sz = ds_pre.size();
            std::vector<uint8_t>().swap(ds_pre);
            if (!decode_rows(draft.get(), &ds_pending0, &ds_pending_h0, 1, ds_pos0, true, &ds_rc) || ds_rc != 0)
                return fail("draft state oracle REF propose replay decode failed");
            const float* ds_ref_prop_lg = llama_get_logits_ith(draft.get(), -1);
            if (ds_ref_prop_lg == nullptr) return fail("draft state oracle REF propose logits null");
            ds_ref_propose_logits.assign(ds_ref_prop_lg, ds_ref_prop_lg + (size_t) n_vocab);
            const llama_token ds_ref_propose = greedy_argmax(ds_ref_propose_logits.data(), n_vocab, &ok);
            if (!ok) return fail("draft state oracle REF propose logits not finite");
            if (draft_oracle_accept) {
                const llama_token ds_ref_accepted = draft_id;   // mode 0: the real accepted token
                const std::vector<float> ds_h0_copy = ds_h0;
                if (!decode_rows(draft.get(), &ds_ref_accepted, &ds_h0_copy, 1, ds_pos0 + 1, false, &ds_rc) ||
                    ds_rc != 0)
                    return fail("draft state oracle REF accepted replay decode failed");
            }
            const llama_token ds_ref_next_tok = ds_next_tok;
            const std::vector<float> ds_next_h_copy = ds_next_h;
            if (!decode_rows(draft.get(), &ds_ref_next_tok, &ds_next_h_copy, 1, ds_next_pos, true, &ds_rc) ||
                ds_rc != 0)
                return fail("draft state oracle REF consumer decode failed");
            const float* ds_ref_lg = llama_get_logits_ith(draft.get(), -1);
            if (ds_ref_lg == nullptr) return fail("draft state oracle REF consumer logits null");
            ds_ref_logits.assign(ds_ref_lg, ds_ref_lg + (size_t) n_vocab);
            const llama_token ds_ref_next = greedy_argmax(ds_ref_logits.data(), n_vocab, &ok);
            if (!ok) return fail("draft state oracle REF consumer head not finite");
            ds_ref_h = copy_nextn_row(draft.get(), 0, &ok);
            if (!ok) return fail("draft state oracle REF consumer hidden not finite");
            if ((int64_t) ds_ref_h.size() != n_embd_out || ds_all_zero(ds_ref_h))
                return ds_no_go("ref_hidden_unavailable_or_all_zero");
            size_t ds_ref_sz = 0;
            if (!ds_state_save(&ds_ref, &ds_ref_sz, &ds_err))
                return fail("draft state oracle REF state save failed: " + ds_err);
            if (!ds_export_one("ref", ds_ref)) return fail("draft state oracle REF export failed");
            const llama_pos ds_ref_pmin = ds_dmem != nullptr ? llama_memory_seq_pos_min(ds_dmem, 0) : -1;
            const llama_pos ds_ref_pmax = ds_dmem != nullptr ? llama_memory_seq_pos_max(ds_dmem, 0) : -1;
            // Numeric + raw-state compare.  The serialized draft state byte equality is mandatory for a
            // scope PASS; a typed number is only detail when the bytes differ.
            const OracleRowDiff ds_d_next = oracle_diff_elems((const uint8_t*) ds_act_logits.data(),
                                                              (const uint8_t*) ds_ref_logits.data(),
                                                              n_vocab, 0, 4);
            const OracleRowDiff ds_d_hid = oracle_diff_elems((const uint8_t*) ds_act_h.data(),
                                                             (const uint8_t*) ds_ref_h.data(),
                                                             n_embd_out, 0, 4);
            const bool ds_prop_present = ((int64_t) ds_act_propose_logits.size() == n_vocab &&
                                          (int64_t) ds_ref_propose_logits.size() == n_vocab);
            if (!ds_prop_present) return ds_no_go("propose_logits_unavailable");
            const OracleRowDiff ds_d_prop = oracle_diff_elems((const uint8_t*) ds_act_propose_logits.data(),
                                                              (const uint8_t*) ds_ref_propose_logits.data(),
                                                              n_vocab, 0, 4);
            const llama_token ds_act_prop_next = greedy_argmax(ds_act_propose_logits.data(), n_vocab, &ok);
            if (!ok) return fail("draft state oracle ACT propose logits not finite");
            const bool ds_size_ok = (ds_act_sz == ds_ref_sz);
            std::vector<OracleTensor> ds_tensors;
            std::string ds_note;
            int64_t ds_first = -1;
            int64_t ds_ndiff = 0;
            bool ds_byte_equal = false;
            int ds_st_rc = 2;
            if (ds_size_ok) {
                ds_st_rc = oracle_compare_state_bytes(ds_act, ds_ref, &ds_tensors, &ds_note, &ds_first,
                                                      &ds_ndiff, &ds_byte_equal);
            } else {
                ds_note = "act/ref serialized size mismatch";
            }
            OracleRowDiff ds_st_acc;
            for (const auto& t : ds_tensors) oracle_accum(&ds_st_acc, t.d);
            const char* ds_st_status = (ds_st_rc == 0 || ds_st_rc == 1)
                                           ? (ds_byte_equal ? "BYTE_EQUAL" : "TYPED_COMPARED")
                                           : "PENDING_DECODER";
            auto ds_rms = [](const OracleRowDiff& d) {
                return d.n > 0 ? std::sqrt(d.sumsq / (double) d.n) : 0.0;
            };
            auto ds_l1rel = [](const OracleRowDiff& d) {
                return d.l1 / (d.ref_l1 > 0.0 ? d.ref_l1 : 1.0);
            };
            const bool ds_prop_repro = (ds_act_prop_next == original_draft) &&
                                       (ds_ref_propose == original_draft);
            const bool ds_greedy_ok = (ds_act_next == ds_ref_next);
            const bool ds_finite_ok = (ds_d_next.nonfinite == 0 && ds_d_hid.nonfinite == 0 &&
                                       ds_d_prop.nonfinite == 0 && ds_st_acc.nonfinite == 0);
            const bool ds_head_abs_ok = (ds_d_next.max_abs <= 1e-5);
            const bool ds_head_rel_ok = (ds_l1rel(ds_d_next) <= 1e-5);
            const bool ds_hid_abs_ok = (ds_d_hid.max_abs <= 1e-5);
            const bool ds_hid_rel_ok = (ds_l1rel(ds_d_hid) <= 1e-5);
            const bool ds_state_ok = (ds_size_ok && ds_byte_equal);
            const bool ds_scope_pass = (ds_prop_repro && ds_greedy_ok && ds_finite_ok && ds_head_abs_ok &&
                                        ds_head_rel_ok && ds_hid_abs_ok && ds_hid_rel_ok && ds_state_ok);
            std::printf(
                "{\"event\":\"qwen35_mtp_txn_draft_state_oracle_compare\",\"probe\":%lld,\"round\":%lld,"
                "\"case\":\"%s\",\"probe_kind\":\"draft_ctx_warm_round1\",\"pos\":%lld,\"pending\":%d,"
                "\"original_draft\":%d,\"injected\":%s,\"draft\":%d,\"t0\":%d,\"t1\":%d,\"accept\":%s,"
                "\"rollback\":%lld,\"next_token\":%d,\"next_pos\":%lld,\"act_propose_next\":%d,"
                "\"ref_propose_next\":%d,\"act_next\":%d,\"ref_next\":%d,"
                "\"provenance\":{\"pending_h\":\"round1_pre_deepcopy\",\"accepted_h\":\"target_verify_row0\","
                "\"next_h\":\"%s\"},"
                "\"gates\":{\"case_ok\":true,\"propose_repro\":%s,\"greedy_equal\":%s,\"finite\":%s,"
                "\"head_abs_1e5\":%s,\"head_rel_1e5\":%s,\"hid_abs_1e5\":%s,\"hid_rel_1e5\":%s,"
                "\"state_size_equal\":%s,\"state_byte_equal\":%s,\"scope_pass\":%s},"
                "\"logits_consumer\":{\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g,"
                "\"l1_rel\":%.9g},\"hidden_consumer\":{\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,\"l1\":%.9g,"
                "\"rms\":%.9g,\"l1_rel\":%.9g},\"propose_logits\":{\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,"
                "\"l1\":%.9g,\"rms\":%.9g,\"l1_rel\":%.9g},"
                "\"state\":{\"status\":\"%s\",\"byte_equal\":%s,\"tensors\":%zu,\"elems\":%lld,\"neq\":%lld,"
                "\"max_abs\":%.9g,\"l1\":%.9g,\"rms\":%.9g,\"first_diff\":%lld,\"diff_bytes\":%lld,"
                "\"pre_size\":%zu,\"act_size\":%zu,\"ref_size\":%zu,\"restore_ret\":%zu,\"note\":\"%s\"},"
                "\"seq_range\":{\"act_pos_min\":%lld,\"act_pos_max\":%lld,\"ref_pos_min\":%lld,"
                "\"ref_pos_max\":%lld},\"extra_vs_normal_decodes\":%lld,"
                "\"note\":\"draft consumption/replay/state contract only; ACT state not restored; diagnostic "
                "stop\"}\n",
                (long long) ds_probe_no, (long long) round_idx, draft_oracle_case, (long long) ds_pos0,
                (int) ds_pending0, (int) original_draft, injected_this_round ? "true" : "false", (int) draft_id,
                (int) t0, (int) t1, accepted ? "true" : "false", (long long) rollback, (int) ds_next_tok,
                (long long) ds_next_pos, (int) ds_act_prop_next, (int) ds_ref_propose, (int) ds_act_next,
                (int) ds_ref_next, draft_oracle_accept ? "target_verify_row1" : "target_verify_row0",
                ds_prop_repro ? "true" : "false", ds_greedy_ok ? "true" : "false",
                ds_finite_ok ? "true" : "false", ds_head_abs_ok ? "true" : "false",
                ds_head_rel_ok ? "true" : "false", ds_hid_abs_ok ? "true" : "false",
                ds_hid_rel_ok ? "true" : "false", ds_size_ok ? "true" : "false",
                ds_byte_equal ? "true" : "false", ds_scope_pass ? "true" : "false",
                (long long) ds_d_next.n, (long long) ds_d_next.neq, ds_d_next.max_abs, ds_d_next.l1,
                ds_rms(ds_d_next), ds_l1rel(ds_d_next), (long long) ds_d_hid.n, (long long) ds_d_hid.neq,
                ds_d_hid.max_abs, ds_d_hid.l1, ds_rms(ds_d_hid), ds_l1rel(ds_d_hid),
                (long long) ds_d_prop.n, (long long) ds_d_prop.neq, ds_d_prop.max_abs, ds_d_prop.l1,
                ds_rms(ds_d_prop), ds_l1rel(ds_d_prop), ds_st_status, ds_byte_equal ? "true" : "false",
                ds_tensors.size(), (long long) ds_st_acc.n, (long long) ds_st_acc.neq, ds_st_acc.max_abs,
                ds_st_acc.l1, ds_rms(ds_st_acc), (long long) ds_first, (long long) ds_ndiff, ds_pre_sz,
                ds_act_sz, ds_ref_sz, ds_setret, ds_note.c_str(), (long long) ds_act_pmin,
                (long long) ds_act_pmax, (long long) ds_ref_pmin, (long long) ds_ref_pmax,
                (long long) (draft_oracle_accept ? 3 : 2));
            std::fflush(stdout);
            if (ds_st_rc == 1) {
                for (const auto& t : ds_tensors) {
                    std::printf("{\"event\":\"qwen35_mtp_txn_draft_state_oracle_state_tensor\",\"probe\":%lld,"
                                "\"name\":\"%s\",\"dtype\":\"%s\",\"n\":%lld,\"neq\":%lld,\"max_abs\":%.9g,"
                                "\"l1\":%.9g,\"rms\":%.9g}\n",
                                (long long) ds_probe_no, t.name.c_str(), t.dtype, (long long) t.d.n,
                                (long long) t.d.neq, t.d.max_abs, t.d.l1, ds_rms(t.d));
                }
                std::fflush(stdout);
            }
            if (!ds_prop_repro)
                return fail("draft state oracle REF propose greedy does not reproduce the ACT draft");
            if (!ds_greedy_ok) return fail("draft state oracle ACT/REF consumer greedy mismatch");
            if (!ds_finite_ok) return fail("draft state oracle ACT/REF non-finite numeric difference");
            if (!ds_size_ok) return fail("draft state oracle ACT/REF serialized draft state size mismatch");
            if (!ds_byte_equal)
                return fail("draft state oracle ACT/REF serialized draft state not BYTE_EQUAL (scope not PASS)");
            if (!(ds_head_abs_ok && ds_head_rel_ok && ds_hid_abs_ok && ds_hid_rel_ok))
                return fail("draft state oracle ACT/REF head/hidden beyond the 1e-5 abs/rel gate");
            ++g_draft_oracle_probes_done;
            ds_completed = true;
            // Diagnostic prefix stop: the ACT state is NOT restored and no continued request / product speed
            // is claimed; a later request re-prefills explicitly.
            stop = "draft_state_oracle_probe";
            break;
        }
        last_round_accepted = accepted;   // task 273: mode-3 pattern input for the next round
        print_cost_line("round", rounds);
        ++round_idx;
        ++rounds;
    }

    // Task 317: a draft-state probe that never armed (round never reached) or never completed is an
    // explicit NO_GO, never a silent success - the diagnostic run had no evidence value.
    if (draft_state_oracle_on && !ds_completed) {
        const char* ds_reason = ds_armed ? "probe_not_completed" : "probe_round_not_reached";
        std::printf("{\"event\":\"qwen35_mtp_txn_draft_state_oracle_no_go\",\"round\":%lld,\"case\":\"%s\","
                    "\"reason\":\"%s\",\"armed\":%s,\"fail_closed\":true}\n",
                    (long long) round_idx, draft_oracle_case, ds_reason, ds_armed ? "true" : "false");
        std::fflush(stdout);
        return fail(std::string("draft state oracle NO_GO: ") + ds_reason);
    }

    // Task 320: a T3 verification probe that never armed (the warm round 1 was never reached, e.g. the seed
    // or a pending hit EOG first, the visible budget ended, or the round was already the context tail) or
    // never completed is an explicit NO_GO, never a silent success - the diagnostic run had no evidence
    // value.
    if (t3_verify_oracle_on && !t3_done) {
        const char* t3_reason = t3_armed ? "probe_not_completed" : "probe_round_not_reached";
        std::printf("{\"event\":\"qwen35_mtp_txn_t3_verify_oracle_no_go\",\"round\":%lld,\"reason\":\"%s\","
                    "\"armed\":%s,\"fail_closed\":true}\n",
                    (long long) round_idx, t3_reason, t3_armed ? "true" : "false");
        std::fflush(stdout);
        return fail(t3_restore_serial_control_on
                        ? std::string("t3 restore-serial control NO_GO: ") + t3_reason
                        : std::string("t3 verify oracle NO_GO: ") + t3_reason);
    }

    // The stop decision instant is final here; record it before the done event's own write.
    const double wall = t_s();
    if (timing != nullptr) {
        timing->ok = true;
        timing->engine_wall_s = wall;
        timing->rounds = rounds;
        timing->emitted = emitted;
        timing->eos_count = eos_count;
        timing->end_pos = pos;
        timing->eog_id = (long long) eog_token;
        timing->stop = stop;
    }
    // Task 268: the done event is written AFTER the stop decision, i.e. outside the finalized engine wall;
    // its write cost is therefore not attributed to any phase (the previous version counted it in emit
    // while the wall had already ended, which pushed the residual negative).
    std::printf("{\"event\":\"qwen35_mtp_txn_done\",\"rounds\":%lld,\"emitted\":%lld,\"eos\":%lld,"
                "\"stop\":\"%s\",\"end_pos\":%lld,\"max_new\":%lld,\"wall_s\":%.9g}\n",
                (long long) rounds, (long long) emitted, (long long) eos_count, stop, (long long) pos,
                (long long) max_new, wall);
    std::fflush(stdout);
    if (timing != nullptr && cost_split_on) {
        const double accounted_ms = ph[0] + ph[1] + ph[2] + ph[3] + ph[4] + ph[5] + ph[6];
        const double unattr_ms = wall * 1000.0 - accounted_ms;
        timing->ph_initdraftctx_ms = ph[0];
        timing->ph_draft_ms = ph[1];
        timing->ph_verify_ms = ph[2];
        timing->ph_hcopy_ms = ph[3];
        timing->ph_replay_ms = ph[4];
        timing->ph_rollback_ms = ph[5];
        timing->ph_emit_ms = ph[6];
        // Signed, no clamp (task 268): a negative residual honestly shows spans exceeding the wall - it is
        // never hidden by a floor.
        timing->ph_unattributed_ms = unattr_ms;
        timing->cost_sync_calls = cost_sync_calls;
        timing->cost_sync_wait_s = cost_sync_wait_s;
        print_cost_line("final", rounds);
    }
    if (emitted_out != nullptr) *emitted_out = std::move(emits);
    if (err != nullptr) err->clear();
    return 0;
}

}  // namespace strata::core
