// src/core/qwen35_bridge.cpp - the Qwen35 routed-expert boundary.  See the header.
//
// THE BOUNDARY IS THIN ON PURPOSE.  It owns only the graph-level pieces: the CUSTOM node for the model graph,
// the scheduler's external-producer callback, and the D2H/H2D of the router ids and the activation image.
// Everything about WHERE experts come from, which are resident, the Launch (GPU hits) -> CPU misses -> Combine
// order and the cache budget/admission is the SHARED EXPERT SERVICE's: `ExpertSource` (`expert_source.cpp`),
// `ExpertCache` and `expert_native_*` in `expert_source.cpp`.  This file never launches an expert kernel,
// never picks a slot and never fills one.
#include "strata/core/qwen35_bridge.hpp"
#include "strata/core/qwen35_prefill.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {
// Task 127 (diagnostic, default off; defined in qwen35_prefill.cu): the prompt path's optional d2h split.
// The gate accessor keeps the off path untouched: unset -> this file makes no further call; on -> the setter
// hands the current call's `pre` graph-prefix event handle to the prompt path.
bool qwen_prefill_d2h_split_on();
void qwen_prefill_d2h_split_set_pre(void* ev);
namespace {

/// The graph boundary's per-layer state: the scheduler events, the two streams, and the BOUNDED pinned staging
/// (router ids, the activation image, the miss rows).  No expert buffers, no cache - those are the service's.
struct LayerState {
    int layer = -1;
    bool boundary_init = false;
    ggml_backend_t backend = nullptr;
    ggml_backend_event_t pre = nullptr, post = nullptr;
    /// The overlap prototype's join: recorded by the routed seam on the backend stream so that it covers the
    /// shared-expert submission that sits between the pre marker and the routed node.
    ggml_backend_event_t join = nullptr;
    bool pre_fresh = false;   ///< the pre marker recorded `pre` for this layer this round; consumed once
    cudaStream_t work = nullptr, copy = nullptr;
    cudaEvent_t in_ids = nullptr, in_act = nullptr, miss_up = nullptr;
    cudaEvent_t small_t_fence = nullptr;   ///< P271: the per-row work-completion fence (gate-ON path only)
    int32_t* host_ids = nullptr;
    float* host_act = nullptr;
    float* host_miss = nullptr;
    std::vector<int32_t> local_ids;
    std::vector<uint8_t> is_hit;
};

struct State {
    bool initialized = false;
    bool prefill_gpu = false;   ///< the prompt layers run the shared service's GPU expert pipeline
    bool shared_overlap = false;   ///< STRATA_QWEN35_SHARED_OVERLAP=1: decode layers get the pre marker
    std::atomic<bool> seam_armed{false};
    Qwen35BridgeConfig cfg;
    int64_t N = 0, NE = 0, H = 0, FF = 0, K = 0;
    std::unique_ptr<kernels::cpu::ExpertPool> pool;
    std::unique_ptr<FileExpertSource> src;
    ExpertDispatch dispatch;
    std::vector<std::unique_ptr<LayerState>> layers;
    std::atomic<uint64_t> jobs{0}, quants{0}, seconds_us{0};
    uint64_t decode_layers = 0, prefill_layers = 0, prefill_entries = 0;
    uint64_t overlap_marker_layers = 0, overlap_join_layers = 0;
    uint64_t start_hits = 0, start_misses = 0;
    // P271 (default off): the small-T native prototype's own counters; deliberately separate from
    // decode_layers/prefill_layers so the prototype cannot masquerade as either accounting.
    uint64_t small_t_layers = 0, small_t_rows = 0;
    // Task 284 (MULTI gate only): the multirow window's own seam counters; same separation rationale.
    uint64_t small_t_multi_windows = 0, small_t_multi_rows = 0;
    // M272 diagnostic (default off, exact STRATA_QWEN35_SMALL_T_COMPARE=1; only reachable inside the small-T
    // gate's branch): the first-target-T2 native-vs-prompt comparison.  One private reference buffer (never
    // read by the graph), a per-layer done mask (each of the five fixed layers compares at most once; hard
    // cap five), and the pass/fail counters.
    float* small_t_compare_dev = nullptr;
    uint64_t small_t_compare_bytes = 0;
    uint64_t small_t_compare_mask = 0;   ///< bit per layer: this fixed layer's comparison already ran
    uint64_t small_t_compare_layers = 0, small_t_compare_pass = 0, small_t_compare_fail = 0;
    std::mutex routing_mutex;
    std::ofstream routing;
    bool routing_tried = false;
    // routing-detail metadata (default off; STRATA_QWEN35_ROUTING_DETAIL_LOG as a path; request-scoped append)
    std::ofstream routing_detail;
    bool routing_detail_tried = false;
    bool routing_detail_armed = false;
    uint64_t routing_detail_rows = 0;
    uint32_t routing_detail_request = 0;
    // step-7/layer-0 probe (default off, one capture per request; armed after finish_prompt): round 7 of the
    // post-prompt layer-0 T1 calls is the evaluation that produces output index 7.
    bool step_armed = false;
    uint64_t step_base = 0;   ///< S.decode_layers right after finish_prompt (the fixed case: 1)
    int step_round = 0;       ///< layer-0 T1 calls since arming (prompt-tail T1 is not counted: armed later)
    bool step_done = false;
    std::vector<uint8_t> step_a, step_b, step_w;
    std::vector<int32_t> step_ids;
    std::vector<uint8_t> step_hit;
    // Task 325 t3 MoE slice probe (default off; exact STRATA_QWEN35_MTP_T3_MOE_SLICE=1, armed by the
    // transaction ONLY in the exact T3 verify-oracle true-T3 warm round 1): bounded boundary captures at the
    // fixed physical layers 3/4 - REF T1 rows 0..2 (6 callbacks) and the one ACT T3 (2 callbacks), hard caps
    // 8 callbacks / 8 MiB total.  Task 327 prefix mode (default off; exact
    // STRATA_QWEN35_MTP_T3_MOE_SLICE_PREFIX=1, meaningful only with the 325 gate armed) moves the same
    // captures to physical layers 0..2: REF 9 + ACT 3 = 12 callbacks, 8 MiB unchanged.  Private host vectors
    // only; when the gate is off nothing here is touched.
    int t3_probe_phase = 0;       ///< 0 disarmed; 1 REF T1 (row 0..2); 2 ACT T3 (row -1)
    int t3_probe_row = -1;
    int t3_probe_calls = 0;       ///< completed captures (hard cap 8)
    uint32_t t3_probe_seen = 0;   ///< bit per (phase,row,layer) slot; a duplicate capture is an error
    uint64_t t3_probe_bytes = 0;  ///< total capture bytes counted at enqueue (hard cap 8 MiB)
    bool t3_probe_error = false;  ///< sticky fail-closed flag recorded by the phase setter
    bool t3_probe_pre = false;    ///< a pre-capture is in flight inside the current seam call
    int t3_probe_pre_phase = 0, t3_probe_pre_row = -1, t3_probe_pre_layer = -1;
    int64_t t3_probe_pre_T = 0;
    int64_t t3_probe_ne[3][4] = {}, t3_probe_nb[3][4] = {};   ///< x/ids/w ne,nb as consumed
    uint64_t t3_probe_sz_x = 0, t3_probe_sz_ids = 0, t3_probe_sz_w = 0;
    std::vector<uint8_t> t3_probe_x, t3_probe_ids, t3_probe_w, t3_probe_out;
};

State* g_state = nullptr;

int layer_from_name(const char* name) {
    const char* dash = std::strrchr(name, '-');
    return dash != nullptr ? std::atoi(dash + 1) : -1;
}

double now_seconds() {
    static const auto epoch = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
}

/// The decode shared-expert overlap prototype is opt-in by VALUE: only the exact string "1" enables it.  An
/// unset variable or any other value ("0" included) keeps the existing graph and event order.
bool shared_overlap_requested() {
    const char* v = std::getenv("STRATA_QWEN35_SHARED_OVERLAP");
    return v != nullptr && std::strcmp(v, "1") == 0;
}

/// P271 prototype (default off; exact STRATA_QWEN35_SMALL_T_NATIVE=1): a prompt-shaped routed MoE node of a
/// 2..5-token batch runs through the per-token native decode service instead of the prompt expert pipeline.
/// The gate is read once per process; unset or any other value keeps the default path byte-identical (no new
/// check, allocation, CUDA call, sync or timing anywhere below the cached-bool test).
bool small_t_native_requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_SMALL_T_NATIVE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

/// Task 284 prototype (default off; exact STRATA_QWEN35_SMALL_T_MULTI=1): a small-T batch's expert math runs as
/// ONE multirow window (`expert_native_decode_window`) instead of per token.  Queried only where it can change
/// the outcome: inside the SMALL_T_NATIVE gate's branch and inside the diag compare.  With it unset the per-row
/// fenced path, the T1 decode path and the prompt path stay byte-identical.
bool small_t_multi_requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_SMALL_T_MULTI");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

/// M272 diagnostic (default off; exact STRATA_QWEN35_SMALL_T_COMPARE=1).  It is queried ONLY inside the
/// small-T gate's branch, so it can never alter the default path or a SMALL_T_NATIVE=0 run: with the small-T
/// gate off, the flag is never read and no comparison can run.
bool small_t_compare_requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_SMALL_T_COMPARE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

/// The five fixed layers of the M272 comparison (hard cap 5; each compares at most once).
bool small_t_compare_layer_is_target(int layer) {
    return layer == 0 || layer == 14 || layer == 34 || layer == 38 || layer == 39;
}

/// Step-7/layer-0 probe (diagnostic, default off): exact value "1" enables it.  When unset, every probe branch
/// is a cached-bool test only - no allocation, no clock, no DMA, no sync and no file I/O on the default path.
bool step_probe_requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_STEP_PROBE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

/// FNV-1a 64 with the explicit seed below (the same custom seed as the route fingerprint); printed in the record
/// line so the byte count and the seed are part of the evidence.
constexpr uint64_t kStepProbeFnvSeed = 1469598103934665603ull;
uint64_t step_probe_fnv(uint64_t h, const void* p, size_t n) {
    constexpr uint64_t kPrime = 1099511628211ull;
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= kPrime; }
    return h;
}

/// Step probe phase 1 (called after the graph pre/work waits, BEFORE the original decode service): round
/// bookkeeping for layer-0 T1 calls and, at round 7 only, the bounded A (MoE input) / W (router weights) D2H on
/// the work stream.  The input is captured before the service runs (no alias with the output).  Shape/type are
/// validated against the real tensors; a mismatch or a CUDA error fails loudly through the original contract.
bool step_probe_begin(State& S, LayerState& L, const ggml_tensor* node, int layer, bool& active) {
    active = false;
    if (!step_probe_requested()) return true;
    if (!S.step_armed || S.step_done || layer != 0 || node->ne[1] != 1) return true;
    ++S.step_round;   // the prompt-tail T1 call happened before arming, so round 7 == output index 7
    if (S.step_round != 7) return true;
    // Guard: the bounded contiguous copies below require complete F32, contiguous single-token A/W and a
    // contiguous single-token F32 output.  The null case is checked FIRST and the rejection message never
    // dereferences a missing source; any other layout is rejected too (the expert computation, streams and
    // scratch are untouched - the probe just does not run).
    const bool shapes_ok =
        node->src[0] != nullptr && node->src[1] != nullptr && node->src[2] != nullptr && node->data != nullptr &&
        node->type == GGML_TYPE_F32 && node->ne[0] == S.H && ggml_nelements(node) == S.H &&
        ggml_is_contiguous(node) && node->src[0]->data != nullptr && node->src[0]->type == GGML_TYPE_F32 &&
        node->src[0]->ne[0] == S.H && ggml_nelements(node->src[0]) == S.H && ggml_is_contiguous(node->src[0]) &&
        node->src[2]->data != nullptr && node->src[2]->type == GGML_TYPE_F32 &&
        ggml_nelements(node->src[2]) == S.K && ggml_is_contiguous(node->src[2]);
    if (!shapes_ok) {
        std::fprintf(stderr, "strata qwen35: step probe: the layer-0 tensors are not the expected contiguous "
                             "F32 single-token layout; probe rejected\n");
        return false;
    }
    const size_t a_bytes = (size_t) S.H * sizeof(float);
    const size_t w_bytes = (size_t) S.K * sizeof(float);
    S.step_a.assign(a_bytes, 0);
    S.step_b.assign(a_bytes, 0);
    S.step_w.assign(w_bytes, 0);
    if (cudaMemcpyAsync(S.step_a.data(), node->src[0]->data, a_bytes, cudaMemcpyDeviceToHost, L.work) != cudaSuccess ||
        cudaMemcpyAsync(S.step_w.data(), node->src[2]->data, w_bytes, cudaMemcpyDeviceToHost, L.work) != cudaSuccess) {
        std::fprintf(stderr, "strata qwen35: step probe: the input/weight D2H could not be enqueued\n");
        return false;
    }
    active = true;
    return true;
}

/// Step probe phase 2 (after the original service returned successfully): B (MoE output) D2H on the same work
/// stream with ONE checked sync (before the bridge's post record), then the existing host_ids/is_hit copies and
/// the bounded raw files + record line.  Any CUDA/file failure fails loudly; nothing prints a hash on failure.
bool step_probe_end(State& S, LayerState& L, const ggml_tensor* node, bool active) {
    if (!active) return true;
    const size_t a_bytes = (size_t) S.H * sizeof(float);
    const size_t w_bytes = (size_t) S.K * sizeof(float);
    if (cudaMemcpyAsync(S.step_b.data(), node->data, a_bytes, cudaMemcpyDeviceToHost, L.work) != cudaSuccess ||
        cudaStreamSynchronize(L.work) != cudaSuccess) {
        std::fprintf(stderr, "strata qwen35: step probe: the output D2H/sync failed\n");
        return false;
    }
    // Guard: every captured float must be finite BEFORE anything is written or marked successful; a NaN/Inf is
    // a real diagnostic failure (no clipping, no tolerance), and nothing is trimmed into a passing result.
    int64_t fa = 0, fb = 0, fw = 0;
    for (int64_t i = 0; i < S.H; ++i) {
        float va, vb;
        std::memcpy(&va, S.step_a.data() + (size_t) i * sizeof(float), sizeof(va));
        std::memcpy(&vb, S.step_b.data() + (size_t) i * sizeof(float), sizeof(vb));
        if (std::isfinite(va)) ++fa;
        if (std::isfinite(vb)) ++fb;
    }
    for (int64_t i = 0; i < S.K; ++i) {
        float vw;
        std::memcpy(&vw, S.step_w.data() + (size_t) i * sizeof(float), sizeof(vw));
        if (std::isfinite(vw)) ++fw;
    }
    if (fa != S.H || fb != S.H || fw != S.K) {
        std::fprintf(stderr, "strata qwen35: step probe: non-finite values in the captured tensors (finite A "
                             "%lld/%lld B %lld/%lld W %lld/%lld); probe failed before any raw/success record\n",
                     (long long) fa, (long long) S.H, (long long) fb, (long long) S.H, (long long) fw,
                     (long long) S.K);
        return false;
    }
    S.step_ids.assign((size_t) S.K, 0);
    S.step_hit.assign((size_t) S.K, 0);
    for (int64_t k = 0; k < S.K; ++k) {
        S.step_ids[(size_t) k] = L.host_ids[k];
        S.step_hit[(size_t) k] = L.is_hit[(size_t) k];
    }
    const char* dir = std::getenv("STRATA_QWEN35_STEP_PROBE_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        std::fprintf(stderr, "strata qwen35: step probe: no output directory (STRATA_QWEN35_STEP_PROBE_DIR)\n");
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "strata qwen35: step probe: cannot create %s (%s)\n", dir, ec.message().c_str());
        return false;
    }
    const size_t ids_bytes = (size_t) S.K * sizeof(int32_t);
    struct Part { const char* name; const void* p; size_t n; };
    const Part parts[5] = {{"A.bin", S.step_a.data(), a_bytes}, {"B.bin", S.step_b.data(), a_bytes},
                           {"W.bin", S.step_w.data(), w_bytes}, {"ids.bin", S.step_ids.data(), ids_bytes},
                           {"hit.bin", S.step_hit.data(), (size_t) S.K}};
    for (const Part& pt : parts) {
        const std::string path = std::string(dir) + "/" + pt.name;
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write((const char*) pt.p, (std::streamsize) pt.n);
        f.flush();
        if (!f.good()) {
            std::fprintf(stderr, "strata qwen35: step probe: cannot write %s\n", path.c_str());
            return false;
        }
        f.close();
        if (f.fail()) {
            std::fprintf(stderr, "strata qwen35: step probe: close failed for %s\n", path.c_str());
            return false;
        }
    }
    std::fprintf(stderr, "strata qwen35: step probe (STRATA_QWEN35_STEP_PROBE=1): round %d index %d layer 0 base "
                         "%llu count %llu H %lld K %lld bytes A %zu B %zu W %zu ids %zu hit %zu finite A %lld B "
                         "%lld W %lld fnv_seed 0x%016llx A 0x%016llx B 0x%016llx W 0x%016llx\n",
                 S.step_round, S.step_round, (unsigned long long) S.step_base,
                 (unsigned long long) S.decode_layers, (long long) S.H, (long long) S.K, a_bytes, a_bytes,
                 w_bytes, ids_bytes, (size_t) S.K, (long long) fa, (long long) fb, (long long) fw,
                 (unsigned long long) kStepProbeFnvSeed,
                 (unsigned long long) step_probe_fnv(kStepProbeFnvSeed, S.step_a.data(), a_bytes),
                 (unsigned long long) step_probe_fnv(kStepProbeFnvSeed, S.step_b.data(), a_bytes),
                 (unsigned long long) step_probe_fnv(kStepProbeFnvSeed, S.step_w.data(), w_bytes));
    S.step_done = true;
    return true;
}

/// Task 325 (default off; exact STRATA_QWEN35_MTP_T3_MOE_SLICE=1): the bounded T3 MoE boundary probe gate.
/// Read once; unset or any other value keeps every branch below a cached-bool test - no allocation, copy,
/// sync, clock or file I/O.  The transaction arms the phases only in the exact T3 verify-oracle warm round 1.
bool t3_probe_requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_T3_MOE_SLICE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

/// Task 327 (default off; exact STRATA_QWEN35_MTP_T3_MOE_SLICE_PREFIX=1; meaningful only with the 325 slice
/// gate armed): the probe selects the first three physical layers (0..2) instead of 3/4 - REF T1 rows 0..2 x
/// 3 layers = 9 plus one ACT T3 x 3 layers = 3 callbacks (hard cap 12; 8 MiB total unchanged).  Read once;
/// unset keeps the original 325 selection byte-identical.
bool t3_probe_prefix_requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_MTP_T3_MOE_SLICE_PREFIX");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

/// The fixed physical layers of the t3 probe: layers 0..2 in the 327 prefix mode, else exactly 3 and 4.
bool t3_probe_layer_is_target(int layer) {
    return t3_probe_prefix_requested() ? (layer >= 0 && layer <= 2) : (layer == 3 || layer == 4);
}

/// The unique capture slot per (phase,row,layer).  Prefix mode (327): REF row 0..2 x layers 0..2 = bits 0..8;
/// ACT T3 x layers 0..2 = bits 9..11.  Original mode (325): REF row 0..2 x layers 3/4 = bits 0..5; ACT T3
/// (row -1) x layers 3/4 = bits 6..7.  -1 for anything else, so an unexpected phase/row/layer fails closed
/// instead of silently dropping or mislabelling a capture.
int t3_probe_slot(int phase, int row, int layer) {
    if (t3_probe_prefix_requested()) {
        if (layer < 0 || layer > 2) return -1;
        if (phase == 1 && row >= 0 && row <= 2) return row * 3 + layer;
        if (phase == 2 && row == -1) return 9 + layer;
        return -1;
    }
    const int l = layer - 3;
    if (l < 0 || l > 1) return -1;
    if (phase == 1 && row >= 0 && row <= 2) return row * 2 + l;
    if (phase == 2 && row == -1) return 6 + l;
    return -1;
}

/// The probe's owned export directory (ASCII relative path; the runner's cwd is the ROOT): 327 prefix mode
/// writes its own directory.  File names carry phase/refrow/layer and each slot captures at most once, so
/// nothing is ever overwritten across modes or runs.
const char* t3_probe_dir() {
    return t3_probe_prefix_requested() ? "results/phase2_mtp/moePrefix327/captures"
                                       : "results/phase2_mtp/moeSlice325/captures";
}

bool t3_probe_write_file(const std::string& path, const void* p, size_t n) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write((const char*) p, (std::streamsize) n);
    f.flush();
    if (!f.good()) return false;
    f.close();
    return !f.fail();
}

/// Task 325 phase 1 (called after the graph pre/work waits, BEFORE the original decode service): guard the
/// expected phase/row/layer/T and the REAL tensor layouts, then enqueue the bounded x/ids/w D2H on the SAME
/// work stream.  Two accepted layouts, both the ones the consumers actually address: REF T1 = contiguous
/// single-row tensors; ACT T3 = per-token rows with the actual nb strides (never an assumed packed layout).
/// Any mismatch, duplicate slot, ninth callback or >8 MiB refuses the seam (fail closed); the input snapshot
/// cannot alias the output because it is enqueued before the service runs.
bool t3_probe_begin(State& S, LayerState& L, const ggml_tensor* node, int layer, bool& active) {
    active = false;
    if (!t3_probe_requested()) return true;
    if (S.t3_probe_error) {
        std::fprintf(stderr, "strata qwen35: t3 probe: a phase-setter misuse was recorded; fail closed\n");
        return false;
    }
    if (S.t3_probe_phase == 0) return true;
    if (!t3_probe_layer_is_target(layer)) return true;
    const int phase = S.t3_probe_phase;
    const int row = S.t3_probe_row;
    const int slot = t3_probe_slot(phase, row, layer);
    if (slot < 0) {
        std::fprintf(stderr, "strata qwen35: t3 probe: unexpected phase %d row %d at layer %d; fail closed\n",
                     phase, row, layer);
        return false;
    }
    if (S.t3_probe_pre) {
        std::fprintf(stderr, "strata qwen35: t3 probe: a previous capture never completed; fail closed\n");
        return false;
    }
    if ((S.t3_probe_seen & (1u << slot)) != 0) {
        std::fprintf(stderr, "strata qwen35: t3 probe: duplicate capture (phase %d row %d layer %d); fail "
                             "closed\n", phase, row, layer);
        return false;
    }
    const int cap = t3_probe_prefix_requested() ? 12 : 8;
    if (S.t3_probe_calls >= cap) {
        std::fprintf(stderr, "strata qwen35: t3 probe: callback cap %d exceeded; fail closed\n", cap);
        return false;
    }
    const ggml_tensor* x = node->src[0];
    const ggml_tensor* ids = node->src[1];
    const ggml_tensor* w = node->src[2];
    const int64_t T = node->ne[1];
    size_t x_row = 0, id_row = 0, w_row = 0;
    if (phase == 1) {
        const bool shapes_ok = T == 1 && node->data != nullptr && node->type == GGML_TYPE_F32 &&
            node->ne[0] == S.H && ggml_nelements(node) == S.H && ggml_is_contiguous(node) &&
            x != nullptr && x->data != nullptr && x->type == GGML_TYPE_F32 && x->ne[0] == S.H &&
            ggml_nelements(x) == S.H && ggml_is_contiguous(x) &&
            ids != nullptr && ids->data != nullptr && ids->type == GGML_TYPE_I32 &&
            ggml_nelements(ids) == S.K && ggml_is_contiguous(ids) &&
            w != nullptr && w->data != nullptr && w->type == GGML_TYPE_F32 &&
            ggml_nelements(w) == S.K && ggml_is_contiguous(w);
        if (!shapes_ok) {
            std::fprintf(stderr, "strata qwen35: t3 probe: phase 1 layer %d is not the exact contiguous "
                                 "single-row F32/I32 layout (T %lld); refused\n", layer, (long long) T);
            return false;
        }
        x_row = (size_t) S.H * sizeof(float);
        id_row = (size_t) S.K * sizeof(int32_t);
        w_row = (size_t) S.K * sizeof(float);
    } else {
        const uint64_t xb = (uint64_t) S.H * sizeof(float);
        const uint64_t ib = (uint64_t) S.K * sizeof(int32_t);
        const uint64_t wb = (uint64_t) S.K * sizeof(float);
        const bool shapes_ok = T == 3 && node->data != nullptr && node->type == GGML_TYPE_F32 &&
            node->ne[0] == S.H && node->ne[1] == T && node->ne[2] == 1 && node->ne[3] == 1 &&
            node->nb[0] == sizeof(float) && node->nb[1] >= xb &&
            (uint64_t) (T - 1) * node->nb[1] + xb <= ggml_nbytes(node) &&
            x != nullptr && x->data != nullptr && x->type == GGML_TYPE_F32 && x->ne[0] == S.H &&
            x->ne[1] == T && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) && x->nb[1] >= xb &&
            (uint64_t) (T - 1) * x->nb[1] + xb <= ggml_nbytes(x) &&
            ids != nullptr && ids->data != nullptr && ids->type == GGML_TYPE_I32 && ids->ne[0] == S.K &&
            ids->ne[1] == T && ids->ne[2] == 1 && ids->ne[3] == 1 && ids->nb[0] == sizeof(int32_t) &&
            ids->nb[1] >= ib && (uint64_t) (T - 1) * ids->nb[1] + ib <= ggml_nbytes(ids) &&
            w != nullptr && w->data != nullptr && w->type == GGML_TYPE_F32 && w->ne[0] == 1 &&
            w->ne[1] == S.K && w->ne[2] == T && w->ne[3] == 1 && w->nb[0] == sizeof(float) &&
            w->nb[1] == sizeof(float) && w->nb[2] >= wb &&
            (uint64_t) (T - 1) * w->nb[2] + wb <= ggml_nbytes(w);
        if (!shapes_ok) {
            std::fprintf(stderr, "strata qwen35: t3 probe: phase 2 layer %d is not the exact per-token-row "
                                 "layout of the T3 batch (T %lld); refused\n", layer, (long long) T);
            return false;
        }
        x_row = (size_t) xb;
        id_row = (size_t) ib;
        w_row = (size_t) wb;
    }
    const size_t x_sz = (size_t) T * x_row;
    const size_t ids_sz = (size_t) T * id_row;
    const size_t w_sz = (size_t) T * w_row;
    const size_t out_sz = (size_t) T * x_row;
    const uint64_t capture_bytes = (uint64_t) x_sz + ids_sz + w_sz + out_sz;
    if (S.t3_probe_bytes + capture_bytes > 8ull * 1024 * 1024) {
        std::fprintf(stderr, "strata qwen35: t3 probe: 8 MiB capture cap exceeded (%llu + %llu); fail "
                             "closed\n", (unsigned long long) S.t3_probe_bytes,
                     (unsigned long long) capture_bytes);
        return false;
    }
    S.t3_probe_x.assign(x_sz, 0);
    S.t3_probe_ids.assign(ids_sz, 0);
    S.t3_probe_w.assign(w_sz, 0);
    S.t3_probe_out.assign(out_sz, 0);
    for (int64_t t = 0; t < T; ++t) {
        const uint8_t* sx = (const uint8_t*) x->data + (size_t) t * x->nb[1];
        const uint8_t* si = (const uint8_t*) ids->data + (size_t) t * ids->nb[1];
        const uint8_t* sw = (const uint8_t*) w->data + (size_t) t * w->nb[2];
        if (cudaMemcpyAsync(S.t3_probe_x.data() + (size_t) t * x_row, sx, x_row, cudaMemcpyDeviceToHost,
                            L.work) != cudaSuccess ||
            cudaMemcpyAsync(S.t3_probe_ids.data() + (size_t) t * id_row, si, id_row, cudaMemcpyDeviceToHost,
                            L.work) != cudaSuccess ||
            cudaMemcpyAsync(S.t3_probe_w.data() + (size_t) t * w_row, sw, w_row, cudaMemcpyDeviceToHost,
                            L.work) != cudaSuccess) {
            std::fprintf(stderr, "strata qwen35: t3 probe: the input D2H could not be enqueued\n");
            return false;
        }
    }
    S.t3_probe_pre = true;
    S.t3_probe_pre_phase = phase;
    S.t3_probe_pre_row = row;
    S.t3_probe_pre_layer = layer;
    S.t3_probe_pre_T = T;
    const ggml_tensor* meta[3] = { x, ids, w };
    for (int m = 0; m < 3; ++m) {
        for (int d = 0; d < 4; ++d) {
            S.t3_probe_ne[m][d] = meta[m]->ne[d];
            S.t3_probe_nb[m][d] = (int64_t) meta[m]->nb[d];
        }
    }
    S.t3_probe_sz_x = x_sz;
    S.t3_probe_sz_ids = ids_sz;
    S.t3_probe_sz_w = w_sz;
    S.t3_probe_seen |= (1u << slot);
    S.t3_probe_bytes += capture_bytes;
    active = true;
    return true;
}

/// Task 325 phase 2 (after the original service returned and before the original post record/join): the
/// output D2H on the SAME work stream, ONE checked sync, finite/legality guards, then the raw files, the
/// meta file and the stderr record with the actual byte counts and hashes (no selective values).  Any guard
/// failure refuses the seam; nothing is written for an incomplete capture.
bool t3_probe_end(State& S, LayerState& L, const ggml_tensor* node, bool active) {
    if (!active) return true;
    const int phase = S.t3_probe_pre_phase;
    const int row = S.t3_probe_pre_row;
    const int layer = S.t3_probe_pre_layer;
    const int64_t T = S.t3_probe_pre_T;
    const size_t x_row = S.t3_probe_sz_x / (size_t) T;
    bool enq_ok = true;
    for (int64_t t = 0; t < T && enq_ok; ++t) {
        const uint8_t* so = (const uint8_t*) node->data + (size_t) t * node->nb[1];
        enq_ok = cudaMemcpyAsync(S.t3_probe_out.data() + (size_t) t * x_row, so, x_row,
                                 cudaMemcpyDeviceToHost, L.work) == cudaSuccess;
    }
    if (!enq_ok || cudaStreamSynchronize(L.work) != cudaSuccess) {
        std::fprintf(stderr, "strata qwen35: t3 probe: phase %d row %d layer %d output D2H/sync failed\n",
                     phase, row, layer);
        return false;
    }
    const size_t nf_x = S.t3_probe_x.size() / sizeof(float);
    const size_t nf_w = S.t3_probe_w.size() / sizeof(float);
    const size_t nf_o = S.t3_probe_out.size() / sizeof(float);
    int64_t fx = 0, fw = 0, fo = 0, legal = 0;
    for (size_t i = 0; i < nf_x; ++i) {
        float v;
        std::memcpy(&v, S.t3_probe_x.data() + i * sizeof(float), sizeof(v));
        if (std::isfinite(v)) ++fx;
    }
    for (size_t i = 0; i < nf_w; ++i) {
        float v;
        std::memcpy(&v, S.t3_probe_w.data() + i * sizeof(float), sizeof(v));
        if (std::isfinite(v)) ++fw;
    }
    for (size_t i = 0; i < nf_o; ++i) {
        float v;
        std::memcpy(&v, S.t3_probe_out.data() + i * sizeof(float), sizeof(v));
        if (std::isfinite(v)) ++fo;
    }
    const size_t n_ids = S.t3_probe_ids.size() / sizeof(int32_t);
    for (size_t i = 0; i < n_ids; ++i) {
        int32_t v;
        std::memcpy(&v, S.t3_probe_ids.data() + i * sizeof(int32_t), sizeof(v));
        if (v >= 0 && (int64_t) v < S.NE) ++legal;
    }
    if (fx != (int64_t) nf_x || fw != (int64_t) nf_w || fo != (int64_t) nf_o || legal != (int64_t) n_ids) {
        std::fprintf(stderr, "strata qwen35: t3 probe: guard failed (finite x %lld/%zu w %lld/%zu out "
                             "%lld/%zu legal ids %lld/%zu); refused\n", (long long) fx, nf_x, (long long) fw,
                     nf_w, (long long) fo, nf_o, (long long) legal, n_ids);
        return false;
    }
    const std::string dir = t3_probe_dir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "strata qwen35: t3 probe: cannot create %s (%s)\n", dir.c_str(),
                     ec.message().c_str());
        return false;
    }
    char stem[96];
    std::snprintf(stem, sizeof(stem), "t325-p%d-r%d-L%d", phase, row, layer);
    struct T3Part { const char* suf; const void* p; size_t n; };
    const T3Part parts[4] = { { "-x.bin", S.t3_probe_x.data(), S.t3_probe_x.size() },
                              { "-ids.bin", S.t3_probe_ids.data(), S.t3_probe_ids.size() },
                              { "-w.bin", S.t3_probe_w.data(), S.t3_probe_w.size() },
                              { "-out.bin", S.t3_probe_out.data(), S.t3_probe_out.size() } };
    uint64_t fnv[4] = { 0, 0, 0, 0 };
    for (int i = 0; i < 4; ++i) {
        fnv[i] = step_probe_fnv(kStepProbeFnvSeed, parts[i].p, parts[i].n);
        if (!t3_probe_write_file(dir + "/" + stem + parts[i].suf, parts[i].p, parts[i].n)) {
            std::fprintf(stderr, "strata qwen35: t3 probe: cannot write %s/%s%s\n", dir.c_str(), stem,
                         parts[i].suf);
            return false;
        }
    }
    {
        std::ofstream m(dir + "/" + stem + "-meta.txt", std::ios::binary | std::ios::trunc);
        m << (t3_probe_prefix_requested() ? "task 327 t3 moe prefix capture\n"
                                          : "task 325 t3 moe slice capture\n")
          << "phase " << phase << "\nrefrow " << row << "\nlayer " << layer << "\nT " << T;
        const char* names[3] = { "x", "ids", "w" };
        for (int tns = 0; tns < 3; ++tns) {
            m << "\n" << names[tns] << "_ne";
            for (int d = 0; d < 4; ++d) m << " " << S.t3_probe_ne[tns][d];
            m << "\n" << names[tns] << "_nb";
            for (int d = 0; d < 4; ++d) m << " " << S.t3_probe_nb[tns][d];
        }
        m << "\nout_ne";
        for (int d = 0; d < 4; ++d) m << " " << node->ne[d];
        m << "\nout_nb";
        for (int d = 0; d < 4; ++d) m << " " << (int64_t) node->nb[d];
        m << "\nbytes x " << S.t3_probe_x.size() << " ids " << S.t3_probe_ids.size() << " w "
          << S.t3_probe_w.size() << " out " << S.t3_probe_out.size();
        m << "\nfinite_f32 x " << fx << " w " << fw << " out " << fo;
        m << "\nlegal_ids " << legal << "\nfnv_seed 0x" << std::hex << kStepProbeFnvSeed << "\nfnv64 x 0x"
          << fnv[0] << " ids 0x" << fnv[1] << " w 0x" << fnv[2] << " out 0x" << fnv[3] << "\n";
        m.flush();
        if (!m.good()) {
            std::fprintf(stderr, "strata qwen35: t3 probe: cannot write %s/%s-meta.txt\n", dir.c_str(), stem);
            return false;
        }
    }
    std::fprintf(stderr, "strata qwen35: t3 probe capture phase %d refrow %d layer %d T %lld bytes x %zu "
                         "ids %zu w %zu out %zu total %llu finite x %lld w %lld out %lld legal_ids %lld "
                         "fnv_seed 0x%016llx x 0x%016llx ids 0x%016llx w 0x%016llx out 0x%016llx dir %s\n",
                 phase, row, layer, (long long) T, S.t3_probe_x.size(), S.t3_probe_ids.size(),
                 S.t3_probe_w.size(), S.t3_probe_out.size(), (unsigned long long) S.t3_probe_bytes,
                 (long long) fx, (long long) fw, (long long) fo, (long long) legal,
                 (unsigned long long) kStepProbeFnvSeed, (unsigned long long) fnv[0],
                 (unsigned long long) fnv[1], (unsigned long long) fnv[2], (unsigned long long) fnv[3],
                 dir.c_str());
    std::fflush(stderr);
    ++S.t3_probe_calls;
    S.t3_probe_pre = false;
    return true;
}

void append_routing(const std::string& payload) {
    State& S = *g_state;
    if (payload.empty()) return;
    std::lock_guard<std::mutex> guard(S.routing_mutex);
    if (!S.routing_tried) {
        S.routing_tried = true;
        S.routing.open(S.cfg.routing_log, std::ios::app | std::ios::binary);
    }
    if (S.routing.is_open()) S.routing << payload;
}

/// The routing-detail diagnostic (default off): STRATA_QWEN35_ROUTING_DETAIL_LOG is a file path and only a
/// non-empty value enables it.  Read once (cached); unset -> every branch below is a cached-pointer test.
const char* routing_detail_path() {
    static const char* v = [] {
        const char* e = std::getenv("STRATA_QWEN35_ROUTING_DETAIL_LOG");
        return (e != nullptr && e[0] != '\0') ? e : nullptr;
    }();
    return v;
}

uint64_t routing_detail_fnv(uint64_t h, const void* p, size_t n) {
    constexpr uint64_t kPrime = 1099511628211ull;
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= kPrime; }
    return h;
}

/// Opens the detail file on first use (APPEND: an existing file is never truncated), then writes one request
/// marker and the bounded host residency summary.  False with an explicit message on any open/write/flush
/// failure - the diagnostic was requested, so an incomplete file must not look like a pass.  This runs after
/// the prompt-tail T1 call is already counted and the prompt loan was returned, so the marker is the
/// post-prompt baseline: decode_base is S.decode_layers there (1 in the fixed case), hits_base/misses_base are
/// the per-expert counters there, and only LATER layers are logged; the row sums therefore correspond to the
/// stats DELTAS (rows = (decode_layers - decode_base) * K, hit_sum = decode_hits - hits_base).  map_fnv is a
/// standard FNV-1a64 (offset basis 14695981039346656037, prime 1099511628211) over the resident
/// (layer, expert, slot) int32 tuples in ascending (layer, expert) order, hashed as their native little-endian
/// bytes; the resident count and decode_base are separate marker fields and are NOT part of the digest.  It is
/// read-only host state (no cache/GPU mutation) and only a summary: hashes can collide in principle and the
/// slot payload bytes are not observed, so equal digests do NOT prove equal cache data.  The request ordinal is
/// State-local (it starts at 1 in every process): a path reused across processes is still segmented by its
/// markers, but the N values are not globally increasing.
bool routing_detail_begin(State& S) {
    const char* path = routing_detail_path();
    if (path == nullptr) return true;
    if (!S.routing_detail_tried) {
        S.routing_detail_tried = true;
        S.routing_detail.open(path, std::ios::app | std::ios::binary);
        if (!S.routing_detail.is_open()) {
            std::fprintf(stderr, "strata qwen35: routing detail log cannot open %s\n", path);
            return false;
        }
    }
    ++S.routing_detail_request;
    S.routing_detail_rows = 0;
    S.routing_detail_armed = true;
    uint64_t h = 14695981039346656037ull;
    int64_t resident = 0;
    for (int64_t l = 0; l < S.N; ++l) {
        for (int64_t e = 0; e < S.NE; ++e) {
            const int32_t slot = S.dispatch.cache != nullptr ? S.dispatch.cache->slot_of(l, e) : (int32_t) -1;
            if (slot < 0) continue;
            ++resident;
            const int32_t v[3] = {(int32_t) l, (int32_t) e, slot};
            h = routing_detail_fnv(h, v, sizeof(v));
        }
    }
    S.routing_detail << "#request " << S.routing_detail_request << " decode_base " << S.decode_layers
                     << " hits_base " << S.dispatch.native_hits << " misses_base " << S.dispatch.native_misses
                     << " resident " << resident << " map_fnv 0x" << std::hex << h << std::dec
                     << " layers " << S.N << " experts " << S.NE
                     << " schema before_count,layer,rank,id,hit,slot_after\n";
    S.routing_detail.flush();
    if (!S.routing_detail.good()) {
        std::fprintf(stderr, "strata qwen35: routing detail log write/flush failed (request marker)\n");
        return false;
    }
    return true;
}

LayerState& layer_state(State& S, int layer) {
    if (S.layers[(size_t) layer] == nullptr) {
        S.layers[(size_t) layer] = std::make_unique<LayerState>();
        S.layers[(size_t) layer]->layer = layer;
    }
    return *S.layers[(size_t) layer];
}

/// Releases whatever this layer's boundary created - a PARTIAL failure included - and nulls the handles, so a
/// retry starts clean and shutdown never leaks the pieces a mid-way failure left behind.
void boundary_free(LayerState& L) {
    if (L.work) cudaStreamSynchronize(L.work);
    if (L.copy) cudaStreamSynchronize(L.copy);
    if (L.host_ids) { cudaFreeHost(L.host_ids); L.host_ids = nullptr; }
    L.host_act = nullptr;   // views into the pinned block above
    L.host_miss = nullptr;
    if (L.work) { cudaStreamDestroy(L.work); L.work = nullptr; }
    if (L.copy) { cudaStreamDestroy(L.copy); L.copy = nullptr; }
    if (L.in_ids) { cudaEventDestroy(L.in_ids); L.in_ids = nullptr; }
    if (L.in_act) { cudaEventDestroy(L.in_act); L.in_act = nullptr; }
    if (L.miss_up) { cudaEventDestroy(L.miss_up); L.miss_up = nullptr; }
    if (L.small_t_fence) { cudaEventDestroy(L.small_t_fence); L.small_t_fence = nullptr; }
    if (L.pre) { ggml_backend_event_free(L.pre); L.pre = nullptr; }
    if (L.post) { ggml_backend_event_free(L.post); L.post = nullptr; }
    if (L.join) { ggml_backend_event_free(L.join); L.join = nullptr; }
    L.pre_fresh = false;
    L.local_ids.clear();
    L.is_hit.clear();
    L.boundary_init = false;
}

/// The graph boundary's own buffers: two non-blocking streams, the scheduler events, and one bounded pinned
/// staging block (k ids | n_embd activation | k x n_embd miss rows).
bool boundary_init(LayerState& L, ggml_backend_t backend, int64_t K, int64_t H) {
    boundary_free(L);   // a retry starts clean even after a partial failure
    L.backend = backend;
    bool ok = true;
    // The overlap prototype's join is allocated ONLY when the prototype is latched: the default path
    // allocates exactly the events it did before, and the join consumer cannot run without the latch.
    const bool want_join = g_state != nullptr && g_state->shared_overlap;
    L.pre = ggml_backend_event_new(ggml_backend_get_device(backend));
    L.post = ggml_backend_event_new(ggml_backend_get_device(backend));
    if (want_join) L.join = ggml_backend_event_new(ggml_backend_get_device(backend));
    if (L.pre == nullptr || L.post == nullptr || (want_join && L.join == nullptr)) ok = false;
    if (ok && cudaStreamCreateWithFlags(&L.work, cudaStreamNonBlocking) != cudaSuccess) ok = false;
    if (ok && cudaStreamCreateWithFlags(&L.copy, cudaStreamNonBlocking) != cudaSuccess) ok = false;
    for (cudaEvent_t* e : {&L.in_ids, &L.in_act, &L.miss_up}) {
        if (!ok) break;
        if (cudaEventCreateWithFlags(e, cudaEventDisableTiming) != cudaSuccess) ok = false;
    }
    // Task 284: with the multirow gate latched the SAME pinned block is sized for the largest small-T window
    // (T <= 5) so one window's ids/activation/miss rows need no further allocation; the default path keeps
    // exactly the previous (K, 1, K) footprint.
    const int64_t rows = small_t_multi_requested() ? 5 : 1;
    const size_t act = (size_t) H * sizeof(float);
    const size_t off_act = ((size_t) (rows * K) * sizeof(int32_t) + 63) & ~size_t(63);
    const size_t off_miss = (off_act + (size_t) rows * act + 63) & ~size_t(63);
    const size_t bytes = off_miss + (size_t) rows * (size_t) K * act;
    uint8_t* pinned = nullptr;
    if (ok && cudaHostAlloc((void**) &pinned, bytes, cudaHostAllocDefault) != cudaSuccess) ok = false;
    if (!ok) {
        if (pinned != nullptr) cudaFreeHost(pinned);
        boundary_free(L);
        return false;
    }
    L.host_ids = (int32_t*) pinned;
    L.host_act = (float*) (pinned + off_act);
    L.host_miss = (float*) (pinned + off_miss);
    L.is_hit.assign((size_t) (rows * K), 0);
    L.boundary_init = true;
    return true;
}

/// The CPU fallback if the external callback did not run: a silent wrong answer is worse.
void seam_cpu_abort(ggml_tensor*, int, int, void*) {
    std::fprintf(stderr, "strata qwen35: a decode MoE node reached the CPU path - the external scheduler seam "
                         "was not installed\n");
    std::fflush(stderr);
    std::abort();
}

void prefill_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    LayerState* L = (LayerState*) userdata;
    State& S = *g_state;
    const bool trace = std::getenv("STRATA_QWEN35_TRACE") != nullptr;
    if (trace) std::fprintf(stderr, "qwen35 trace: prefill_compute layer %d ith %d nth %d\n", L->layer, ith, nth);
    if (ith != 0 || nth != 1 || L == nullptr) {
        std::fprintf(stderr, "strata qwen35: the prefill expert node must run as one task\n");
        std::abort();
    }
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* ids = dst->src[1];
    const ggml_tensor* w = dst->src[2];
    const int64_t n_tok = x->ne[1];
    // The graph's tensors are strided views (selected_experts keeps nb[1] = n_expert*4 from argsort_top_k),
    // so gather a contiguous host image of the three inputs first.
    static thread_local std::vector<int32_t> idv;
    static thread_local std::vector<float> xv, wv;
    idv.resize((size_t) (n_tok * S.K));
    xv.resize((size_t) (n_tok * S.H));
    wv.resize((size_t) (n_tok * S.K));
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t j = 0; j < S.K; ++j) {
            idv[(size_t) (t * S.K + j)] =
                *(const int32_t*) ((const uint8_t*) ids->data + (size_t) t * ids->nb[1] + (size_t) j * ids->nb[0]);
            wv[(size_t) (t * S.K + j)] =
                *(const float*) ((const uint8_t*) w->data + (size_t) t * w->nb[2] + (size_t) j * w->nb[1]);
        }
        std::memcpy(xv.data() + (size_t) t * S.H, (const uint8_t*) x->data + (size_t) t * x->nb[1],
                    (size_t) S.H * sizeof(float));
    }
    const int32_t* idf = idv.data();
    const float* xf = xv.data();
    const float* wf = wv.data();
    float* out = (float*) dst->data;
    const size_t need = (size_t) n_tok * (size_t) S.K * (size_t) S.H;
    static thread_local std::vector<float> parts;
    if (parts.size() < need) parts.resize(need);
    const double t0 = now_seconds();
    if (trace) {
        std::fprintf(stderr, "qwen35 trace: layer %d dispatch begin n_tok %lld\n", L->layer, (long long) n_tok);
        std::fprintf(stderr, "qwen35 trace: ids ne [%lld %lld] nb [%lld %lld] first:", (long long) ids->ne[0],
                     (long long) ids->ne[1], (long long) ids->nb[0], (long long) ids->nb[1]);
        for (int i = 0; i < 16 && i < (int) (n_tok * S.K); ++i) std::fprintf(stderr, " %d", idf[i]);
        std::fprintf(stderr, " | x nb0 nb1 %lld %lld\n", (long long) x->nb[0], (long long) x->nb[1]);
    }
    S.dispatch.layers = L->layer;
    expert_native_qwen_dispatch(S.dispatch, xf, idf, n_tok, S.K, nullptr, parts.data());
    if (trace) std::fprintf(stderr, "qwen35 trace: layer %d dispatch end (failed %d)\n", L->layer, S.dispatch.failed ? 1 : 0);
    if (S.dispatch.failed) {
        std::fprintf(stderr, "strata qwen35: prefill expert dispatch failed at layer %lld: %s\n",
                     (long long) L->layer, S.dispatch.fail != nullptr ? S.dispatch.fail : "?");
        std::fflush(stderr);
        std::abort();
    }
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t r = 0; r < S.H; ++r) {
            float acc = 0.0f;
            for (int64_t j = 0; j < S.K; ++j)
                acc += wf[(size_t) (t * S.K + j)] * parts[(size_t) ((t * S.K + j) * S.H + r)];
            out[(size_t) (t * S.H + r)] = acc;
        }
    ++S.prefill_layers;
    S.prefill_entries += (uint64_t) (n_tok * S.K);
    S.jobs.fetch_add((uint64_t) (n_tok * S.K), std::memory_order_relaxed);
    S.quants.fetch_add((uint64_t) n_tok, std::memory_order_relaxed);
    S.seconds_us.fetch_add((uint64_t) ((now_seconds() - t0) * 1e6), std::memory_order_relaxed);
}

/// The overlap prototype's decode pre marker.  It computes nothing: it records `pre` on the backend stream at
/// a graph position after the router/activation producers and before the shared expert, so the routed seam
/// that follows can let the CPU miss window overlap the shared expert while still ordering its own hits and
/// combine behind the shared expert's arena reads (through `join`).
bool seam_marker_exec(ggml_backend_t backend, ggml_tensor* node) {
    State& S = *g_state;
    if (!S.shared_overlap) return false;
    const bool trace = std::getenv("STRATA_QWEN35_TRACE") != nullptr;
    const int layer = layer_from_name(node->name);
    if (trace) std::fprintf(stderr, "qwen35 trace: seam_marker layer %d begin\n", layer);
    if (layer < 0 || layer >= (int) S.layers.size() || S.layers[(size_t) layer] == nullptr) return false;
    LayerState& L = *S.layers[(size_t) layer];
    if (!L.boundary_init && !boundary_init(L, backend, S.K, S.H)) {
        std::fprintf(stderr, "strata qwen35: layer %d's graph boundary could not initialise (pre marker)\n", layer);
        return false;
    }
    ggml_backend_event_record(L.pre, backend);
    L.pre_fresh = true;
    ++S.overlap_marker_layers;
    if (trace) std::fprintf(stderr, "qwen35 trace: seam_marker layer %d pre recorded\n", layer);
    return true;
}

// ---- Task 133: seam-outside host gap vs backend-stream device span (exact STRATA_QWEN35_SEAM_GAP=1, off by
// default).  For every seam callback: host_gap = host wall from the previous callback's return to this one's
// entry; dev_span = the backend-stream event window recorded at those same two points (E0 at exit, E1 at
// entry).  host_only = host_gap - dev_span.  Flushed from qwen35_bridge_stats, i.e. after the request's
// prefill/TTFT windows.  Unset: zero calls, zero output, the default path untouched.
bool seam_gap_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_SEAM_GAP");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
struct SeamGapState {
    struct Pair {
        ggml_backend_event_t e0 = nullptr, e1 = nullptr;
        int layer = -1;
        bool decode = false;
        double host_gap_ms = 0.0;
    };
    std::vector<Pair> pairs;
    ggml_backend_event_t pending_e0 = nullptr;  ///< recorded at the last seam exit; paired at the next entry
    double t_arm = -1.0, t_prev_exit = -1.0, t_last_exit = -1.0, pre0_ms = -1.0;
    int64_t create_fail = 0, read_fail = 0, dropped = 0;
    bool flushed = false;
};
SeamGapState g_seam_gap;

// Task 149 (diagnostic; STRATA_QWEN35_SEAM_GAP=1 only, default path untouched): cudaEventElapsedTime
// refuses the backend's own events (cudaEventDisableTiming - the 133 read_fail), so the seam-gap gate
// allocates its own timing-capable event and wraps it in the same ggml_backend_event struct that
// ggml_backend_event_record/free already understand.  Never called when the gate is off.
ggml_backend_event_t seam_timing_event_new(ggml_backend_t backend) {
    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDefault) != cudaSuccess) return nullptr;
    return new ggml_backend_event{ggml_backend_get_device(backend), (void*) ev};
}

void seam_gap_enter(ggml_backend_t backend, int layer, bool decode, double now) {
    SeamGapState& G = g_seam_gap;
    if (G.t_prev_exit < 0.0) {
        // The request's first seam: nothing precedes it but the arm point (the request-start account).
        if (G.t_arm >= 0.0 && G.pre0_ms < 0.0) G.pre0_ms = (now - G.t_arm) * 1000.0;
        return;
    }
    if (G.pending_e0 == nullptr) { ++G.dropped; G.t_prev_exit = now; return; }
    ggml_backend_event_t e1 = seam_timing_event_new(backend);
    if (e1 == nullptr) {
        ++G.create_fail;
        ++G.dropped;
        ggml_backend_event_free(G.pending_e0);
        G.pending_e0 = nullptr;
        G.t_prev_exit = now;
        return;
    }
    ggml_backend_event_record(e1, backend);
    G.pairs.push_back({G.pending_e0, e1, layer, decode, (now - G.t_prev_exit) * 1000.0});
    G.pending_e0 = nullptr;
}

void seam_gap_exit(ggml_backend_t backend, double now) {
    SeamGapState& G = g_seam_gap;
    if (G.pending_e0 == nullptr) {
        ggml_backend_event_t e0 = seam_timing_event_new(backend);
        if (e0 == nullptr) { ++G.create_fail; }
        else { ggml_backend_event_record(e0, backend); G.pending_e0 = e0; }
    }
    G.t_prev_exit = now;
    G.t_last_exit = now;
}

void seam_gap_flush() {
    SeamGapState& G = g_seam_gap;
    G.flushed = true;
    double host_sum = 0.0, dev_sum = 0.0, host_min = 0.0, host_max = 0.0, dev_min = 0.0, dev_max = 0.0;
    int ok = 0, prefill_windows = 0, decode_windows = 0;
    double decode_host = -1.0, decode_dev = -1.0;
    std::string per;
    char buf[160];
    for (const auto& p : G.pairs) {
        double dev = -1.0;
        if (p.e0 != nullptr && p.e1 != nullptr) {
            cudaError_t q = cudaEventQuery((cudaEvent_t) p.e1->context);
            if (q == cudaErrorNotReady) q = cudaEventSynchronize((cudaEvent_t) p.e1->context);   // request tail only
            float ms = -1.0f;
            if (q == cudaSuccess &&
                cudaEventElapsedTime(&ms, (cudaEvent_t) p.e0->context, (cudaEvent_t) p.e1->context) == cudaSuccess) {
                dev = (double) ms;
            } else {
                ++G.read_fail;
            }
        } else {
            ++G.read_fail;
        }
        if (dev >= 0.0) {
            ++ok;
            dev_sum += dev;
            host_sum += p.host_gap_ms;
            if (ok == 1) { host_min = host_max = p.host_gap_ms; dev_min = dev_max = dev; }
            else {
                if (p.host_gap_ms < host_min) host_min = p.host_gap_ms;
                if (p.host_gap_ms > host_max) host_max = p.host_gap_ms;
                if (dev < dev_min) dev_min = dev;
                if (dev > dev_max) dev_max = dev;
            }
            if (p.decode) { ++decode_windows; decode_host = p.host_gap_ms; decode_dev = dev; }
            else ++prefill_windows;
        }
        std::snprintf(buf, sizeof buf, "%d%s:%.2f:%.2f ", p.layer, p.decode ? "d" : "", p.host_gap_ms, dev);
        per += buf;
    }
    const double tail = G.t_last_exit >= 0.0 ? (now_seconds() - G.t_last_exit) * 1000.0 : -1.0;
    std::fprintf(stderr,
                 "strata qwen35: seam gap (STRATA_QWEN35_SEAM_GAP=1): windows %d pairs_ok %d create_fail %lld "
                 "read_fail %lld dropped %lld | host_gap_ms %.2f dev_span_ms %.2f host_only_ms %.2f | "
                 "per_window host avg %.3f max %.3f min %.3f | dev avg %.3f max %.3f min %.3f | "
                 "prefill_windows %d decode_windows %d decode host_gap_ms %.3f dev_span_ms %.3f | "
                 "pre0_ms %.3f tail_ms %.3f\n",
                 (int) G.pairs.size(), ok, (long long) G.create_fail, (long long) G.read_fail,
                 (long long) G.dropped, host_sum, dev_sum, host_sum - dev_sum,
                 ok > 0 ? host_sum / ok : 0.0, host_max, host_min,
                 ok > 0 ? dev_sum / ok : 0.0, dev_max, dev_min,
                 prefill_windows, decode_windows, decode_host, decode_dev, G.pre0_ms, tail);
    std::fprintf(stderr, "strata qwen35: seam gap per-window (enter_layer[d=decode]:host_gap_ms:dev_span_ms): %s\n",
                 per.c_str());
    for (auto& p : G.pairs) {
        if (p.e0 != nullptr) ggml_backend_event_free(p.e0);
        if (p.e1 != nullptr) ggml_backend_event_free(p.e1);
    }
    G.pairs.clear();
    if (G.pending_e0 != nullptr) { ggml_backend_event_free(G.pending_e0); G.pending_e0 = nullptr; }
}

// ---- Task 134 (exact STRATA_QWEN35_A_SPLIT=1, default off): the A-window split ---------------------------
// The runner marks the request start and the end of its batch fill; llama's process_ubatch marks its entry,
// the end of the graph build, the end of the sched allocation and a graph-reuse (see the patch's
// llama-context.cpp); the bridge records the first seam callback entry of each round here.  mark(0) starts a
// new round, so a second, identical-shape prompt pass in the same process gets its own phases.  Unset: no
// calls, no output, the default path untouched.
bool a_split_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_A_SPLIT");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
struct ASplitRound { double t0 = -1, t1 = -1, t2 = -1, t3 = -1, t4 = -1, t5 = -1; bool reused = false; };
struct ASplitState { std::vector<ASplitRound> rounds; bool flushed = false; };
ASplitState g_a_split;

void a_split_flush() {
    ASplitState& A = g_a_split;
    A.flushed = true;
    std::fprintf(stderr, "strata qwen35: A split (STRATA_QWEN35_A_SPLIT=1): rounds %d\n", (int) A.rounds.size());
    for (size_t i = 0; i < A.rounds.size(); ++i) {
        const ASplitRound& r = A.rounds[i];
        const double fill = (r.t0 >= 0 && r.t1 >= 0) ? (r.t1 - r.t0) * 1e3 : -1.0;
        const double entry = (r.t1 >= 0 && r.t2 >= 0) ? (r.t2 - r.t1) * 1e3 : -1.0;
        const double build = (r.t2 >= 0 && r.t3 >= 0) ? (r.t3 - r.t2) * 1e3 : -1.0;
        const double alloc = (r.t3 >= 0 && r.t4 >= 0) ? (r.t4 - r.t3) * 1e3 : -1.0;
        const double dispatch = (r.t4 >= 0 && r.t5 >= 0) ? (r.t5 - r.t4) * 1e3 : -1.0;
        const double a = (r.t0 >= 0 && r.t5 >= 0) ? (r.t5 - r.t0) * 1e3 : -1.0;
        std::fprintf(stderr, "strata qwen35: A split round %d: fill_ms %.3f entry_ms %.3f build_ms %.3f "
                             "alloc_ms %.3f dispatch_ms %.3f A_ms %.3f reused %d\n",
                     (int) i, fill, entry, build, alloc, dispatch, a, r.reused ? 1 : 0);
    }
}

// ---- Task 194 (exact STRATA_QWEN35_T1_HOST_TIMELINE=1, default off): bounded HOST timeline for request 2's
// ACTUAL decode evaluations 6..8 (120 layer records + 6 evaluation boundary marks).  Every stamp is a host
// steady_clock instant taken at a point that already exists on the seam path; no new CUDA call, event,
// synchronize, query or allocation is added, the gate-off path is inert (a single cached env check), and
// nothing is printed until the request ends.  The seam callbacks and the native tier run on the same compute
// thread, so the recorder is plain single-thread state (no TLS assumptions); the thread id is recorded as a
// correlation value when readily available.
// ==== T1H-196 INTERNALS BEGIN (private recorder state/helpers; stays in the anonymous namespace) ====
constexpr int kT1HMaxRecords = 144;   // 120 expected + bounded margin; overflow is a reported hard flag
struct T1HRecord {
    int64_t ns_pre0 = 0, ns_pre1 = 0, ns_ids_submit = 0, ns_ids_wait_b = 0, ns_ids_wait_e = 0;
    int64_t ns_hot_end = 0, ns_act_submit = 0, ns_act_wait_b = 0, ns_act_wait_e = 0;
    int64_t ns_cold_b = 0, ns_cold_e = 0, ns_h2d_end = 0, ns_combine_submit = 0;
    int64_t ns_service_return = 0, ns_post_enq = 0, ns_join = 0;
    int32_t eval = 0, layer = -1, hits = 0, misses = 0, act_state = 0;   // act_state: 0 absent(NA), 2 measured
    uint8_t valid = 0;
    long long tid = 0;
    // Task 202 pool-phase sink (requested only when the pool gate + an open record exist)
    uint8_t pool_requested = 0, pool_available = 0;
    int32_t pool_reason = -1, pool_njobs = 0;
    double pool_gu_incl = -1, pool_quant = -1, pool_down_incl = -1;
    double pool_gu_bp = -1, pool_gu_pub = -1, pool_gu_drain = -1, pool_gu_wd = -1, pool_gu_ap = -1;
    double pool_down_bp = -1, pool_down_pub = -1, pool_down_drain = -1, pool_down_wd = -1, pool_down_ap = -1;
};
struct T1HState {
    bool active = false;
    int request = 0, eval = 0, pending = -1, count = 0;
    bool overflow = false, dup = false;
    T1HRecord rec[kT1HMaxRecords];
    int64_t eval_begin_ns[9] = {}, eval_ready_ns[9] = {};
    int eval_layers[9] = {};
    long long tid = 0;
};
T1HState g_t1h;

int64_t t1h_now_ns() {
    return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
long long t1h_thread_id() {
    return (long long) std::hash<std::thread::id>{}(std::this_thread::get_id());
}
bool t1_host_timeline_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_T1_HOST_TIMELINE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Task 202: the pool gate is read once per process; it alone changes nothing (the HOST timeline recorder
/// must also be active and hold an open record for any timestamp to be taken).
bool t1_pool_phase_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QWEN35_T1_POOL_PHASE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
/// Emit-time bounded writer (one per request end when the gate is on; not a per-seam builder, no heap).
struct T1HEmit {
    char* buf = nullptr;
    size_t cap = 0, off = 0;
    bool ok = true;
    void put(const char* fmt, ...) {
        if (!ok) return;
        va_list ap;
        va_start(ap, fmt);
        const int n = std::vsnprintf(buf + off, cap - off, fmt, ap);
        va_end(ap);
        if (n < 0 || (size_t) n >= cap - off) { ok = false; return; }
        off += (size_t) n;
    }
};

// ==== T1H-196 INTERNALS END ====

}  // namespace (T1H private state ends; the public API below has external linkage - task 196 repair)

// ==== T1H-196 PUBLIC BEGIN (strata::core linkage, shared by bridge/expert/model translation units) ====

bool qwen35_t1h_enabled() { return t1_host_timeline_on(); }

void qwen35_t1h_eval_begin(int request, int eval) {
    if (!t1_host_timeline_on()) return;
    if (request != 2 || eval < 6 || eval > 8) return;   // contract: request 2, ACTUAL evaluations 6..8 only
    T1HState& T = g_t1h;
    if (T.pending >= 0) { T.rec[T.pending].valid = 0; T.dup = true; T.pending = -1; }
    T.active = true;
    T.request = request;
    T.eval = eval;
    T.eval_begin_ns[eval] = t1h_now_ns();
}

void qwen35_t1h_eval_ready(int request, int eval) {
    if (!t1_host_timeline_on()) return;
    if (request != 2 || eval < 6 || eval > 8) return;
    T1HState& T = g_t1h;
    if (!T.active) return;
    T.eval_ready_ns[eval] = t1h_now_ns();
    if (eval == 8) T.active = false;   // cleared only after the existing get-logits readiness of eval 8
}

void qwen35_t1h_scope_reset(int request) {
    if (!t1_host_timeline_on()) return;
    T1HState& T = g_t1h;
    // Error/EOS/reset safety: an open record or active scope never carries into another request.
    if (T.pending >= 0) { T.rec[T.pending].valid = 0; T.dup = true; T.pending = -1; }
    T.active = false;
    (void) request;
}

void qwen35_t1h_stamp(int field) {
    T1HState& T = g_t1h;
    if (!T.active || T.pending < 0) return;
    const int64_t ns = t1h_now_ns();
    T1HRecord& r = T.rec[T.pending];
    switch (field) {
        case QWEN35_T1H_PRE1: r.ns_pre1 = ns; break;
        case QWEN35_T1H_IDS_SUBMIT: r.ns_ids_submit = ns; break;
        case QWEN35_T1H_IDS_WAIT_B: r.ns_ids_wait_b = ns; break;
        case QWEN35_T1H_IDS_WAIT_E: r.ns_ids_wait_e = ns; break;
        case QWEN35_T1H_HOT_END: r.ns_hot_end = ns; break;
        case QWEN35_T1H_ACT_SUBMIT: r.ns_act_submit = ns; break;
        case QWEN35_T1H_ACT_WAIT_B: r.ns_act_wait_b = ns; break;
        case QWEN35_T1H_ACT_WAIT_E: r.ns_act_wait_e = ns; break;
        case QWEN35_T1H_COLD_B: r.ns_cold_b = ns; break;
        case QWEN35_T1H_COLD_E: r.ns_cold_e = ns; break;
        case QWEN35_T1H_H2D_END: r.ns_h2d_end = ns; break;
        case QWEN35_T1H_COMBINE_SUBMIT: r.ns_combine_submit = ns; break;
        case QWEN35_T1H_SERVICE_RETURN: r.ns_service_return = ns; break;
        case QWEN35_T1H_POST_ENQ: r.ns_post_enq = ns; break;
        case QWEN35_T1H_JOIN: r.ns_join = ns; break;
        default: break;
    }
}

void qwen35_t1h_note(int note, long long value) {
    T1HState& T = g_t1h;
    if (!T.active || T.pending < 0) return;
    T1HRecord& r = T.rec[T.pending];
    if (note == QWEN35_T1H_NOTE_HITS) r.hits = (int32_t) value;
    else if (note == QWEN35_T1H_NOTE_MISSES) r.misses = (int32_t) value;
    else if (note == QWEN35_T1H_NOTE_ACT_STATE) r.act_state = (int32_t) value;
}

void qwen35_t1h_begin_layer(int layer, bool t1) {
    T1HState& T = g_t1h;
    if (!T.active || !t1) return;
    if (T.pending >= 0) { T.rec[T.pending].valid = 0; T.dup = true; }
    if (T.count >= kT1HMaxRecords) { T.overflow = true; T.pending = -1; return; }
    T.pending = T.count;
    T1HRecord& r = T.rec[T.count];
    r = T1HRecord();
    r.eval = T.eval;
    r.layer = (int32_t) layer;
    r.ns_pre0 = t1h_now_ns();
    if (T.tid == 0) T.tid = t1h_thread_id();
    r.tid = T.tid;
    if (T.eval >= 0 && T.eval < 9) ++T.eval_layers[T.eval];
}

void qwen35_t1h_end_layer() {
    T1HState& T = g_t1h;
    if (!T.active || T.pending < 0) return;
    T1HRecord& r = T.rec[T.pending];
    r.valid = 1;
    ++T.count;
    T.pending = -1;
}

bool qwen35_t1h_pool_phase_wanted() {
    return t1_pool_phase_on() && t1_host_timeline_on() && g_t1h.active && g_t1h.pending >= 0;
}

void qwen35_t1h_pool_phase(const strata::kernels::cpu::SlicePhaseTiming* timing) {
    if (timing == nullptr || !t1_pool_phase_on()) return;
    T1HState& T = g_t1h;
    if (!T.active || T.pending < 0) return;
    T1HRecord& r = T.rec[T.pending];
    r.pool_requested = 1;
    r.pool_available = timing->available ? 1 : 0;
    r.pool_reason = timing->available ? 0 : timing->reason;
    r.pool_njobs = (int32_t) timing->n_jobs;
    r.pool_gu_incl = timing->gu_inclusive_ms;
    r.pool_quant = timing->quant_ms;
    r.pool_down_incl = timing->down_inclusive_ms;
    r.pool_gu_bp = timing->gu_before_park_ms;
    r.pool_gu_pub = timing->gu_publish_ms;
    r.pool_gu_drain = timing->gu_host_drain_ms;
    r.pool_gu_wd = timing->gu_wait_done_ms;
    r.pool_gu_ap = timing->gu_after_park_ms;
    r.pool_down_bp = timing->down_before_park_ms;
    r.pool_down_pub = timing->down_publish_ms;
    r.pool_down_drain = timing->down_host_drain_ms;
    r.pool_down_wd = timing->down_wait_done_ms;
    r.pool_down_ap = timing->down_after_park_ms;
}

void qwen35_t1h_emit_request_end(int request) {
    if (!t1_host_timeline_on() || request != 2) return;
    T1HState& T = g_t1h;
    bool complete = (T.count == 120) && !T.overflow && !T.dup;
    for (int e = 6; e <= 8; ++e)
        complete = complete && T.eval_layers[e] == 40 && T.eval_begin_ns[e] != 0 && T.eval_ready_ns[e] != 0;
    static char t1h_buf[1 << 17];
    T1HEmit E{t1h_buf, sizeof(t1h_buf), 0, true};
    E.put("{\"event\":\"t1_host_timeline\",\"request\":2,\"clock\":\"steady_clock_ns\",\"complete\":%s,"
          "\"count\":%d,\"overflow\":%s,\"dup\":%s,\"layer_capacity\":%d,"
          "\"eval6_begin\":%lld,\"eval6_ready\":%lld,\"eval7_begin\":%lld,\"eval7_ready\":%lld,"
          "\"eval8_begin\":%lld,\"eval8_ready\":%lld,\"eval_layers\":[%d,%d,%d],\"records\":[",
          complete ? "true" : "false", T.count, T.overflow ? "true" : "false", T.dup ? "true" : "false",
          kT1HMaxRecords, (long long) T.eval_begin_ns[6], (long long) T.eval_ready_ns[6],
          (long long) T.eval_begin_ns[7], (long long) T.eval_ready_ns[7],
          (long long) T.eval_begin_ns[8], (long long) T.eval_ready_ns[8],
          T.eval_layers[6], T.eval_layers[7], T.eval_layers[8]);
    for (int i = 0; i < T.count && E.ok; ++i) {
        const T1HRecord& r = T.rec[i];
        E.put("%s{\"eval\":%d,\"layer\":%d,\"pre0\":%lld,\"pre1\":%lld,\"ids_submit\":%lld,\"ids_wait_b\":%lld,"
              "\"ids_wait_e\":%lld,\"hot_end\":%lld,\"act_submit\":%lld,\"act_wait_b\":%lld,\"act_wait_e\":%lld,"
              "\"cold_b\":%lld,\"cold_e\":%lld,\"h2d_end\":%lld,\"combine_submit\":%lld,\"service_return\":%lld,"
              "\"post_enq\":%lld,\"join\":%lld,\"hits\":%d,\"misses\":%d,\"act_state\":%d,\"tid\":%lld,"
              "\"valid\":%d,",
              i ? "," : "", r.eval, r.layer, (long long) r.ns_pre0, (long long) r.ns_pre1,
              (long long) r.ns_ids_submit, (long long) r.ns_ids_wait_b, (long long) r.ns_ids_wait_e,
              (long long) r.ns_hot_end, (long long) r.ns_act_submit, (long long) r.ns_act_wait_b,
              (long long) r.ns_act_wait_e, (long long) r.ns_cold_b, (long long) r.ns_cold_e,
              (long long) r.ns_h2d_end, (long long) r.ns_combine_submit, (long long) r.ns_service_return,
              (long long) r.ns_post_enq, (long long) r.ns_join, r.hits, r.misses, r.act_state, r.tid,
              (int) r.valid);
        if (r.pool_requested) {
            E.put("\"pool_phase_ms\":{\"available\":%s,\"reason\":%d,\"n_jobs\":%d,"
                  "\"gu_inclusive\":%.6g,\"quant\":%.6g,\"down_inclusive\":%.6g,"
                  "\"gu_before_park\":%.6g,\"gu_publish\":%.6g,\"gu_host_drain\":%.6g,\"gu_wait_done\":%.6g,"
                  "\"gu_after_park\":%.6g,\"down_before_park\":%.6g,\"down_publish\":%.6g,"
                  "\"down_host_drain\":%.6g,\"down_wait_done\":%.6g,\"down_after_park\":%.6g}}",
                  r.pool_available ? "true" : "false", r.pool_reason, (int) r.pool_njobs,
                  r.pool_gu_incl, r.pool_quant, r.pool_down_incl, r.pool_gu_bp, r.pool_gu_pub, r.pool_gu_drain,
                  r.pool_gu_wd, r.pool_gu_ap, r.pool_down_bp, r.pool_down_pub, r.pool_down_drain, r.pool_down_wd,
                  r.pool_down_ap);
        } else {
            E.put("\"pool_phase_ms\":null}");
        }
    }
    E.put("]}\n");
    if (!E.ok) {
        // A truncated record must never look like a success: one compact, complete hard-diagnostic line.
        std::fprintf(stdout,
                     "{\"event\":\"t1_host_timeline\",\"request\":2,\"status\":\"serialization_overflow\","
                     "\"count\":%d,\"overflow\":%s,\"dup\":%s}\n",
                     T.count, T.overflow ? "true" : "false", T.dup ? "true" : "false");
        std::fflush(stdout);
        return;
    }
    std::fwrite(t1h_buf, 1, E.off, stdout);
    std::fflush(stdout);
}

// ==== T1H-196 PUBLIC END ====

namespace {   // the file's original anonymous helpers resume

/// P271 (default off, exact STRATA_QWEN35_SMALL_T_NATIVE=1): one 2..5-token routed MoE node through the
/// per-token native decode service.  The graph still builds the real batch node and the GDN/attention/
/// rollback semantics around it are untouched; only this node's expert math is re-routed, one row at a time,
/// exactly as a one-token decode layer.  FAIL-CLOSED: with the gate on, any input layout the per-token calls
/// cannot address byte-exactly (dtype, ne, nb, extent) fails the seam instead of falling back to the prompt
/// pipeline, so a mis-shaped batch can never silently take a different path.
///
/// LIFECYCLE.  The service returns with its combine - and the miss-row H2Ds that combine waits on - still in
/// flight on the shared work/copy streams, and the next row rewrites the tier's activation/meta/parts buffers
/// and this layer's pinned staging.  The per-row fence below waits for the work-stream tail BEFORE the next
/// row's copies are enqueued; the work tail is itself ordered behind the row's copy uploads (the service's
/// miss-event wait), so the fence proves both "parts no longer read" and "host staging no longer read".
/// Stream order alone is NOT enough across rows: the copy stream is independent of work, and re-recording the
/// service's shared events is only benign because the fence lets every earlier wait retire first.
bool small_t_native_rows(State& S, LayerState& L, int layer, const ggml_tensor* node, std::string& err,
                         void* out_override = nullptr, uint64_t out_pitch = 0) {
    const ggml_tensor* x = node->src[0];
    const ggml_tensor* ids = node->src[1];
    const ggml_tensor* w = node->src[2];
    const int64_t T = node->ne[1];
    const uint64_t x_row = (uint64_t) S.H * sizeof(float);
    const uint64_t id_row = (uint64_t) S.K * sizeof(int32_t);
    const uint64_t w_row = (uint64_t) S.K * sizeof(float);
    // Each per-token call copies ONE contiguous row of each input and writes one row of the output: x, ids,
    // weights and the node output must be exactly that, positioned by the tensors' OWN strides.  The weights'
    // token axis is nb[2] and the ids'/x's row pitches may be wider than their payloads (the argsort view), so
    // no K*T / H*T linear guess is used anywhere below.
    const bool shapes_ok =
        node != nullptr && x != nullptr && ids != nullptr && w != nullptr && node->data != nullptr &&
        x->data != nullptr && ids->data != nullptr && w->data != nullptr && T >= 2 && T <= 5 &&
        node->type == GGML_TYPE_F32 && node->ne[0] == S.H && node->ne[1] == T &&
        node->ne[2] == 1 && node->ne[3] == 1 && node->nb[0] == sizeof(float) && node->nb[1] >= x_row &&
        (uint64_t) (T - 1) * node->nb[1] + x_row <= ggml_nbytes(node) &&
        x->type == GGML_TYPE_F32 && x->ne[0] == S.H && x->ne[1] == T && x->ne[2] == 1 && x->ne[3] == 1 &&
        x->nb[0] == sizeof(float) && x->nb[1] >= x_row &&
        (uint64_t) (T - 1) * x->nb[1] + x_row <= ggml_nbytes(x) &&
        ids->type == GGML_TYPE_I32 && ids->ne[0] == S.K && ids->ne[1] == T && ids->ne[2] == 1 &&
        ids->ne[3] == 1 && ids->nb[0] == sizeof(int32_t) && ids->nb[1] >= id_row &&
        (uint64_t) (T - 1) * ids->nb[1] + id_row <= ggml_nbytes(ids) &&
        w->type == GGML_TYPE_F32 && w->ne[0] == 1 && w->ne[1] == S.K && w->ne[2] == T && w->ne[3] == 1 &&
        w->nb[0] == sizeof(float) && w->nb[1] == sizeof(float) && w->nb[2] >= w_row &&
        (uint64_t) (T - 1) * w->nb[2] + w_row <= ggml_nbytes(w);
    if (!shapes_ok) {
        err = "small-T native: the batched tensors are not per-token contiguous rows (x/ids/weights/output "
              "dtype/ne/nb/extent)";
        return false;
    }
    if (L.small_t_fence == nullptr &&
        cudaEventCreateWithFlags(&L.small_t_fence, cudaEventDisableTiming) != cudaSuccess) {
        err = "small-T native: cannot create the per-row work fence event";
        return false;
    }
    core::NativeDecodeStage st;
    st.work_stream = L.work;
    st.copy_stream = L.copy;
    st.host_ids = L.host_ids;
    st.host_act = L.host_act;
    st.host_miss = L.host_miss;
    st.is_hit = L.is_hit.data();
    for (int64_t t = 0; t < T; ++t) {
        const uint8_t* x_t = (const uint8_t*) x->data + (size_t) t * x->nb[1];
        const uint8_t* ids_t = (const uint8_t*) ids->data + (size_t) t * ids->nb[1];
        const uint8_t* w_t = (const uint8_t*) w->data + (size_t) t * w->nb[2];
        uint8_t* out_t = out_override != nullptr
                             ? (uint8_t*) out_override + (size_t) t * out_pitch
                             : (uint8_t*) node->data + (size_t) t * node->nb[1];
        if (!expert_native_decode_layer(S.dispatch, layer, S.K, x_t, ids_t, w_t, out_t, st, err)) return false;
        ++S.small_t_rows;
        // The row's work-stream tail (grouped hits -> [miss-upload wait] -> combine) must be COMPLETE before
        // the next row's launches/copies may rewrite the shared buffers this tail is still reading.
        if (cudaEventRecord(L.small_t_fence, L.work) != cudaSuccess ||
            cudaEventSynchronize(L.small_t_fence) != cudaSuccess) {
            err = "small-T native: the per-row work fence failed";
            return false;
        }
    }
    ++S.small_t_layers;
    return true;
}

/// Task 284 (default off; exact STRATA_QWEN35_SMALL_T_MULTI=1, and only inside the SMALL_T_NATIVE gate's
/// branch): ONE multirow window for a 2..5-token routed MoE node through `expert_native_decode_window`.  The
/// graph still builds the real batch node; GDN/attention/rollback and every other node are untouched.  The
/// shape contract is the SAME fail-closed one the per-row path uses: with the gate on, any layout the window
/// cannot address byte-exactly (dtype, ne, nb, extent) fails the seam instead of falling back.  The window
/// service synchronises its own single work fence before returning, so no per-row fence exists here.
bool small_t_multi_window(State& S, LayerState& L, int layer, const ggml_tensor* node, std::string& err) {
    const ggml_tensor* x = node->src[0];
    const ggml_tensor* ids = node->src[1];
    const ggml_tensor* w = node->src[2];
    const int64_t T = node->ne[1];
    const uint64_t x_row = (uint64_t) S.H * sizeof(float);
    const uint64_t id_row = (uint64_t) S.K * sizeof(int32_t);
    const uint64_t w_row = (uint64_t) S.K * sizeof(float);
    const bool shapes_ok =
        node != nullptr && x != nullptr && ids != nullptr && w != nullptr && node->data != nullptr &&
        x->data != nullptr && ids->data != nullptr && w->data != nullptr && T >= 2 && T <= 5 &&
        node->type == GGML_TYPE_F32 && node->ne[0] == S.H && node->ne[1] == T &&
        node->ne[2] == 1 && node->ne[3] == 1 && node->nb[0] == sizeof(float) && node->nb[1] >= x_row &&
        (uint64_t) (T - 1) * node->nb[1] + x_row <= ggml_nbytes(node) &&
        x->type == GGML_TYPE_F32 && x->ne[0] == S.H && x->ne[1] == T && x->ne[2] == 1 && x->ne[3] == 1 &&
        x->nb[0] == sizeof(float) && x->nb[1] >= x_row &&
        (uint64_t) (T - 1) * x->nb[1] + x_row <= ggml_nbytes(x) &&
        ids->type == GGML_TYPE_I32 && ids->ne[0] == S.K && ids->ne[1] == T && ids->ne[2] == 1 &&
        ids->ne[3] == 1 && ids->nb[0] == sizeof(int32_t) && ids->nb[1] >= id_row &&
        (uint64_t) (T - 1) * ids->nb[1] + id_row <= ggml_nbytes(ids) &&
        w->type == GGML_TYPE_F32 && w->ne[0] == 1 && w->ne[1] == S.K && w->ne[2] == T && w->ne[3] == 1 &&
        w->nb[0] == sizeof(float) && w->nb[1] == sizeof(float) && w->nb[2] >= w_row &&
        (uint64_t) (T - 1) * w->nb[2] + w_row <= ggml_nbytes(w);
    if (!shapes_ok) {
        err = "small-T multirow: the batched tensors are not per-token rows (x/ids/weights/output "
              "dtype/ne/nb/extent)";
        return false;
    }
    core::NativeDecodeStage st;
    st.work_stream = L.work;
    st.copy_stream = L.copy;
    st.host_ids = L.host_ids;
    st.host_act = L.host_act;
    st.host_miss = L.host_miss;
    st.is_hit = L.is_hit.data();
    if (!expert_native_decode_window(S.dispatch, layer, T, S.K, x->data, x->nb[1], ids->data, ids->nb[1],
                                     w->data, w->nb[2], node->data, node->nb[1], st, err)) {
        return false;
    }
    ++S.small_t_multi_windows;
    S.small_t_multi_rows += (uint64_t) T;
    return true;
}

/// M272 diagnostic (default off; only called from the small-T gate's branch on T == 2): one fixed layer of the
/// FIRST target T2 batch runs BOTH ways on the same real node inputs - the ORIGINAL prompt pipeline as the
/// reference, into a private device buffer the graph never reads, and the per-token native service into the
/// node output, which stays the candidate result (it is written last).  The native per-row fences wait on the
/// work-stream tail, and the reference combine was enqueued on that same stream before the rows, so after
/// `small_t_native_rows` returns both buffers are settled; the host copy can then read both without a second
/// sync.  The gate is the existing native-parity contract: all-finite and L1 rel = sum|nat-ref|/sum|ref| <
/// 3e-2, with an exact absolute guard when the reference is all zeros.  Every failing path (alloc, reference
/// call, copy, non-finite, parity) fails the seam loudly - there is no silent fallback and no CPU simulation.
/// CAVEAT: the reference call perturbs the tier's admission/cache and the workspace-loan evolution; this is a
/// diagnostic arm, not a performance arm.
bool small_t_compare_layer(State& S, LayerState& L, int layer, ggml_tensor* node, std::string& err) {
    const ggml_tensor* x = node->src[0];
    const ggml_tensor* ids = node->src[1];
    const ggml_tensor* w = node->src[2];
    const int64_t T = node->ne[1];
    const int64_t N = S.dispatch.n_embd;
    if (N <= 0 || N != S.H) {
        err = "small-T compare: the dispatch geometry does not match the bridge";
        return false;
    }
    if (x == nullptr || ids == nullptr || w == nullptr || node->data == nullptr) {
        err = "small-T compare: a node input is missing";
        return false;
    }
    const size_t row = (size_t) N * sizeof(float);
    const size_t bytes = (size_t) T * row;
    if (S.small_t_compare_dev == nullptr) {
        if (cudaMalloc((void**) &S.small_t_compare_dev, bytes) != cudaSuccess) {
            err = "small-T compare: the reference device buffer could not be allocated";
            return false;
        }
        S.small_t_compare_bytes = bytes;
    } else if (S.small_t_compare_bytes < bytes) {
        err = "small-T compare: the reference device buffer is smaller than this batch";
        return false;
    }
    // Reference FIRST, into the private buffer (never the node output); a failure here is fatal for the run
    // (fail-closed, no fallback to the candidate or the prompt path).
    if (small_t_multi_requested()) {
        // Task 284 diagnostic arm: the reference is the ORIGINAL per-row native path (the previous candidate),
        // written per row into the compact private buffer with its OWN row pitch; the candidate is the new
        // multirow window into the node output, which the graph keeps consuming (it is written last).
        if (!small_t_native_rows(S, L, layer, node, err, S.small_t_compare_dev, row)) return false;
        if (!small_t_multi_window(S, L, layer, node, err)) return false;
    } else {
        // M272: the reference is the original PROMPT pipeline into the private buffer.
        if (!qwen_prefill_experts(S.dispatch, layer, T, x->data, x->nb[1], ids->data, ids->nb[1], w->data,
                                  w->nb[1], w->nb[2], S.small_t_compare_dev, L.work, err)) {
            return false;
        }
        // Candidate: the per-token native service into the node output.
        if (!small_t_native_rows(S, L, layer, node, err)) return false;
    }
    // Both buffers are settled: the reference combine sits on L.work before the native rows, and the native
    // rows' per-row fences already synchronised L.work's tail after that combine.  Copy both back row by row
    // through the tensors' OWN row pitches (the node output is a real graph tensor; its rows carry nb[1]).
    std::vector<float> ref((size_t) T * (size_t) N), nat((size_t) T * (size_t) N);
    for (int64_t t = 0; t < T; ++t) {
        if (cudaMemcpy(ref.data() + (size_t) t * (size_t) N,
                       (const uint8_t*) S.small_t_compare_dev + (size_t) t * row, row,
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(nat.data() + (size_t) t * (size_t) N,
                       (const uint8_t*) node->data + (size_t) t * node->nb[1], row,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "small-T compare: the host copy of the two outputs failed";
            return false;
        }
    }
    const size_t n = ref.size();
    uint64_t nonfinite_ref = 0, nonfinite_nat = 0;
    double sum_abs_ref = 0.0, sum_abs_delta = 0.0, sum_sq_ref = 0.0, sum_sq_delta = 0.0;
    double max_abs = 0.0, max_el_rel = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double r = ref[i], a = nat[i];
        if (!std::isfinite(r)) ++nonfinite_ref;
        if (!std::isfinite(a)) ++nonfinite_nat;
        const double d = a - r;
        sum_abs_ref += std::fabs(r);
        sum_abs_delta += std::fabs(d);
        sum_sq_ref += r * r;
        sum_sq_delta += d * d;
        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
        const double er = std::fabs(d) / (std::fabs(r) + 1e-6);
        if (er > max_el_rel) max_el_rel = er;
    }
    const bool ref_all_zero = (sum_abs_ref == 0.0 && sum_sq_ref == 0.0);
    const double rel = ref_all_zero ? -1.0 : sum_abs_delta / sum_abs_ref;
    const double norm_rms = ref_all_zero ? -1.0 : std::sqrt(sum_sq_delta / sum_sq_ref);
    bool pass = (nonfinite_ref == 0 && nonfinite_nat == 0);
    if (ref_all_zero) {
        // A zero reference makes the ratio meaningless; the guard is exact: the candidate must also be zero.
        pass = pass && (sum_abs_delta == 0.0 && max_abs == 0.0);
    } else {
        pass = pass && (rel < 3e-2);
    }
    std::fprintf(stderr,
                 "strata qwen35: small_t_compare {\"seq\":%llu,\"layer\":%d,\"T\":%lld,\"elems\":%llu,"
                 "\"decode_layers\":%llu,\"small_t_layers\":%llu,\"prefill_layers\":%llu,"
                 "\"nonfinite_ref\":%llu,\"nonfinite_nat\":%llu,\"sum_abs_ref\":%.10g,\"sum_abs_delta\":%.10g,"
                 "\"rel\":%.10g,\"max_abs\":%.10g,\"max_el_rel\":%.10g,\"norm_rms\":%.10g,"
                 "\"ref_all_zero\":%s,\"threshold\":0.03,\"pass\":%s}\n",
                 (unsigned long long) (S.small_t_compare_layers + 1), layer, (long long) T,
                 (unsigned long long) n, (unsigned long long) S.decode_layers,
                 (unsigned long long) S.small_t_layers, (unsigned long long) S.prefill_layers,
                 (unsigned long long) nonfinite_ref, (unsigned long long) nonfinite_nat, sum_abs_ref,
                 sum_abs_delta, rel, max_abs, max_el_rel, norm_rms, ref_all_zero ? "true" : "false",
                 pass ? "true" : "false");
    std::fflush(stderr);
    S.small_t_compare_mask |= (1ull << layer);
    ++S.small_t_compare_layers;
    if (!pass) {
        ++S.small_t_compare_fail;
        err = "small-T compare: the native-vs-prompt parity gate failed";
        return false;
    }
    ++S.small_t_compare_pass;
    return true;
}

bool seam_exec(ggml_backend_t backend, ggml_tensor* node) {
    State& S = *g_state;
    ExpertDispatch& d = S.dispatch;
    const bool trace = std::getenv("STRATA_QWEN35_TRACE") != nullptr;
    const int layer = layer_from_name(node->name);
    if (trace) std::fprintf(stderr, "qwen35 trace: seam_exec layer %d begin\n", layer);
    if (layer < 0 || layer >= (int) S.layers.size() || S.layers[(size_t) layer] == nullptr) return false;
    LayerState& L = *S.layers[(size_t) layer];
    if (!L.boundary_init && !boundary_init(L, backend, S.K, S.H)) {
        std::fprintf(stderr, "strata qwen35: layer %d's graph boundary could not initialise\n", layer);
        return false;
    }
    const int64_t K = S.K;
    const double t0 = now_seconds();
    // Task 194 (default off): open the host-timeline record at this seam entry (T1 layers only).
    qwen35_t1h_begin_layer(layer, node->ne[1] == 1);
    // Task 133 (off by default): close the host-gap/device-span window opened at the previous seam exit.
    if (seam_gap_enabled()) seam_gap_enter(backend, layer, node->ne[1] == 1, t0);
    // Task 134 (off by default): the first seam callback of the current round closes the A window.
    if (a_split_enabled() && !g_a_split.rounds.empty() && g_a_split.rounds.back().t5 < 0) {
        g_a_split.rounds.back().t5 = t0;
    }
    // The overlap prototype: the decode pre marker already recorded `pre` before the shared expert was
    // submitted.  `join`, recorded now, covers the whole backend-stream prefix (router + shared expert); the
    // work stream waits on it, so the CPU miss window stays free to overlap the shared expert while the hits
    // and the combine cannot touch arena regions the shared expert may still be reading.  Without a fresh
    // marker pre (the default path) this is the ordinary pre record.
    const bool fresh_pre = node->ne[1] == 1 && L.pre_fresh && L.join != nullptr;
    L.pre_fresh = false;
    if (fresh_pre) {
        ggml_backend_event_record(L.join, backend);
        cudaStreamWaitEvent(L.work, (cudaEvent_t) L.join->context, 0);
        ++S.overlap_join_layers;
        if (trace) std::fprintf(stderr, "qwen35 trace: seam layer %d join recorded (fresh pre)\n", layer);
    } else {
        ggml_backend_event_record(L.pre, backend);
    }
    cudaStreamWaitEvent(L.copy, (cudaEvent_t) L.pre->context, 0);
    cudaStreamWaitEvent(L.work, (cudaEvent_t) L.pre->context, 0);
    qwen35_t1h_stamp(QWEN35_T1H_PRE1);   // task 194: pre event recorded + copy/work waits enqueued
    // Step probe (default off): the graph pre/work dependencies are established; capture A/W BEFORE the
    // original service runs so no output/alias can affect the input snapshot.
    bool step_active = false;
    if (!step_probe_begin(S, L, node, layer, step_active)) return false;
    // Task 325 (default off): the bounded t3 slice pre-capture, on the same established pre/work order and
    // strictly BEFORE the original service (source snapshot cannot alias the output).  Inert when disarmed.
    bool t3_probe_active = false;
    if (!t3_probe_begin(S, L, node, layer, t3_probe_active)) return false;
    std::string err;
    if (node->ne[1] == 1) {
        // The WHOLE decode layer's routed-MoE dispatch (ids transfer, residency, GPU hits, CPU misses, upload,
        // combine, admission) is the shared service's; the boundary only orders it on its own streams.
        core::NativeDecodeStage st;
        st.work_stream = L.work;
        st.copy_stream = L.copy;
        st.host_ids = L.host_ids;
        st.host_act = L.host_act;
        st.host_miss = L.host_miss;
        st.is_hit = L.is_hit.data();
        if (!expert_native_decode_layer(d, layer, K, node->src[0]->data, node->src[1]->data, node->src[2]->data,
                                        node->data, st, err)) {
            std::fprintf(stderr, "strata qwen35: layer %d: %s (%s)\n", layer, err.c_str(),
                         d.fail != nullptr ? d.fail : "no dispatch message");
            return false;
        }
        qwen35_t1h_stamp(QWEN35_T1H_SERVICE_RETURN);   // task 196: explicitly named extra instant (NOT post/join)
        // Step probe (default off): capture B on the same work stream after the service, with one checked sync;
        // `count` in its record line is S.decode_layers BEFORE this layer's increment (241 before index 7).
        if (!step_probe_end(S, L, node, step_active)) return false;
        if (trace) std::fprintf(stderr, "qwen35 trace: seam layer %d ids [%d %d %d %d %d %d %d %d]\n", layer,
                                L.host_ids[0], L.host_ids[1], L.host_ids[2], L.host_ids[3], L.host_ids[4],
                                L.host_ids[5], L.host_ids[6], L.host_ids[7]);
        if (!S.cfg.routing_log.empty()) {
            std::string routing;
            for (int64_t r = 0; r < K; ++r) {
                routing += std::to_string(layer);
                routing += ',';
                routing += std::to_string(L.host_ids[r]);
                routing += '\n';
            }
            append_routing(routing);
        }
        // Routing-detail rows (default off; request-scoped, hard-capped).  Schema (see the request marker):
        // before_count,layer,rank,id,hit,slot_after.  before_count is S.decode_layers BEFORE this layer's
        // increment; the marker's decode_base is the already-counted prompt-tail T1 call, so the logged rows
        // start after it and cover only post-prompt layers (the fixed short teacher8 run: 2240 rows / 280
        // layers, before_count 1..280).  2248 is only a hard cap: for a longer run the log is a prefix and the
        // row sums then match the stats deltas only up to that prefix - they never cover the whole request.
        // slot_after is the read-only residence query at service return (admissions made inside the call are
        // already visible); it is NOT the slot the kernel actually used.  With the diagnostic enabled, each
        // layer's rows are flushed and the stream is checked, so a delayed write error stops the run through
        // the seam error path; that per-layer flush perturbs scheduling - not a speed-comparison configuration.
        if (S.routing_detail_armed && S.routing_detail_rows < 2248) {
            const int64_t before_count = (int64_t) S.decode_layers;
            for (int64_t r = 0; r < K && S.routing_detail_rows < 2248; ++r) {
                const int32_t id = L.host_ids[r];
                const int32_t slot_after = (S.dispatch.cache != nullptr && id >= 0 && id < S.NE)
                                               ? S.dispatch.cache->slot_of(layer, id)
                                               : (int32_t) -1;
                S.routing_detail << before_count << ',' << layer << ',' << r << ',' << id << ','
                                 << (int) (L.is_hit[(size_t) r] != 0) << ',' << slot_after << '\n';
                ++S.routing_detail_rows;
            }
            S.routing_detail.flush();
            if (!S.routing_detail.good()) {
                // The diagnostic was requested: an incomplete file must not look like a pass.
                std::fprintf(stderr, "strata qwen35: routing detail log write/flush failed at layer %d\n", layer);
                return false;
            }
        }
        ++S.decode_layers;
        S.jobs.fetch_add((uint64_t) K, std::memory_order_relaxed);
    } else {
        // Prompt layer: the shared service's GPU expert pipeline (grouped experts, resident cache slots, the
        // bounded cold ring, the MMQ products).  Its counters carry the evidence.
        const int64_t T = node->ne[1];
        if (small_t_native_requested() && T >= 2 && T <= 5) {
            // P271 prototype (default off): the small verify batch runs row by row through the SAME native
            // decode service the one-token path uses; a shape this path cannot address byte-exactly fails
            // closed (no fallback to the prompt pipeline).
            // M272 diagnostic (default off): on a T==2 node of a fixed layer, at most once per layer and five
            // layers in total, the layer runs through small_t_compare_layer instead (reference + native).
            const bool compare_here = small_t_compare_requested() && T == 2 &&
                                      small_t_compare_layer_is_target(layer) &&
                                      (S.small_t_compare_mask & (1ull << layer)) == 0 &&
                                      S.small_t_compare_layers < 5;
            const bool small_t_ok =
                compare_here ? small_t_compare_layer(S, L, layer, node, err)
                             : (small_t_multi_requested() ? small_t_multi_window(S, L, layer, node, err)
                                                          : small_t_native_rows(S, L, layer, node, err));
            if (!small_t_ok) {
                std::fprintf(stderr, "strata qwen35: small-T %s layer %d: %s (%s)\n",
                             compare_here ? "compare" : "native", layer, err.c_str(),
                             d.fail != nullptr ? d.fail : "no dispatch message");
                return false;
            }
        } else {
            // Task 127 (diagnostic, default off): hand the prompt path the graph-prefix boundary event so the
            // optional d2h split can attribute its wait; zero calls when the exact gate is unset.
            if (qwen_prefill_d2h_split_on()) qwen_prefill_d2h_split_set_pre((void*) (cudaEvent_t) L.pre->context);
            if (!qwen_prefill_experts(d, layer, T, node->src[0]->data, node->src[0]->nb[1], node->src[1]->data,
                                      node->src[1]->nb[1], node->src[2]->data, node->src[2]->nb[1], node->src[2]->nb[2],
                                      node->data, L.work, err)) {
                std::fprintf(stderr, "strata qwen35: prompt layer %d: %s (%s)\n", layer, err.c_str(),
                             d.fail != nullptr ? d.fail : "no dispatch message");
                return false;
            }
            S.prefill_layers += 1;
            S.prefill_entries += (uint64_t) (T * K);
            S.jobs.fetch_add((uint64_t) (T * K), std::memory_order_relaxed);
        }
    }
    // Task 325 (default off): the bounded output capture AFTER the original service returned, on the same
    // work stream and BEFORE the original post record/join (one checked sync only when a capture is active).
    if (!t3_probe_end(S, L, node, t3_probe_active)) return false;
    if (cudaEventRecord((cudaEvent_t) L.post->context, L.work) != cudaSuccess) return false;
    qwen35_t1h_stamp(QWEN35_T1H_POST_ENQ);   // task 196: host instant of the ORIGINAL post-event record
    ggml_backend_event_wait(backend, L.post);
    qwen35_t1h_stamp(QWEN35_T1H_JOIN);       // task 196: host instant of the ORIGINAL backend wait enqueue
    qwen35_t1h_end_layer();                  // closes the record only after the final required boundary
    {
        const cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            std::fprintf(stderr, "strata qwen35: decode seam CUDA error: %s\n", cudaGetErrorString(cerr));
            return false;
        }
    }
    S.seconds_us.fetch_add((uint64_t) ((now_seconds() - t0) * 1e6), std::memory_order_relaxed);
    // Task 133 (off by default): open the next window's backend-stream event at this seam's exit.
    if (seam_gap_enabled()) seam_gap_exit(backend, now_seconds());
    return true;
}

bool seam_callback(ggml_backend_t backend, ggml_tensor* node, bool ask, void*) {
    if (g_state == nullptr || !g_state->initialized || !g_state->seam_armed.load(std::memory_order_relaxed))
        return false;
    if (node->op != GGML_OP_CUSTOM) return false;
    if (std::strncmp(node->name, "strata_moe_pre-", 15) == 0) {
        // The overlap prototype's decode pre marker: exactly one device source (the activation, one token).
        // Any other node that happens to carry the name is not claimed.
        if (node->src[0] == nullptr || node->src[1] != nullptr || node->src[0]->type != GGML_TYPE_F32 ||
            node->src[0]->ne[0] != g_state->H || node->src[0]->ne[1] != 1)
            return false;
        if (ask) return true;
        return seam_marker_exec(backend, node);
    }
    if (std::strncmp(node->name, "ffn_moe_out-", 12) != 0) return false;
    // llama.cpp's `cb()` renames BOTH custom MoE nodes to ffn_moe_out-<layer>; ours carry exactly the three
    // device inputs (x, ids, weights).  Any other custom node (the CPU fallback has 6 inputs) is not claimed.
    if (node->src[0] == nullptr || node->src[1] == nullptr || node->src[2] == nullptr || node->src[3] != nullptr)
        return false;
    if (ask) return true;
    return seam_exec(backend, node);
}

}  // namespace

/// Task 325 (exact STRATA_QWEN35_MTP_T3_MOE_SLICE=1, default off): the phase setter for the bounded T3 MoE
/// slice probe.  The transaction calls it ONLY around the exact warm round 1's real REF T1 decodes (phase 1,
/// row 0..2) and the one actual ACT T3 (phase 2, row -1), disarming (phase 0) immediately after each decode.
/// Zero CUDA calls, copies, syncs or file I/O here; with the gate unset it is an inert cached-bool test.  A
/// phase/row outside the exact contract, or an arm while a capture is still in flight, records a sticky
/// fail-closed flag that the next seam callback refuses loudly - the probe never silently fakes a capture.
void qwen35_bridge_t3_probe_phase(int phase, int row) {
    if (!t3_probe_requested()) return;
    State* S = g_state;
    if (S == nullptr || !S->initialized) return;
    const bool args_ok = (phase == 0) || (phase == 1 && row >= 0 && row <= 2) ||
                         (phase == 2 && row == -1);
    if (!args_ok || S->t3_probe_pre) {
        S->t3_probe_error = true;
        S->t3_probe_phase = 0;
        S->t3_probe_row = -1;
        std::fprintf(stderr, "strata qwen35: t3 probe: unexpected phase setter call (phase %d row %d, "
                             "in-flight %d); fail closed\n", phase, row, (int) S->t3_probe_pre);
        std::fflush(stderr);
        return;
    }
    S->t3_probe_phase = phase;
    S->t3_probe_row = (phase == 0) ? -1 : row;
}

/// Task 134 (diagnostic, exact STRATA_QWEN35_A_SPLIT=1, default off): phase marks for the A window, called
/// from the runner (0 = request start, 1 = batch fill done) and from llama's process_ubatch (2 = entry,
/// 3 = graph built, 4 = sched allocation done, 6 = graph reused); 5 is recorded internally at the first seam
/// callback of the round.  One no-op call when the gate is off (no CUDA call, no output, no timing change).
void qwen35_a_split_mark(int phase) {
    if (!a_split_enabled()) return;
    ASplitState& A = g_a_split;
    const double now = now_seconds();
    switch (phase) {
        case 0: A.rounds.push_back(ASplitRound{}); A.rounds.back().t0 = now; break;
        case 1: if (!A.rounds.empty() && A.rounds.back().t1 < 0) A.rounds.back().t1 = now; break;
        case 2: if (!A.rounds.empty() && A.rounds.back().t2 < 0) A.rounds.back().t2 = now; break;
        case 3: if (!A.rounds.empty() && A.rounds.back().t3 < 0) A.rounds.back().t3 = now; break;
        case 4: if (!A.rounds.empty() && A.rounds.back().t4 < 0) A.rounds.back().t4 = now; break;
        case 5: if (!A.rounds.empty() && A.rounds.back().t5 < 0) A.rounds.back().t5 = now; break;
        case 6: if (!A.rounds.empty()) {
                    A.rounds.back().reused = true;
                    A.rounds.back().t3 = now;
                    A.rounds.back().t4 = now;
                } break;
        default: break;
    }
}

bool qwen35_bridge_init(const Qwen35BridgeConfig& cfg, std::string& err) {
    if (g_state != nullptr) { err = "qwen35: the bridge is already initialised"; return false; }
    if (cfg.model.empty() || cfg.pack_dir.empty()) { err = "qwen35: --model and the pack directory are required"; return false; }
    if (cfg.n_layers <= 0 || cfg.n_expert <= 0 || cfg.n_embd <= 0 || cfg.n_ff <= 0 || cfg.k <= 0) {
        err = "qwen35: the model geometry is empty";
        return false;
    }
    if (cfg.k > 16) { err = "qwen35: k > 16 is not supported by the dispatch buffers"; return false; }
    if (!kernels::cpu::expert_layout_load(cfg.pack_dir, cfg.n_layers, cfg.n_expert, cfg.n_embd, cfg.n_ff, err))
        return false;
    const auto& lay = kernels::cpu::expert_layout();
    if (!lay.native) { err = "qwen35: the pack has no native_experts.txt"; return false; }
    for (int64_t l = 0; l < (int64_t) lay.fmt.size(); ++l) {
        const auto& f = lay.fmt[(size_t) l];
        if (!kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
            err = "qwen35: layer " + std::to_string(l) + "'s expert formats have no GPU kernels (ggml types " +
                  std::to_string(f.gu_type) + "/" + std::to_string(f.d_type) + ")";
            return false;
        }
    }
    std::unique_ptr<State> S(new State());
    S->cfg = cfg;
    S->N = cfg.n_layers;
    S->NE = cfg.n_expert;
    S->H = cfg.n_embd;
    S->FF = cfg.n_ff;
    S->K = cfg.k;
    S->shared_overlap = shared_overlap_requested();
    S->layers.resize((size_t) S->N);
    S->src = std::make_unique<FileExpertSource>();
    S->src->set_gguf(cfg.model);
    if (!S->src->open(cfg.pack_dir, S->N, S->NE, err)) return false;
    S->pool = std::make_unique<kernels::cpu::ExpertPool>(cfg.pool_workers, true, true);
    S->dispatch.pool = S->pool.get();
    S->dispatch.src = S->src.get();
    S->dispatch.n_expert = S->NE;
    S->dispatch.n_embd = S->H;
    S->dispatch.n_ff = S->FF;
    S->dispatch.native_k = cfg.k;
    // The prompt path runs on the GPU when the MMQ build covers every layer's formats and the tiles fit.
    {
        namespace mmq = strata::prefill::mmq;
        S->prefill_gpu = qwen_prefill_mmq_available();
        for (int64_t l = 0; S->prefill_gpu && l < (int64_t) lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[(size_t) l];
            S->prefill_gpu = mmq::supported(f.gu_type) && mmq::supported(f.d_type) &&
                             mmq::fits(f.gu_type, 2 * f.n_ff) && mmq::fits(f.d_type, f.n_embd);
        }
    }
    if (!expert_native_cache_open(S->dispatch, cfg.cache_slots, cfg.profile, err)) return false;
    g_state = S.release();
    std::fprintf(stderr, "strata qwen35: bridge ready: %lld layers x %lld experts, %lld x %lld experts top-%lld, "
                         "workers %d, native tier %s, prompt experts %s\n", (long long) cfg.n_layers,
                 (long long) cfg.n_expert, (long long) cfg.n_embd, (long long) cfg.n_ff, (long long) cfg.k,
                 g_state->pool->workers(), g_state->dispatch.cache != nullptr ? "on" : "off",
                 g_state->prefill_gpu ? "GPU/MMQ (grouped, resident cache + bounded cold ring)"
                                      : "CPU fallback");
    if (g_state->shared_overlap)
        std::fprintf(stderr, "strata qwen35: decode shared-expert overlap prototype ON (STRATA_QWEN35_SHARED_OVERLAP=1): "
                             "one pre marker per decode layer, combine ordered after the shared expert\n");
    g_state->initialized = true;
    return true;
}

void qwen35_bridge_shutdown() {
    if (g_state == nullptr) return;
    State& S = *g_state;
    S.initialized = false;
    if (S.routing.is_open()) S.routing.close();
    if (S.routing_detail_armed) {
        // Best-effort final flush of the default-off detail log: every layer's rows were already flushed and
        // checked on the write path, so a failure here is reported on stderr only - it cannot change the return
        // code or the already-printed stats summary, and it never aborts the normal resource release below.
        S.routing_detail.flush();
        if (!S.routing_detail.good())
            std::fprintf(stderr, "strata qwen35: routing detail log final flush failed (best effort, rc unchanged)\n");
        else
            std::fprintf(stderr, "strata qwen35: routing detail log: request %u rows %llu (last request, cap 2248)\n",
                         S.routing_detail_request, (unsigned long long) S.routing_detail_rows);
    }
    for (auto& lp : S.layers) {
        if (lp == nullptr) continue;
        boundary_free(*lp);   // partial initializations included: whatever was created is released here
    }
    S.layers.clear();
    if (S.small_t_compare_dev != nullptr) { cudaFree(S.small_t_compare_dev); S.small_t_compare_dev = nullptr; }
    qwen_prefill_close(S.dispatch);
    expert_native_close(S.dispatch);
    S.dispatch.pool = nullptr;
    S.dispatch.src = nullptr;
    S.pool.reset();
    S.src->close();
    S.src.reset();
    delete g_state;
    g_state = nullptr;
}

void qwen35_bridge_stats(Qwen35BridgeStats& out) {
    out = Qwen35BridgeStats{};
    if (g_state == nullptr) return;
    // Task 133 (off by default): the request's timing windows are closed by now; flush the seam-gap account.
    if (seam_gap_enabled() && !g_seam_gap.flushed) seam_gap_flush();
    // Task 134 (off by default): flush the A-window account at the same request-tail boundary.
    if (a_split_enabled() && !g_a_split.flushed) a_split_flush();
    State& S = *g_state;
    out.decode_layers = S.decode_layers;
    out.prefill_layers = S.prefill_layers;
    out.decode_hits = (uint64_t) S.dispatch.native_hits;
    out.decode_misses = (uint64_t) S.dispatch.native_misses;
    out.prefill_entries = S.prefill_entries;
    out.jobs = S.jobs.load(std::memory_order_relaxed);
    out.expert_seconds = (double) S.seconds_us.load(std::memory_order_relaxed) / 1e6;
    out.ms_ids = S.dispatch.native_stats.ms_ids;
    out.ms_launch = S.dispatch.native_stats.ms_launch;
    out.ms_misses = S.dispatch.native_stats.ms_misses;
    out.ms_combine = S.dispatch.native_stats.ms_combine;
    out.ms_queue = S.dispatch.native_stats.ms_queue;
    out.hit_overlap = S.dispatch.native_stats.hit_overlap;
    out.hit_late = S.dispatch.native_stats.hit_late;
    out.hit_inflight = S.dispatch.native_stats.hit_inflight;
    out.all_hit = S.dispatch.native_stats.all_hit;
    out.prefill_layers_gpu = S.dispatch.native_stats.prefill_layers_gpu;
    out.prefill_resident = S.dispatch.native_stats.prefill_resident;
    out.prefill_streamed = S.dispatch.native_stats.prefill_streamed;
    out.prefill_product_calls = S.dispatch.native_stats.prefill_product_calls;
    out.prefill_stream_bytes = S.dispatch.native_stats.prefill_stream_bytes;
    out.prefill_workspace_bytes = S.dispatch.native_stats.prefill_workspace_bytes;
    out.prefill_ring_bytes = S.dispatch.native_stats.prefill_ring_bytes;
    out.prefill_loan_slots = S.dispatch.native_stats.prefill_loan_slots;
    out.prefill_loan_bytes = S.dispatch.native_stats.prefill_loan_bytes;
    out.prefill_owned_bytes = S.dispatch.native_stats.prefill_owned_bytes;
    out.prefill_loan_restored = S.dispatch.native_stats.prefill_loan_restored;
    out.prefill_loan_refill_ms = S.dispatch.native_stats.prefill_loan_refill_ms;
    out.prefill_loan_why = S.dispatch.native_stats.prefill_loan_why;
    out.prefill_gpu_ms = S.dispatch.native_stats.prefill_ms;
    out.shared_overlap_marker_layers = S.overlap_marker_layers;
    out.shared_overlap_join_layers = S.overlap_join_layers;
    if (S.dispatch.cache != nullptr) {
        out.cache_slots = (uint64_t) S.dispatch.cache->slots();
        out.cache_bytes = (uint64_t) S.dispatch.cache->bytes();
        out.cache_resident = (uint64_t) S.dispatch.cache->resident();
        // The REAL resident tensor payload (admitted pairs' own blob sizes), distinct from the arena bytes
        // that the slots' max-blob sizing and the never-filled slots inflate.
        const auto& lay = kernels::cpu::expert_layout();
        uint64_t payload = 0;
        for (int64_t l = 0; l < (int64_t) lay.fmt.size() && l < S.N; ++l)
            for (int64_t e = 0; e < S.NE; ++e)
                if (S.dispatch.cache->slot_of(l, e) >= 0) payload += (uint64_t) lay.blob_bytes(l);
        out.cache_payload_bytes = payload;
    }
}

uint64_t qwen35_native_jobs() { return g_state ? g_state->jobs.load(std::memory_order_relaxed) : 0; }
uint64_t qwen35_native_quants() { return g_state ? g_state->quants.load(std::memory_order_relaxed) : 0; }
double qwen35_native_seconds() {
    return g_state ? (double) g_state->seconds_us.load(std::memory_order_relaxed) / 1e6 : 0.0;
}

bool qwen35_native_enabled() { return g_state != nullptr && g_state->initialized; }

bool qwen35_native_layer_enabled(int layer, int64_t n_tokens) {
    (void) n_tokens;
    // Task 235: the native seam owns the pack's main layers only; the MTP draft block (layer >= 40) must run
    // on llama.cpp's own MoE path because its quants are not in the native pack.  Uninitialized state stays
    // false through qwen35_native_enabled() below.
    if (layer < 0 || layer >= 40) return false;
    // Diagnostic A/B: let llama.cpp's own graph compute the experts (the anchor against this bridge).
    if (std::getenv("STRATA_QWEN35_GGML_MOE") != nullptr) return false;
    return qwen35_native_enabled();
}

bool qwen35_shared_overlap_tensor(ggml_tensor* x) {
    // EXACTLY the marker condition `qwen35_native_moe` uses (`S.shared_overlap && x->ne[1] == 1`, with the
    // same tensor) plus the A/B bypass: the llama-side anchor fires iff the routed node was left unexpanded.
    if (g_state == nullptr || !g_state->initialized || !g_state->shared_overlap) return false;
    if (std::getenv("STRATA_QWEN35_GGML_MOE") != nullptr) return false;
    return x != nullptr && x->ne[1] == 1;
}

void qwen35_install_decode_seam(ggml_backend_sched* sched) {
    if (sched != nullptr && qwen35_native_enabled()) ggml_backend_sched_set_external_callback(sched, seam_callback, nullptr);
}

void qwen35_arm_decode_seam(bool on) {
    if (g_state != nullptr) g_state->seam_armed.store(on, std::memory_order_relaxed);
    // Task 133 (off by default): the runner arms at the request's real start; latch it for the pre0 account.
    if (on && seam_gap_enabled()) g_seam_gap.t_arm = now_seconds();
}

bool qwen35_bridge_finish_prompt() {
    if (g_state == nullptr) return true;
    std::string err;
    if (!qwen_prefill_loan_return(g_state->dispatch, err)) {
        std::fprintf(stderr, "strata qwen35: the prompt loan was not returned: %s\n", err.c_str());
        return false;
    }
    // Step probe (default off): the prompt phase is over; the post-prompt layer-0 T1 calls are counted from
    // here and the 7th produces output index 7.  `base` is the native decode_layers at this boundary (the
    // prompt-tail T1 call included - 1 for the fixed case), so the target's count is base + 6 * layers.
    if (step_probe_requested() && !g_state->step_done) {
        g_state->step_armed = true;
        g_state->step_base = g_state->decode_layers;
        std::fprintf(stderr, "strata qwen35: step probe (STRATA_QWEN35_STEP_PROBE=1): armed after finish_prompt, "
                             "base %llu\n", (unsigned long long) g_state->step_base);
    }
    // Routing-detail diagnostic (default off): the request boundary is here - after the prompt-tail T1 call
    // (already counted in decode_layers) and after the prompt loan return.  The marker records this post-prompt
    // baseline (decode_base, hits_base, misses_base) and only the layers after it are logged, so the row sums
    // are the stats deltas.  A failure is a real error.
    if (!routing_detail_begin(*g_state)) return false;
    return true;
}

void qwen35_report_decode_seam() {
    if (g_state == nullptr) return;
    Qwen35BridgeStats st;
    qwen35_bridge_stats(st);
    std::fprintf(stderr, "strata qwen35: expert service: decode layers %llu (hits %llu, misses %llu), prefill "
                         "layers %llu (%llu entries), jobs %llu, expert time %.3f s, cache %llu slots / %.2f GiB "
                         "(%llu resident)\n",
                 (unsigned long long) st.decode_layers, (unsigned long long) st.decode_hits,
                 (unsigned long long) st.decode_misses, (unsigned long long) st.prefill_layers,
                 (unsigned long long) st.prefill_entries, (unsigned long long) st.jobs, st.expert_seconds,
                 (unsigned long long) st.cache_slots, (double) st.cache_bytes / 1073741824.0,
                 (unsigned long long) st.cache_resident);
    std::fprintf(stderr, "strata qwen35: shared tier phases: ids %.1f ms, launch %.1f ms, misses %.1f ms, combine "
                         "%.1f ms, admit %.1f ms; hit/miss overlap: inflight-at-launch %lld, done-at-miss-end %lld, "
                         "late %lld, all-hit %lld\n",
                 st.ms_ids, st.ms_launch, st.ms_misses, st.ms_combine, st.ms_queue,
                 (long long) st.hit_inflight, (long long) st.hit_overlap, (long long) st.hit_late,
                 (long long) st.all_hit);
    std::fprintf(stderr, "strata qwen35: prompt experts (GPU/MMQ): layers %lld, resident %lld, cold-streamed %lld, "
                         "MMQ product launches %lld, streamed %.1f MiB, workspace %.1f MiB, ring %.1f MiB, %.1f ms\n",
                 (long long) st.prefill_layers_gpu, (long long) st.prefill_resident,
                 (long long) st.prefill_streamed, (long long) st.prefill_product_calls,
                 (double) st.prefill_stream_bytes / 1048576.0, (double) st.prefill_workspace_bytes / 1048576.0,
                 (double) st.prefill_ring_bytes / 1048576.0, st.prefill_gpu_ms);
    if (st.prefill_loan_slots > 0) {
        std::fprintf(stderr, "strata qwen35: prompt workspace loan: %lld cache slots / %.2f MiB borrowed; "
                             "%lld experts refilled in %.1f ms\n", (long long) st.prefill_loan_slots,
                     (double) st.prefill_loan_bytes / 1048576.0, (long long) st.prefill_loan_restored,
                     st.prefill_loan_refill_ms);
    } else if (st.prefill_owned_bytes > 0) {
        std::fprintf(stderr, "strata qwen35: prompt workspace: owned buffers %.2f MiB (%s)\n",
                     (double) st.prefill_owned_bytes / 1048576.0,
                     st.prefill_loan_why.empty() ? "no loan" : st.prefill_loan_why.c_str());
    }
    if (st.shared_overlap_marker_layers > 0 || st.shared_overlap_join_layers > 0) {
        std::fprintf(stderr, "strata qwen35: shared-expert overlap prototype: pre markers %llu, consumed by the "
                             "routed seam %llu (event ordering evidence only, not a speed measurement)\n",
                     (unsigned long long) st.shared_overlap_marker_layers,
                     (unsigned long long) st.shared_overlap_join_layers);
    }
    if (g_state != nullptr && (g_state->small_t_layers > 0 || g_state->small_t_rows > 0)) {
        std::fprintf(stderr, "strata qwen35: small-T native prototype (STRATA_QWEN35_SMALL_T_NATIVE=1): %llu "
                             "prompt-shaped seam(s), %llu per-token row(s); NOT counted as decode_layers or "
                             "prefill_layers (the tier's own hit/miss and phase counters do include their "
                             "dispatches)\n",
                     (unsigned long long) g_state->small_t_layers, (unsigned long long) g_state->small_t_rows);
    }
    if (g_state != nullptr && g_state->small_t_compare_layers > 0) {
        // M272 diagnostic summary: one line per compared layer is printed at compare time (see small_t_compare
        // JSON lines); this counts them and is diagnostic only, never a decode/prefill accounting.
        std::fprintf(stderr, "strata qwen35: small-T compare diagnostic (STRATA_QWEN35_SMALL_T_COMPARE=1): %llu "
                             "fixed layer(s) compared, %llu pass, %llu fail; reference = %s (diagnostic arm, "
                             "not a speed measurement)\n",
                     (unsigned long long) g_state->small_t_compare_layers,
                     (unsigned long long) g_state->small_t_compare_pass,
                     (unsigned long long) g_state->small_t_compare_fail,
                     small_t_multi_requested() ? "the original per-row native path" : "the original prompt pipeline");
    }
    if (g_state != nullptr && g_state->small_t_multi_windows > 0) {
        // Task 284 evidence: the multirow window ran; the tier's win_* counters prove what the new function did
        // (windows/entries/groups and the real GPU-hit vs CPU-miss split), not a per-row impression.
        const auto& ns = g_state->dispatch.native_stats;
        std::fprintf(stderr, "strata qwen35: small-T multirow window (STRATA_QWEN35_SMALL_T_MULTI=1): %llu "
                             "prompt-shaped window(s), %llu row(s); entries %lld, groups %lld, gpu hits %lld, "
                             "cpu misses %lld; NOT counted as decode_layers or prefill_layers\n",
                     (unsigned long long) g_state->small_t_multi_windows,
                     (unsigned long long) g_state->small_t_multi_rows,
                     (long long) ns.win_rows, (long long) ns.win_groups, (long long) ns.win_hits,
                     (long long) ns.win_misses);
    }
}

ggml_tensor* qwen35_reference_moe_marker(ggml_context*, ggml_cgraph*, ggml_tensor* source, ggml_tensor*, ggml_tensor*,
                                         int, bool) {
    return source;
}

ggml_tensor* qwen35_native_moe(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* x, ggml_tensor* ids,
                               ggml_tensor* weights, ggml_tensor* gate, ggml_tensor* up, ggml_tensor* down, int layer) {
    State* S = g_state;
    if (S == nullptr || !S->initialized || layer < 0 || layer >= (int) S->N) return nullptr;
    // weights is llama.cpp's 3D [1, n_expert_used, n_tokens] (this checkout); its per-token k weights are
    // contiguous, which is what the combine kernel and the host prefill combine index.
    if (!(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && weights->type == GGML_TYPE_F32 &&
          x->ne[0] == S->H && ids->ne[0] == S->K && weights->ne[0] == 1 && weights->ne[1] == S->K &&
          ids->ne[1] == x->ne[1] && weights->ne[2] == x->ne[1] && gate != nullptr && up != nullptr &&
          down != nullptr)) {
        std::fprintf(stderr, "strata qwen35: unexpected MoE node shapes at layer %d: x ne [%lld %lld %lld %lld] "
                             "type %d, ids ne [%lld %lld] type %d, weights ne [%lld %lld %lld] type %d (expected "
                             "n_embd %lld, k %lld)\n",
                     layer, (long long) x->ne[0], (long long) x->ne[1], (long long) x->ne[2], (long long) x->ne[3],
                     (int) x->type, (long long) ids->ne[0], (long long) ids->ne[1], (int) ids->type,
                     (long long) weights->ne[0], (long long) weights->ne[1], (long long) weights->ne[2],
                     (int) weights->type, (long long) S->H, (long long) S->K);
        std::fflush(stderr);
        std::abort();
    }
    LayerState& L = layer_state(*S, layer);
    char name[64];
    if (std::getenv("STRATA_QWEN35_TRACE") != nullptr)
        std::fprintf(stderr, "qwen35 trace: native_moe layer %d n_tok %lld\n", layer, (long long) x->ne[1]);
    // Opt-in decode overlap prototype: a compute-free pre marker right after the router nodes and before the
    // shared expert (whose nodes the caller anchors next, see qwen35moe.cpp).  The routed node below is then
    // NOT expanded here: the caller's add pulls it in on its DFS after the shared expert, giving the order
    // marker -> shared expert -> routed external -> add.  The marker records `pre`; the routed seam joins it.
    const bool overlap_decode = S->shared_overlap && x->ne[1] == 1;
    if (overlap_decode) {
        ggml_tensor* marker_srcs[1] = {x};
        ggml_tensor* marker = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, marker_srcs, 1, seam_cpu_abort, 1, &L);
        std::snprintf(name, sizeof name, "strata_moe_pre-%d", layer);
        ggml_set_name(marker, name);
        ggml_build_forward_expand(graph, marker);
    }
    if (x->ne[1] == 1 || S->prefill_gpu) {
        // Both shapes carry exactly the three device inputs; the scheduler's external callback executes them:
        // one token through the decode seam, a prompt through the shared service's GPU expert pipeline.
        ggml_tensor* srcs[3] = {x, ids, weights};
        ggml_tensor* node = ggml_custom_4d(ctx, GGML_TYPE_F32, (int64_t) S->H, x->ne[1], 1, 1, srcs, 3,
                                           seam_cpu_abort, 1, &L);
        std::snprintf(name, sizeof name, "ffn_moe_out-%d", layer);
        ggml_set_name(node, name);
        if (!overlap_decode) ggml_build_forward_expand(graph, node);
        return node;
    }
    // CPU fallback (no MMQ build/coverage): a CPU custom op.  gate/up/down are inputs only so the scheduler
    // places it on the CPU backend (the expert bytes come from the shared source, not from llama.cpp's tensors).
    ggml_tensor* srcs[6] = {x, ids, weights, gate, up, down};
    ggml_tensor* node = ggml_custom_4d(ctx, GGML_TYPE_F32, (int64_t) S->H, x->ne[1], 1, 1, srcs, 6, prefill_compute, 1, &L);
    std::snprintf(name, sizeof name, "ffn_moe_cpu-%d", layer);
    ggml_set_name(node, name);
    ggml_build_forward_expand(graph, node);
    return node;
}

}  // namespace strata::core
