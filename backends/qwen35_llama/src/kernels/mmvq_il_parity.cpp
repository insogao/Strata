// src/kernels/mmvq_il_parity.cpp - native_mmvq_il (2-8 columns from the interleaved q8_1 copy, fork F4) against
// native_mmvq's multi-column kernels, bitwise, for every rows-a-warp choice; --bench times both per call.
//
//     build/mmvq_il_parity [--bench [--emit-table] [--pf N]]
//
// --bench times rows 1/2/4 per (case, T); --emit-table (with --bench) prints the rows-a-warp tables this card measured, in
// the source form of kIlRows (2-4 columns) and IlRows58 (5-8 columns) in native_mmvq.cu: per cell the rows count (1/2/4,
// written 2 where 4 runs as 2) that takes at least 3% off native_mmvq's time, else 0.
//
// X4: the T loop covers 2..8 (a verify window is 2..kVerifyMaxT = 8) and STRATA_MMVQ_IL_TMAX=N (2..8) bounds it, so one run
// stays inside the GPU window it was authorised for.  --pf N puts every case through native_mmvq_il_pf(N) (2..6: the Q6_K
// hoisted-load kernel; 0: the plain one) instead of STRATA_MMVQ_IL_PF: the bit comparison is unchanged, because the parity
// gate is bitwise and never a timing rule.
//
// The same Q8_1 bytes feed both paths (quantize_q8_1_rows, then native_q8_1_interleave). Weights are random bytes with
// the block scales rewritten as normal fp16 values, so every output is finite (a NaN would compare equal whatever
// produced it). The bench cycles through enough copies of the weights to miss the L2 cache, as a layer's call does.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace strata::kernels;

struct Cell { int type, T, n_out; double ref; double il[5]; };
std::vector<Cell> g_cells;
struct Case {
    const char* name;
    int type, n_in, n_out, block_elems, block_bytes, scale_at[2];
};
// the engine's dense shapes: qkv 10240 / gate 6144 / out 2560 (GDN), q 12288 / k,v 512 / o 2560 (QSA), the head
const Case CASES[] = {
    {"IQ4_XS 2560x10240", 23, 2560, 10240, 256, 136, {0, -1}},
    {"IQ4_XS 2560x6144", 23, 2560, 6144, 256, 136, {0, -1}},
    {"IQ4_XS 4096x2560", 23, 4096, 2560, 256, 136, {0, -1}},
    {"IQ4_XS 2560x512", 23, 2560, 512, 256, 136, {0, -1}},
    {"Q4_K 2560x10240", 12, 2560, 10240, 256, 144, {0, 2}},
    {"Q4_K 4096x2560", 12, 4096, 2560, 256, 144, {0, 2}},
    {"Q4_K odd rows 2560x2051", 12, 2560, 2051, 256, 144, {0, 2}},
    {"Q5_K 2560x12288", 13, 2560, 12288, 256, 176, {0, 2}},
    {"Q5_K 2560x6144", 13, 2560, 6144, 256, 176, {0, 2}},
    {"Q5_K 4096x2560", 13, 4096, 2560, 256, 176, {0, 2}},
    {"Q5_K head 2560x248320", 13, 2560, 248320, 256, 176, {0, 2}},
    {"Q6_K 2560x12288", 14, 2560, 12288, 256, 210, {208, -1}},
    {"Q6_K 2560x10240", 14, 2560, 10240, 256, 210, {208, -1}},
    {"Q6_K 4096x2560", 14, 4096, 2560, 256, 210, {208, -1}},
    {"Q6_K odd rows 2560x2051", 14, 2560, 2051, 256, 210, {208, -1}},
    {"Q6_K 2560x512", 14, 2560, 512, 256, 210, {208, -1}},
};

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}
// X4: the widest T one run measures (STRATA_MMVQ_IL_TMAX, 2..8, default 8).  The T loop grew from 2..4 to 2..8, and a
// single --bench run has to stay inside the GPU window it was authorised for, so the bound is settable per run.
int t_max() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_MMVQ_IL_TMAX");
        const int t = e != nullptr ? std::atoi(e) : 8;
        return t < 2 ? 2 : t > 8 ? 8 : t;
    }();
    return v;
}

int run_case(const Case& c, bool bench, cudaStream_t s) {
    if (c.type == 13 && c.n_out > 100000 && !bench) { /* the head: parity too, just big */ }
    const std::size_t wbytes = native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    std::mt19937 rng(77u + (unsigned) c.type * 31u + (unsigned) c.n_out);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    if (n_blocks * (std::size_t) c.block_bytes != wbytes) {
        std::printf("%s: weight bytes mismatch\n", c.name);
        return 1;
    }
    for (std::size_t k = 0; k < n_blocks; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    const int copies = bench ? (int) std::max<std::size_t>(1, std::min<std::size_t>(24, (192u << 20) / wbytes)) : 1;
    std::vector<void*> dw(copies, nullptr);
    for (int i = 0; i < copies; ++i) {
        if (!ck(cudaMalloc(&dw[i], wbytes), "malloc w") || !ck(cudaMemcpy(dw[i], w.data(), wbytes, cudaMemcpyHostToDevice), "copy w"))
            return 1;
    }
    int bad = 0;
    for (int T = 2; T <= t_max(); ++T) {   // X4: 2..8 (STRATA_MMVQ_IL_TMAX bounds it), not 2..4
        std::vector<float> x((std::size_t) T * c.n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        float* dx = nullptr;
        void *xq = nullptr, *xil = nullptr;
        float *yref = nullptr, *yil = nullptr;
        const std::size_t ybytes = (std::size_t) T * c.n_out * 4;
        if (!ck(cudaMalloc(&dx, x.size() * 4), "malloc x") ||
            !ck(cudaMalloc(&xq, native_q8_1_bytes(c.n_in, T)), "malloc xq") ||
            !ck(cudaMalloc(&xil, native_q8_1_il_bytes(c.n_in, T)), "malloc xil") ||
            !ck(cudaMalloc((void**) &yref, ybytes), "malloc yref") || !ck(cudaMalloc((void**) &yil, ybytes), "malloc yil"))
            return 1;
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x");
        quantize_q8_1_rows(dx, T, c.n_in, xq, s);
        native_q8_1_interleave(xq, xil, c.n_in, T, s);
        native_mmvq(c.type, dw[0], xq, yref, c.n_in, c.n_out, T, s);
        ck(cudaStreamSynchronize(s), "ref");
        std::vector<uint32_t> a((std::size_t) T * c.n_out), b(a.size());
        ck(cudaMemcpy(a.data(), yref, ybytes, cudaMemcpyDeviceToHost), "read ref");
        double best_ref = 0, best_il[5] = {0, 0, 0, 0, 0};
        int table_r = 0;
        for (int r : {0, 1, 2, 4}) {
            native_mmvq_il_tune(r);
            cudaMemset(yil, 0xff, ybytes);
            native_mmvq_il(c.type, dw[0], xq, xil, yil, c.n_in, c.n_out, T, s);
            ck(cudaStreamSynchronize(s), "il");
            ck(cudaMemcpy(b.data(), yil, ybytes, cudaMemcpyDeviceToHost), "read il");
            long long diff = 0, nonfinite = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                float fa;
                std::memcpy(&fa, &a[i], 4);
                if (!std::isfinite(fa)) ++nonfinite;
                if (a[i] != b[i]) ++diff;
            }
            const bool used = r != 0 || native_mmvq_il_supported(c.type, T, c.n_out);
            if (diff != 0 || nonfinite != 0) {
                ++bad;
                std::printf("FAIL %-26s T=%d rows=%d: %lld bits differ, %lld non-finite of %zu\n", c.name, T, r, diff, nonfinite, a.size());
            }
            (void) used;
            if (bench && r != 0) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                const int reps = 200;
                for (int rep = 0; rep < 2; ++rep) {   // rep 0 warms, rep 1 times
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < reps; ++i) native_mmvq_il(c.type, dw[i % copies], xq, xil, yil, c.n_in, c.n_out, T, s);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                }
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                best_il[r] = 1000.0 * ms / reps;
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
            }
        }
        native_mmvq_il_tune(0);
        if (bench) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            const int reps = 200;
            for (int rep = 0; rep < 2; ++rep) {
                cudaEventRecord(e0, s);
                for (int i = 0; i < reps; ++i) native_mmvq(c.type, dw[i % copies], xq, yref, c.n_in, c.n_out, T, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
            }
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            best_ref = 1000.0 * ms / reps;
            (void) table_r;
            g_cells.push_back({c.type, T, c.n_out, best_ref, {0, best_il[1], best_il[2], 0, best_il[4]}});
            std::printf("BENCH %-26s T=%d  multi %.1f us | il rows1 %.1f rows2 %.1f rows4 %.1f us\n", c.name, T, best_ref,
                        best_il[1], best_il[2], best_il[4]);
        }
        cudaFree(dx); cudaFree(xq); cudaFree(xil); cudaFree(yref); cudaFree(yil);
    }
    for (void* p : dw) cudaFree(p);
    return bad;
}

// one (type, ncols, n_out class) cell: the fastest rows count under 0.97 * native_mmvq's time, 0 when no choice gets
// there, and 0 when the class's measured shapes disagree (the shipped tables' rule); a class with no measured shape takes
// `prev`, its neighbour's value below it
int pick_cell(int type, int nc, int cls, int prev) {
    int pick = -1;
    for (const Cell& c : g_cells) {
        if (c.type != type || c.T != nc) continue;
        const int cc = c.n_out < 2048 ? 0 : c.n_out < 4096 ? 1 : c.n_out < 8192 ? 2 : c.n_out < 12288 ? 3 : 4;
        if (cc != cls) continue;
        int best = 0;
        double bt = c.ref * 0.97;
        for (int r : {1, 2, 4})
            if (c.il[r] > 0 && c.il[r] < bt) { bt = c.il[r]; best = r; }
        pick = pick < 0 ? best : (pick == best ? pick : 0);   // shapes disagree: native_mmvq
    }
    return pick < 0 ? prev : pick;
}
// the tables as kIlRows (2-4 columns) and IlRows58 (5-8 columns) print them.  X4: at 5+ columns launch_il_rows maps 4 rows
// a warp to 2, so a cell measured as 4 is printed as 2 - the value the table can actually launch.
void emit_table() {
    const int types[] = {23, 12, 13, 14};
    const char* names[] = {"IQ4_XS", "Q4_K", "Q5_K", "Q6_K"};
    std::printf("// measured on this card by mmvq_il_parity --bench --emit-table (T 2-8)\n");
    std::printf("// kIlRows (2-4 columns):\n");
    for (int ti = 0; ti < 4; ++ti) {
        std::printf("    {%d, {", types[ti]);
        for (int nc = 2; nc <= 4; ++nc) {
            int prev = 0;
            std::printf("{");
            for (int cls = 0; cls < 5; ++cls) {
                prev = pick_cell(types[ti], nc, cls, prev);
                std::printf("%d%s", prev, cls < 4 ? ", " : "");
            }
            std::printf("}%s", nc < 4 ? ", " : "");
        }
        std::printf("}},   // %s\n", names[ti]);
    }
    std::printf("// IlRows58 (5-8 columns; 4 rows a warp runs as 2 there):\n");
    for (int ti = 0; ti < 4; ++ti) {
        std::printf("    {%d, {", types[ti]);
        for (int nc = 5; nc <= 8; ++nc) {
            int prev = 0;
            std::printf("{");
            for (int cls = 0; cls < 5; ++cls) {
                prev = pick_cell(types[ti], nc, cls, prev);
                if (prev == 4) prev = 2;
                std::printf("%d%s", prev, cls < 4 ? ", " : "");
            }
            std::printf("}%s", nc < 8 ? ", " : "");
        }
        std::printf("}},   // %s\n", names[ti]);
    }
}

// the per-architecture lookup, no GPU needed for its answers.
// X4: sm_89's 2-4 column table in native_mmvq.cu is a PLACEHOLDER (upstream's RTX 4070/4080 agreement - this card, a 4060
// Laptop, has not measured it).  The expectation below is that same data, kept here so that a mistyped paste into
// kIlRows_89 fails this test instead of shipping.  When the local `--bench --emit-table` run replaces kIlRows_89 and
// kIlRows58_89, replace kIl89Candidate and kIl58Candidate with that run's two blocks in the same commit: this test is the
// spec of what the tables are allowed to say, and "make the test green" is not a reason to edit it alone.
const int kIl89Candidate[4][3][5] = {
    {{0, 0, 0, 1, 1}, {0, 1, 1, 2, 2}, {4, 1, 1, 2, 2}},   // IQ4_XS
    {{0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {0, 1, 1, 1, 1}},   // Q4_K
    {{0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}, {0, 0, 2, 2, 2}},   // Q5_K
    {{0, 0, 0, 0, 0}, {1, 0, 0, 2, 1}, {2, 1, 1, 1, 1}},   // Q6_K
};
const int kIl58Candidate[4][4][5] = {   // 5/6/7/8 columns; measured on RTX 4060 Laptop 2026-11-04
    {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {1, 1, 1, 1, 1}},   // IQ4_XS
    {{0, 2, 2, 2, 2}, {0, 0, 0, 2, 2}, {0, 0, 0, 2, 2}, {0, 0, 0, 2, 2}},   // Q4_K
    {{0, 2, 2, 2, 2}, {0, 2, 2, 2, 0}, {0, 2, 2, 2, 0}, {0, 2, 2, 2, 0}},   // Q5_K
    {{2, 0, 0, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 0, 0, 2, 2}},   // Q6_K
};
int check_tables() {
    int bad = 0;
    // the two Ada switches, read the way native_mmvq.cu's cached lookups read them at their first call (this function)
    const char* e89 = std::getenv("STRATA_MMVQ_IL_89");
    const bool ada = e89 != nullptr && std::atoi(e89) != 0;
    const char* ew = std::getenv("STRATA_MMVQ_IL_WIDE");
    const bool wide = ew == nullptr || ew[0] != '0';
    const bool env_rows = std::getenv("STRATA_MMVQ_IL_ROWS") != nullptr;
    if (env_rows) std::printf("WARN table: STRATA_MMVQ_IL_ROWS is set - the cell comparisons are skipped (a measurement run)\n");
    auto type_i = [](int type) { return type == 23 ? 0 : type == 12 ? 1 : type == 13 ? 2 : 3; };
    auto nout_c = [](int n_out) { return n_out < 2048 ? 0 : n_out < 4096 ? 1 : n_out < 8192 ? 2 : n_out < 12288 ? 3 : 4; };
    for (int type : {23, 12, 13, 14})
        for (int nc = 2; nc <= 4; ++nc)
            for (int n_out : {512, 2560, 6144, 10240, 12288, 248320}) {
                const int a = native_mmvq_il_rows_for(86, type, nc, n_out);
                if (a != 0 && a != 1 && a != 2 && a != 4) { ++bad; std::printf("FAIL table: rows %d\n", a); }
                // sm_89 answers its own table when opted in (STRATA_MMVQ_IL_89=1, off by default here) and the measured
                // Ampere table otherwise; sm_120 still has no table of its own on this branch
                const int b = native_mmvq_il_rows_for(89, type, nc, n_out);
                const int want = ada ? kIl89Candidate[type_i(type)][nc - 2][nout_c(n_out)] : a;
                if (!env_rows && b != want) {
                    ++bad;
                    std::printf("FAIL table: sm_89 is %d, %s says %d at type %d ncols %d n_out %d\n", b,
                                ada ? "kIlRows_89's candidate" : "the shared table", want, type, nc, n_out);
                }
                if (!env_rows && native_mmvq_il_rows_for(120, type, nc, n_out) != a) {
                    ++bad;
                    std::printf("FAIL table: sm_120 differs from sm_86 at type %d ncols %d n_out %d\n", type, nc, n_out);
                }
            }
    // Volta has its own table (PR 1401): Q6_K at 2 columns, 2560 rows -> 1 row a warp; the Ampere table says 0 there
    if (native_mmvq_il_rows_for(70, 14, 2, 2560) != 1 || native_mmvq_il_rows_for(86, 14, 2, 2560) != 0) {
        ++bad;
        std::printf("FAIL table: the sm_70 table is not the V100's\n");
    }
    // X4: 5..8 columns.  Off for every card by default (no local table measures this width yet); opted in, sm_89 answers its
    // candidate block and every card without a 5..8 table answers 0 - never kIlRowsShared (a width nobody measured stays on
    // native_mmvq's kernels).  A 4 is allowed in a table (upstream's own 5..8 tables carry them); launch_il_rows launches it
    // as 2, and --emit-table prints it as 2.
    for (int type : {23, 12, 13, 14})
        for (int nc = 5; nc <= 8; ++nc)
            for (int n_out : {512, 2560, 6144, 10240, 12288, 248320}) {
                const int b = native_mmvq_il_rows_for(89, type, nc, n_out);
                const int want = (ada && wide) ? kIl58Candidate[type_i(type)][nc - 5][nout_c(n_out)] : 0;
                if (!env_rows && b != want) {
                    ++bad;
                    std::printf("FAIL table: sm_89 5..8 columns is %d, want %d at type %d ncols %d n_out %d\n", b, want, type, nc,
                                n_out);
                }
                if (b != 0 && b != 1 && b != 2 && b != 4) { ++bad; std::printf("FAIL table: 5..8 columns rows %d\n", b); }
                if (native_mmvq_il_rows_for(86, type, nc, n_out) != 0 || native_mmvq_il_rows_for(120, type, nc, n_out) != 0) {
                    ++bad;
                    std::printf("FAIL table: a card without a 5..8 table answered at type %d ncols %d\n", type, nc);
                }
            }
    // the old "an unknown type or ncols must be 0", with the new bounds: 1 column is not in this path and 9 is past it.
    // ncols 5 with both Ada switches off is still 0 - that is what this branch ships.
    if (native_mmvq_il_rows_for(89, 99, 2, 2560) != 0 || native_mmvq_il_rows_for(89, 12, 1, 2560) != 0 ||
        native_mmvq_il_rows_for(89, 12, 9, 2560) != 0) {
        ++bad;
        std::printf("FAIL table: an unknown type or ncols must be 0\n");
    }
    if (!env_rows && (!ada || !wide) && native_mmvq_il_rows_for(89, 12, 5, 2560) != 0) {
        ++bad;
        std::printf("FAIL table: 5..8 columns answer without STRATA_MMVQ_IL_89=1 and STRATA_MMVQ_IL_WIDE=1\n");
    }
    if (!env_rows && ada && wide && native_mmvq_il_rows_for(89, 12, 5, 2560) == 0) {
        ++bad;
        std::printf("FAIL table: 5..8 columns are opted in but answer 0 - is kIl58Arch / kIlRows58_89 wired?\n");
    }
    return bad;
}
}  // namespace

int main(int argc, char** argv) {
    bool bench = false, emit = false;
    int pf = -1;   // X4: --pf N runs the Q6_K hoisted-load kernel (native_mmvq_il_pf) through the same bit comparison
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench") bench = true;
        else if (a == "--emit-table") emit = true;
        else if (a == "--pf" && i + 1 < argc) pf = std::atoi(argv[++i]);
        else { std::printf("usage: mmvq_il_parity [--bench [--emit-table]] [--pf 0|2..6]\n"); return 2; }
    }
    emit = emit && bench;   // --emit-table prints what --bench measured
    if (pf >= 0) native_mmvq_il_pf(pf);
    cudaStream_t s;
    cudaStreamCreate(&s);
    int bad = 0;
    bad += check_tables();
    for (const Case& c : CASES) bad += run_case(c, bench, s);
    if (emit) emit_table();
    if (bad) {
        std::printf("mmvq_il_parity: FAIL\n");
    } else {
        std::printf("mmvq_il_parity: OK (every case, T 2-%d, rows 1/2/4 and the table, bitwise%s)\n", t_max(),
                    pf >= 0 ? ", --pf variant" : "");
    }
    return bad ? 1 : 0;
}
