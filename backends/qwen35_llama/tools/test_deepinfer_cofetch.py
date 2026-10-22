#!/usr/bin/env python3
"""DI03 candidate 1: compile the ACTUAL changed control block of expert_native_decode_layer against a mock
CUDA seam and check copy/event ordering, skip behaviour and error propagation.  CPU only; no model, no GPU."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src", "core", "expert_source.cpp"))

HELPER_RE = re.compile(
    r"namespace \{\n(?:// DI03 candidate 1.*?)\n\}   // namespace\n", re.S)
BLOCK_START = "    // Router ids (32 B).  With the tier OFF every entry is a miss"
BLOCK_END = "        // exists in this configuration, so its ACT fields stay absent - the cost is inside the id wait.)"

HARNESS_PREAMBLE = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef int cudaError_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorNotReady = 34;

struct MockEvent { const char* name; };
static MockEvent g_ev_ids{"ids"}, g_ev_act{"act"}, g_ev_hit_done{"hit_done"};

static std::vector<std::string> g_ops;
static bool g_fail_ids = false, g_fail_act = false;
static int g_launch_hits = 0;
static int g_act_state_notes = 0;
static bool g_launch_called = false;
static void* g_host_ids = nullptr;
static void* g_host_act = nullptr;

static void logop(const std::string& s) { g_ops.push_back(s); }
static std::string evname(cudaEvent_t e) { return e ? ((MockEvent*) e)->name : "null"; }

static cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t) {
    (void) src; (void) bytes;
    const char* buf = dst == g_host_ids ? "ids" : (dst == g_host_act ? "act" : "other");
    logop(std::string("copy:") + buf + (kind == cudaMemcpyDeviceToHost ? ":D2H" : ":H2D"));
    if (kind == cudaMemcpyDeviceToHost && dst == g_host_ids && g_fail_ids) return 1;
    if (kind == cudaMemcpyDeviceToHost && dst == g_host_act && g_fail_act) return 1;
    return cudaSuccess;
}
static cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) { logop("record:" + evname(e)); return cudaSuccess; }
static cudaError_t cudaEventSynchronize(cudaEvent_t e) { logop("sync:" + evname(e)); return cudaSuccess; }
static cudaError_t cudaEventQuery(cudaEvent_t e) { logop("query:" + evname(e)); return cudaSuccess; }
static cudaError_t cudaGetLastError() { return cudaSuccess; }

static double tier_now_ms() { return 0.0; }

enum {
    QWEN35_T1H_IDS_SUBMIT, QWEN35_T1H_IDS_WAIT_B, QWEN35_T1H_IDS_WAIT_E, QWEN35_T1H_HOT_END,
    QWEN35_T1H_ACT_SUBMIT, QWEN35_T1H_ACT_WAIT_B, QWEN35_T1H_ACT_WAIT_E,
    QWEN35_T1H_NOTE_HITS, QWEN35_T1H_NOTE_MISSES, QWEN35_T1H_NOTE_ACT_STATE
};
static void qwen35_t1h_stamp(int) {}
static void qwen35_t1h_note(int kind, long long v) {
    if (kind == QWEN35_T1H_NOTE_ACT_STATE && v == 2) ++g_act_state_notes;
}

struct ExpertDispatch {
    void* cache = nullptr;
    void* native_ev_ids = nullptr;
    void* native_ev_act = nullptr;
    void* native_ev_hit_done = nullptr;
    int64_t n_embd = 8;
    int64_t native_hits = 0;
    struct { double ms_ids = 0, ms_launch = 0; int hit_inflight = 0, hit_overlap = 0, hit_late = 0,
             all_hit = 0; } native_stats;
};

struct NativeDecodeStage {
    void* work_stream = nullptr;
    void* copy_stream = nullptr;
    int32_t* host_ids = nullptr;
    float* host_act = nullptr;
    int32_t* is_hit = nullptr;
};

static bool expert_native_launch(ExpertDispatch& d, int64_t, void*, const float*, const int32_t*, int64_t, int32_t*) {
    logop("launch");
    g_launch_called = true;
    d.native_hits += g_launch_hits;
    return true;
}
"""

HARNESS_CASE = r"""
static bool run_case(ExpertDispatch& d, NativeDecodeStage& st, int64_t K, const void* x_dev,
                     const void* ids_dev, std::string& err) {
    const int64_t layer = 0;
    cudaStream_t work = (cudaStream_t) st.work_stream, copy = (cudaStream_t) st.copy_stream;
    const double t0 = tier_now_ms();
"""

HARNESS_TAIL = r"""
    }
    return true;
}

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

static std::string joined() {
    std::string s;
    for (size_t i = 0; i < g_ops.size(); ++i) { if (i) s += ","; s += g_ops[i]; }
    return s;
}
static bool has(const char* op) { for (auto& s : g_ops) if (s == op) return true; return false; }
static int idx(const char* op) { for (size_t i = 0; i < g_ops.size(); ++i) if (g_ops[i] == op) return (int) i; return -1; }
static int count(const char* op) { int n = 0; for (auto& s : g_ops) if (s == op) ++n; return n; }

static void reset_case(ExpertDispatch& d, NativeDecodeStage& st, int32_t* ids, float* act, int32_t* hit,
                       int hits, bool cache) {
    g_ops.clear();
    g_fail_ids = false;
    g_fail_act = false;
    g_launch_hits = hits;
    g_launch_called = false;
    g_act_state_notes = 0;
    g_host_ids = ids;
    g_host_act = act;
    d = ExpertDispatch();
    d.cache = cache ? (void*) 0x1 : nullptr;
    d.native_ev_ids = &g_ev_ids;
    d.native_ev_act = &g_ev_act;
    d.native_ev_hit_done = &g_ev_hit_done;
    st = NativeDecodeStage();
    st.work_stream = (void*) 0x10;
    st.copy_stream = (void*) 0x20;
    st.host_ids = ids;
    st.host_act = act;
    st.is_hit = hit;
    for (int i = 0; i < 8; ++i) hit[i] = i < hits ? 1 : 0;
}

int main() {
    const bool on = qwen35_id_act_cofetch_enabled();
    const bool skip_probe = qwen35_skip_hit_overlap_probe_enabled();
    int32_t ids[8] = {41, 42, 43, 44, 45, 46, 47, 48};
    float act[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t hit[8] = {0};
    ExpertDispatch d;
    NativeDecodeStage st;
    std::string err;
    const void* x_dev = (const void*) 0x1000;
    const void* ids_dev = (const void*) 0x2000;
    const int64_t K = 4;

    if (skip_probe) {
        reset_case(d, st, ids, act, hit, 2, true);
        expect(run_case(d, st, K, x_dev, ids_dev, err), "S1 returns true");
        expect(idx("launch") > idx("sync:ids"), "S1 launch follows the id wait");
        expect(!has("record:hit_done") && !has("query:hit_done"),
               "S1 no hit_done record/query with the probe gate on");
        expect(d.native_stats.hit_overlap == -1 && d.native_stats.hit_late == -1 &&
               d.native_stats.hit_inflight == -1, "S1 counters are unavailable, not measured 0");
        expect(d.native_stats.hit_overlap + d.native_stats.hit_inflight != 0,
               "S1 sentinel is not a zero delta");

        reset_case(d, st, ids, act, hit, 0, false);
        expect(run_case(d, st, K, x_dev, ids_dev, err), "S2 all-miss returns true");
        expect(!has("record:hit_done") && !has("query:hit_done"), "S2 no hit_done traffic");
        expect(d.native_stats.hit_overlap == -1 && d.native_stats.hit_late == -1 &&
               d.native_stats.hit_inflight == -1, "S2 counters stay unavailable");
        std::printf("%s skip %s\n", failures ? "FAIL" : "PASS", joined().c_str());
        return failures ? 1 : 0;
    }

    if (!on) {
        reset_case(d, st, ids, act, hit, 2, true);
        expect(run_case(d, st, K, x_dev, ids_dev, err), "O1 returns true");
        expect(idx("copy:act:D2H") > idx("sync:ids"), "O1 separate activation copy after id wait");
        expect(count("copy:act:D2H") == 1, "O1 exactly one activation copy");
        expect(has("record:act") && has("sync:act"), "O1 separate activation event wait");
        expect(g_act_state_notes == 1, "O1 act_state=2 measured");
        expect(count("record:hit_done") == 1 && count("query:hit_done") == 1,
               "O1 original hit_done record and launch query preserved");

        reset_case(d, st, ids, act, hit, 0, false);
        expect(run_case(d, st, K, x_dev, ids_dev, err), "O2 all-miss returns true");
        expect(idx("copy:act:D2H") < idx("record:ids"), "O2 all-miss cofetches before wait");
        expect(count("copy:act:D2H") == 1, "O2 exactly one activation copy");
        expect(!has("record:act"), "O2 no second activation wait");

        reset_case(d, st, ids, act, hit, 4, true);
        expect(run_case(d, st, K, x_dev, ids_dev, err), "O3 all-hit returns true");
        expect(!has("copy:act:D2H"), "O3 all-hit copies no activation when off");
        expect(d.native_stats.all_hit == 1, "O3 all_hit counted");

        reset_case(d, st, ids, act, hit, 2, true);
        g_fail_ids = true;
        expect(!run_case(d, st, K, x_dev, ids_dev, err), "O4 id failure returns false");
        expect(err == "native tier: the router-id transfer failed", "O4 id error text");
        expect(!g_launch_called && !has("copy:act:D2H"), "O4 no launch and no activation copy");

        reset_case(d, st, ids, act, hit, 2, true);
        g_fail_act = true;
        expect(!run_case(d, st, K, x_dev, ids_dev, err), "O5 act failure returns false");
        expect(err == "native tier: the activation transfer failed", "O5 act error text");
        std::printf("%s off %s\n", failures ? "FAIL" : "PASS", joined().c_str());
        return failures ? 1 : 0;
    }

    reset_case(d, st, ids, act, hit, 2, true);
    expect(run_case(d, st, K, x_dev, ids_dev, err), "N1 returns true");
    expect(idx("copy:act:D2H") >= 0 && idx("copy:act:D2H") < idx("record:ids"), "N1 activation precedes waited event");
    expect(idx("copy:act:D2H") < idx("sync:ids"), "N1 activation copy precedes id wait");
    expect(count("copy:act:D2H") == 1, "N1 no double activation copy");
    expect(!has("record:act") && !has("sync:act"), "N1 no second activation wait");
    expect(g_act_state_notes == 0, "N1 no fabricated act_state");

    reset_case(d, st, ids, act, hit, 4, true);
    expect(run_case(d, st, K, x_dev, ids_dev, err), "N2 all-hit returns true");
    expect(idx("copy:act:D2H") >= 0 && idx("copy:act:D2H") < idx("record:ids"), "N2 all-hit cofetches");
    expect(count("copy:act:D2H") == 1 && !has("record:act"), "N2 one copy, no second wait");
    expect(d.native_stats.all_hit == 1, "N2 all_hit counted");

    reset_case(d, st, ids, act, hit, 0, false);
    expect(run_case(d, st, K, x_dev, ids_dev, err), "N3 all-miss returns true");
    expect(idx("copy:act:D2H") < idx("record:ids") && count("copy:act:D2H") == 1, "N3 all-miss path unchanged");

    reset_case(d, st, ids, act, hit, 2, true);
    g_fail_act = true;
    expect(!run_case(d, st, K, x_dev, ids_dev, err), "N4 act failure returns false");
    expect(err == "native tier: the activation transfer failed", "N4 act error text");
    expect(!g_launch_called && !has("record:ids"), "N4 fails before wait and launch");

    reset_case(d, st, ids, act, hit, 2, true);
    g_fail_ids = true;
    expect(!run_case(d, st, K, x_dev, ids_dev, err), "N5 id failure returns false");
    expect(err == "native tier: the router-id transfer failed", "N5 id error text");
    expect(!has("copy:act:D2H") && !g_launch_called, "N5 no activation copy after id failure");

    std::printf("%s on %s\n", failures ? "FAIL" : "PASS", joined().c_str());
    return failures ? 1 : 0;
}
"""


def extract():
    with open(SRC, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    hel = HELPER_RE.search(text)
    if not hel:
        raise SystemExit("helper block not found")
    s = text.find(BLOCK_START)
    e = text.find(BLOCK_END)
    if s < 0 or e < 0 or e < s:
        raise SystemExit("changed control block not found")
    e += len(BLOCK_END)
    block = text[s:e]
    checks = [
        'STRATA_QWEN35_ID_ACT_COFETCH' in hel.group(0),
        'STRATA_QWEN35_SKIP_HIT_OVERLAP_PROBE' in hel.group(0),
        'std::strcmp(v, "1") == 0' in hel.group(0),
        'const bool id_act_cofetch = !all_miss && qwen35_id_act_cofetch_enabled();' in block,
        'if ((all_miss || id_act_cofetch) &&' in block,
        'if (!all_miss && !id_act_cofetch) {' in block,
        block.count('cudaMemcpyAsync(st.host_act') == 2,
        block.count('cudaMemcpyAsync(st.host_ids') == 1,
        'const bool skip_hit_overlap = qwen35_skip_hit_overlap_probe_enabled();' in block,
        'if (have_hits && !skip_hit_overlap &&' in block,
        block.count('if (have_hits && !skip_hit_overlap) {') == 1,
        text.count('if (have_hits && !skip_hit_overlap) {') == 2,
    ]
    if not all(checks):
        raise SystemExit("extracted block does not match the expected DI03 shape: %r" % (checks,))
    return hel.group(0), block


def find_vcvars():
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        raise SystemExit("vswhere not found")
    out = subprocess.check_output([vswhere, "-latest", "-products", "*", "-requires",
                                   "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                   "-property", "installationPath"], text=True).strip().splitlines()
    if not out:
        raise SystemExit("VS with VC tools not found")
    vcvars = os.path.join(out[0].strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not os.path.exists(vcvars):
        raise SystemExit("vcvars64 not found")
    return vcvars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(tempfile.gettempdir(), "di03-cofetch-harness"))
    ap.add_argument("--keepdir", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    try:
        args.outdir.encode("ascii")
    except UnicodeEncodeError:
        raise SystemExit("--outdir must be ASCII for cmd.exe; pass --keepdir for a Unicode evidence copy")
    hel, block = extract()
    os.makedirs(args.outdir, exist_ok=True)
    cpp = os.path.join(args.outdir, "harness.cpp")
    with open(cpp, "w", encoding="utf-8", newline="\n") as f:
        f.write(HARNESS_PREAMBLE)
        f.write("\n")
        f.write(hel)
        f.write("\n")
        f.write(HARNESS_CASE)
        f.write(block)
        f.write("\n")
        f.write(HARNESS_TAIL)
    block_sha = hashlib.sha256(block.encode("utf-8")).hexdigest()
    print("extracted block sha256=%s lines=%d" % (block_sha, block.count("\n") + 1))
    bat = os.path.join(args.outdir, "build.bat")
    with open(bat, "w", encoding="ascii", newline="\r\n") as f:
        f.write("@echo off\n")
        f.write('call "%s" >nul 2>&1\n' % find_vcvars())
        f.write('if errorlevel 1 exit /b 901\n')
        f.write('cd /d "%s"\n' % args.outdir)
        f.write('cl /nologo /std:c++17 /EHsc /O2 /W4 harness.cpp /Fe:harness.exe >build.log 2>&1\n')
        f.write('exit /b %errorlevel%\n')
    rc = subprocess.call(["cmd.exe", "/d", "/c", bat])
    if rc != 0:
        log = os.path.join(args.outdir, "build.log")
        if os.path.exists(log):
            print(open(log, "r", errors="replace").read())
        raise SystemExit("harness compile failed rc=%d" % rc)
    exe = os.path.join(args.outdir, "harness.exe")
    for label, env_on in (("off", False), ("on", True), ("skip", None)):
        env = dict(os.environ)
        env.pop("STRATA_QWEN35_ID_ACT_COFETCH", None)
        env.pop("STRATA_QWEN35_SKIP_HIT_OVERLAP_PROBE", None)
        if env_on is True:
            env["STRATA_QWEN35_ID_ACT_COFETCH"] = "1"
        elif env_on is None:
            env["STRATA_QWEN35_SKIP_HIT_OVERLAP_PROBE"] = "1"
        p = subprocess.run([exe], capture_output=True, text=True, env=env)
        marker = p.stderr.count("cached-tier router-id/activation cofetch enabled")
        skip_marker = p.stderr.count("hit-overlap probe counters disabled")
        print("run=%s rc=%d marker=%d skip_marker=%d" % (label, p.returncode, marker, skip_marker))
        sys.stdout.write(p.stdout)
        if p.stderr:
            sys.stderr.write(p.stderr)
        if p.returncode != 0:
            raise SystemExit("harness %s failed" % label)
        if env_on is True and marker != 1:
            raise SystemExit("expected exactly one marker in the on run")
        if env_on is not True and marker != 0:
            raise SystemExit("unexpected cofetch marker in the %s run" % label)
        if env_on is None and skip_marker != 1:
            raise SystemExit("expected exactly one skip-probe marker in the skip run")
        if env_on is not None and skip_marker != 0:
            raise SystemExit("unexpected skip-probe marker in the %s run" % label)
    if args.keepdir:
        os.makedirs(args.keepdir, exist_ok=True)
        for name in ("harness.cpp", "build.log", "harness.exe"):
            p = os.path.join(args.outdir, name)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(args.keepdir, name))
    if not args.keep:
        for name in ("harness.obj", "harness.exe", "build.log"):
            p = os.path.join(args.outdir, name)
            if os.path.exists(p):
                os.remove(p)
    print("OK compile+ordering+skip+error tests passed (CPU mock, no GPU)")


if __name__ == "__main__":
    main()
