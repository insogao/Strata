"""Portable CPU regression for the exact production RequestDiag emitter block.

The helper extracts the gate + RequestDiag block from the real qwen35_serve.cpp by
stable markers, compiles it against a CPU-only stub core, and validates GEN,
SCORE, cont, unavailable and gate-OFF stderr JSON.  It never loads a model, starts
the engine or touches the GPU.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

START_MARKER = "/// ---- DI01:"
END_MARKER = "~RequestDiag() { emit(); }"
CASES = ("gen", "score", "cont", "unavailable", "off")
DIAG_CASES = ("gen", "score", "cont", "unavailable")


def extract_request_diag(source_text):
    lines = source_text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(START_MARKER)]
    if len(starts) != 1:
        raise ValueError("expected exactly one RequestDiag start marker, got %d" % len(starts))
    start = starts[0]
    ends = [i for i, line in enumerate(lines) if line.strip() == END_MARKER]
    if len(ends) != 1 or ends[0] <= start:
        raise ValueError("expected exactly one RequestDiag end marker after start")
    end = ends[0]
    if "};" not in lines[end + 1]:
        raise ValueError("RequestDiag end marker is not followed by struct close")
    block = "\n".join(lines[start:end + 2]) + "\n"
    if "struct RequestDiag" not in block or "decode_diag_enabled" not in block:
        raise ValueError("extracted block lacks gate or RequestDiag")
    return {"block": block, "start_line": start + 1, "end_line": end + 2,
            "sha256": hashlib.sha256(block.encode("utf-8")).hexdigest()}


STUB = r'''#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace core {
struct Qwen35BridgeStats {
    uint64_t decode_layers = 0, decode_hits = 0, decode_misses = 0, jobs = 0;
    uint64_t prefill_layers = 0, prefill_entries = 0, prefill_stream_bytes = 0;
    int64_t prefill_layers_gpu = 0, prefill_resident = 0, prefill_streamed = 0, prefill_product_calls = 0;
    double prefill_gpu_ms = 0.0, ms_ids = 0.0, ms_launch = 0.0, ms_misses = 0.0, ms_combine = 0.0;
    double ms_queue = 0.0, expert_seconds = 0.0;
    int64_t hit_overlap = 0, hit_late = 0, hit_inflight = 0, all_hit = 0;
    uint64_t cache_slots = 0, cache_resident = 0, cache_bytes = 0, cache_payload_bytes = 0;
};

static bool g_native = true;
static bool g_cache = true;
static int g_calls = 0;

static void fill(Qwen35BridgeStats& s, int call) {
    s = Qwen35BridgeStats{};
    if (call <= 0) return;
    s.prefill_layers = 40;
    s.prefill_entries = 256;
    s.prefill_layers_gpu = 40;
    s.prefill_resident = 40;
    s.prefill_streamed = 0;
    s.prefill_product_calls = 40;
    s.prefill_stream_bytes = 1048576;
    s.prefill_gpu_ms = 1800.0;
    s.ms_ids = 10.5;
    s.ms_launch = 5.0;
    s.ms_misses = 0.0;
    s.ms_combine = 1.0;
    s.ms_queue = 0.5;
    s.cache_slots = g_cache ? 3200 : 0;
    s.cache_resident = g_cache ? 3200 : 0;
    s.cache_bytes = g_cache ? 123 : 0;
    s.cache_payload_bytes = g_cache ? 100 : 0;
    if (call >= 2) {
        s.decode_layers = 5160;
        s.decode_hits = 3000;
        s.decode_misses = 900;
        s.jobs = 5160;
        s.ms_ids = 120.25;
        s.ms_launch = 80.0;
        s.ms_misses = 900.0;
        s.ms_combine = 300.0;
        s.expert_seconds = 1.25;
        s.hit_overlap = 4000;
    }
}

static void reset(bool native, bool cache) {
    g_native = native;
    g_cache = cache;
    g_calls = 0;
}

static bool qwen35_native_enabled() { return g_native; }

static void qwen35_bridge_stats(Qwen35BridgeStats& out) {
    fill(out, g_calls);
    ++g_calls;
}
}

namespace production {
#include "request_diag.inc"
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const std::string c = argv[1];
    core::reset(true, true);
    production::RequestDiag d;
    const bool score = c == "score";
    d.begin(1, 1, score ? "score" : "gen", c == "cont", false, false,
            3817, score ? 0 : 16, 0);
    if (c != "unavailable") d.after_prefill(12.5);
    if (c == "gen" || c == "cont") d.after_decode(34.25);
    d.produced = score ? 0 : 16;
    d.finish = score ? "score" : "length";
    return 0;
}
'''


def find_vcvars():
    candidates = []
    if os.environ.get("STRATA_MSVC_VCVARS"):
        candidates.append(os.environ["STRATA_MSVC_VCVARS"])
    program_files = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    candidates.append(os.path.join(program_files, "Microsoft Visual Studio", "2022",
                                   "BuildTools", "VC", "Auxiliary", "Build", "vcvars64.bat"))
    candidates.append(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                   "Microsoft Visual Studio", "2022", "Community", "VC",
                                   "Auxiliary", "Build", "vcvars64.bat"))
    for path in candidates:
        if path and Path(path).is_file():
            return path
    compiler = shutil.which("cl.exe") or shutil.which("cl")
    if compiler:
        return compiler
    raise ValueError("no supported MSVC compiler or vcvars64.bat found")


def compile_harness(work_dir, timeout_s=120):
    work_dir = Path(work_dir)
    source = work_dir / "harness_main.cpp"
    exe = work_dir / "harness.exe"
    if not source.is_file():
        raise ValueError("harness source missing")
    compiler = find_vcvars()
    if compiler.lower().endswith("cl.exe") or compiler.lower().endswith("cl"):
        cmd = [compiler, "/nologo", "/std:c++17", "/EHsc", "/Fe:harness.exe",
               "harness_main.cpp"]
    else:
        batch = work_dir / "build.bat"
        batch.write_text('@echo off\r\ncall "%s" >nul\r\nif errorlevel 1 exit /b 1\r\n'
                         'cl /nologo /std:c++17 /EHsc /Fe:harness.exe harness_main.cpp\r\n'
                         % compiler, encoding="mbcs")
        cmd = ["cmd", "/c", "build.bat"]
    result = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True,
                            timeout=timeout_s)
    (work_dir / "compile.log").write_text(
        "command: %r\nexit: %s\nstdout:\n%s\nstderr:\n%s\n"
        % (cmd, result.returncode, result.stdout, result.stderr), encoding="utf-8")
    if result.returncode != 0 or not exe.is_file():
        raise ValueError("harness compile failed; see compile.log")
    return {"exe": str(exe), "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
            "command": cmd, "compiler": compiler}


def run_case(exe, case, work_dir, timeout_s=60):
    work_dir = Path(work_dir)
    env = {k: v for k, v in os.environ.items() if k != "STRATA_QWEN35_DECODE_DIAGNOSTICS"}
    if case != "off":
        env["STRATA_QWEN35_DECODE_DIAGNOSTICS"] = "1"
    result = subprocess.run([str(exe), case], cwd=str(work_dir), env=env,
                            capture_output=True, timeout=timeout_s)
    raw = result.stderr
    (work_dir / ("case-%s.stderr" % case)).write_bytes(raw)
    return {"case": case, "exit": result.returncode, "stderr": raw.decode("utf-8", "replace"),
            "stdout": result.stdout.decode("utf-8", "replace")}


def validate_case(case, row):
    errors = []
    text = row["stderr"]
    if row["exit"] != 0:
        errors.append("%s: exit %r" % (case, row["exit"]))
    if case == "off":
        if text.strip():
            errors.append("off: gate OFF emitted stderr")
        return errors
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        errors.append("%s: expected one stderr line, got %d" % (case, len(lines)))
        return errors
    try:
        rec = json.loads(lines[0])
    except Exception as exc:
        errors.append("%s: invalid JSON: %r" % (case, exc))
        return errors
    expected_kind = "score" if case == "score" else "gen"
    if rec.get("schema") != "strata-decode-diag-v1" or rec.get("event") != "request_end":
        errors.append("%s: schema/event mismatch" % case)
    if rec.get("kind") != expected_kind or rec.get("cont") is not (case == "cont"):
        errors.append("%s: kind/cont mismatch" % case)
    produced, max_new = rec.get("produced"), rec.get("max_new")
    if not isinstance(produced, int) or not isinstance(max_new, int) or not 0 <= produced <= max_new:
        errors.append("%s: produced %r outside 0..max_new %r" % (case, produced, max_new))
    if case == "score":
        if produced != 0 or rec.get("finish") != "score":
            errors.append("score: produced/finish mismatch")
        if rec.get("decode") != {"ran": False, "counters_available": False}:
            errors.append("score: decode must stay unavailable")
    elif case == "unavailable":
        if rec.get("prefill") != {"ran": False, "counters_available": False}:
            errors.append("unavailable: prefill must stay unavailable")
        if rec.get("decode") != {"ran": False, "counters_available": False}:
            errors.append("unavailable: decode must stay unavailable")
        if rec.get("cache") != {"slots": None, "resident": None, "bytes": None, "payload_bytes": None}:
            errors.append("unavailable: cache must be null")
    else:
        pre = rec.get("prefill", {}).get("counter_delta", {})
        dec = rec.get("decode", {}).get("counter_delta", {})
        if pre.get("prefill_layers") != 40 or pre.get("prefill_stream_bytes") != 1048576:
            errors.append("%s: prefill delta mismatch" % case)
        if dec.get("decode_layers") != 5160 or dec.get("jobs") != 5160:
            errors.append("%s: decode delta mismatch" % case)
        if dec.get("ms_ids") != 109.75:
            errors.append("%s: decode ms_ids mismatch" % case)
        if rec.get("cache", {}).get("slots") != 3200:
            errors.append("%s: cache slots mismatch" % case)
    if rec.get("native_tier") is not True:
        errors.append("%s: native_tier must be true in this stub" % case)
    return errors


def run(work_dir, compile=True):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / "request_diag.inc"
    stub = work_dir / "harness_main.cpp"
    if not source.is_file():
        raise ValueError("extracted request_diag.inc missing; run extraction first")
    if not stub.is_file():
        stub.write_text(STUB, encoding="utf-8")
    if not compile:
        return {"work_dir": str(work_dir), "compiled": False}
    built = compile_harness(work_dir)
    rows, errors = [], []
    for case in CASES:
        row = run_case(built["exe"], case, work_dir)
        rows.append({k: row[k] for k in ("case", "exit", "stderr")})
        errors.extend(validate_case(case, row))
    result = {"schema": "strata-deepinfer-emitter-regression/1",
              "compiled": True, "build": built, "cases": rows, "errors": errors,
              "pass": not errors}
    (work_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="portable actual-C++ RequestDiag regression")
    parser.add_argument("--source", default="backends/qwen35_llama/src/program/qwen35_serve.cpp")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    extracted = extract_request_diag(Path(args.source).read_text(encoding="utf-8"))
    (work_dir / "request_diag.inc").write_text(extracted["block"], encoding="utf-8")
    (work_dir / "harness_main.cpp").write_text(STUB, encoding="utf-8")
    summary = {"source": str(Path(args.source).resolve()),
               "start_line": extracted["start_line"], "end_line": extracted["end_line"],
               "block_sha256": extracted["sha256"], "work_dir": str(work_dir.resolve())}
    if args.compile:
        result = run(work_dir, compile=True)
        summary.update({"pass": result["pass"], "errors": result["errors"],
                        "build_sha256": result["build"]["sha256"]})
        print(json.dumps(summary), flush=True)
        raise SystemExit(0 if result["pass"] else 1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
