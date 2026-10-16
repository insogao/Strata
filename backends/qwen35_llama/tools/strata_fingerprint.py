#!/usr/bin/env python3
"""strata_fingerprint.py - device + model fingerprint for the P3 autotuner foundation (P3-1, 0 GPU).

    python backends/qwen35_llama/tools/strata_fingerprint.py --model <model.gguf> \
        [--context 4096] [--max-new 128] [--batch 1] [--sha <registered sha256>] [--out fp.json]

Writes one JSON document: the host (CPU/RAM), the GPU as the driver reports it, the model's
GGUF header facts and the workload the caller declares.  Every leaf is {"value", "source"}
(plus an optional "note"), and a value that cannot be obtained is null with source "unknown" -
never a guess.  Nothing here is benchmarked and no engine is run: bandwidth comes from a local
vendor-spec table (source "table"), the model file is never hashed (a registered SHA may be
passed with --sha, otherwise size+mtime stands in), and the GPU is queried read-only with
nvidia-smi.

Source vocabulary (the JSON carries this legend as "source_legend"):
    nvidia-smi  read from the NVIDIA driver CLI; read-only query
    table       this tool's local table (vendor nominal spec); no measurement
    gguf-meta   read from the GGUF header/metadata; tensor bytes are computed from declared
                shapes and the quantization block geometry (gguf_reader.py)
    os-api      an OS API (ctypes / winreg / shutil); read-only
    fs          filesystem stat (size, mtime) - the file is not read beyond its GGUF header
    ask         supplied by the caller (workload, registered SHA)
    policy      a deliberate default of this tool (documented in the plan tool/doc); not a fact
    unknown     not obtainable; deliberately not guessed
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
import datetime as _dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

SOURCE_LEGEND = {
    "nvidia-smi": "read from the NVIDIA driver's own CLI (read-only query; no compute)",
    "table": "a value in this tool's local table (vendor nominal spec); no benchmark was run",
    "gguf-meta": "read from the GGUF header/metadata; tensor bytes from declared shapes and block geometry",
    "os-api": "read from an OS API (Windows ctypes / winreg / shutil); read-only",
    "fs": "filesystem stat (size, mtime); the file is never hashed here",
    "ask": "supplied by the caller (workload, registered SHA)",
    "policy": "a deliberate default of this toolchain, documented in docs/backends/qwen35_llama_autotuner_foundation.md; not a measurement",
    "unknown": "not obtainable on this host/path; deliberately not guessed",
}

# Vendor nominal memory bandwidth for the GPUs this project has seen, in GB/s (decimal).
# Local table only, longest-key-first substring match on the nvidia-smi name.  No measurement.
TABLE_BANDWIDTH_GBPS: list[tuple[str, int]] = [
    ("RTX 4090 Laptop", 576),
    ("RTX 4080 Laptop", 432),
    ("RTX 4070 Laptop", 256),
    ("RTX 4060 Laptop", 256),
    ("RTX 4050 Laptop", 192),
    ("RTX 4090", 1008),
    ("RTX 3090", 936),
    ("RTX 3060", 360),
    ("RTX 4060", 272),
    ("RTX 3050", 224),
    ("T1000", 160),
    ("A100", 1555),
]


def field(value, source: str, note: str | None = None) -> dict:
    """One leaf: value + source (+ optional note).  None always means source "unknown"."""
    if value is None:
        source = "unknown"
    out = {"value": value, "source": source}
    if note:
        out["note"] = note
    return out


def unknown(note: str | None = None) -> dict:
    return field(None, "unknown", note)


def bandwidth_for(gpu_name: str | None) -> dict:
    if not gpu_name:
        return unknown("no GPU name to match the local table against")
    for key, gbps in TABLE_BANDWIDTH_GBPS:
        if key.lower() in gpu_name.lower():
            return field(gbps, "table", f"vendor nominal spec matched on {key!r}; GB/s decimal, no measurement")
    return unknown("this GPU model has no entry in the local bandwidth table; not extrapolated")


def _run(cmd: list[str], timeout: float = 15.0) -> tuple[int | None, str]:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout or "") + (done.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return None, ""


def parse_nvidia_smi_query(text: str) -> dict:
    """The --query-gpu=csv,noheader,nounits line -> raw fields.  Empty/garbled input -> {}."""
    for line in text.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        name, cap, total, free, driver = parts[0], parts[1], parts[2], parts[3], parts[4]
        try:
            return {"name": name, "compute_capability": cap,
                    "vram_total_mib": int(float(total)), "vram_free_mib": int(float(free)),
                    "driver_version": driver}
        except ValueError:
            continue
    return {}


def probe_gpu(index: int = 0, runner=_run) -> dict:
    """Read-only nvidia-smi probe.  Any failure marks the fields unknown - nothing is guessed."""
    rc, out = runner(["nvidia-smi",
                      f"--id={index}",
                      "--query-gpu=name,compute_cap,memory.total,memory.free,driver_version",
                      "--format=csv,noheader,nounits"])
    parsed = parse_nvidia_smi_query(out) if rc == 0 else {}
    if not parsed:
        note = "nvidia-smi not found or did not answer; this field is unknown, not estimated"
        return {"index": index,
                "name": unknown(note), "compute_capability": unknown(note),
                "vram_total_mib": unknown(note), "vram_free_mib": unknown(note),
                "driver_version": unknown(note),
                "cuda_driver_max": unknown("nvidia-smi did not answer"),
                "memory_bandwidth_gbps": unknown("no GPU name to match the local table against")}
    rc2, plain = runner(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([0-9.]+)", plain) if rc2 == 0 else None
    return {
        "index": index,
        "name": field(parsed["name"], "nvidia-smi"),
        "compute_capability": field(parsed["compute_capability"], "nvidia-smi"),
        "vram_total_mib": field(parsed["vram_total_mib"], "nvidia-smi"),
        "vram_free_mib": field(parsed["vram_free_mib"], "nvidia-smi",
                               "idle reading at fingerprint time - not a booking and not a placement result; "
                               "the running engine re-clamps at placement time (strata_plan.py models that)"),
        "driver_version": field(parsed["driver_version"], "nvidia-smi"),
        "cuda_driver_max": field(m.group(1), "nvidia-smi") if m else
                            unknown("nvidia-smi did not print a CUDA Version header"),
        "memory_bandwidth_gbps": bandwidth_for(parsed["name"]),
    }


def probe_cpu() -> dict:
    name = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        name = platform.processor() or None
    logical = os.cpu_count()
    physical = None
    phys_source = "psutil"
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical, phys_source = None, "unknown"
    isa = {}
    for label, pf in (("avx", 39), ("avx2", 40), ("avx512f", 41)):
        try:
            isa[label] = field(bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(pf)), "os-api",
                               "Windows IsProcessorFeaturePresent; on this OS a 0 is a real 'no'")
        except Exception:
            isa[label] = unknown("IsProcessorFeaturePresent unavailable")
    return {
        "model_name": field(name, "os-api" if name else "unknown",
                            "registry ProcessorNameString" if name else "not obtainable"),
        "physical_cores": field(physical, "os-api" if physical else "unknown",
                                f"via {phys_source}" if physical else "psutil unavailable; not inferred from logical count"),
        "logical_cores": field(logical, "os-api"),
        "isa": isa,
    }


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def probe_ram() -> dict:
    m = _MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return {"total_mib": unknown("GlobalMemoryStatusEx failed"),
                "avail_mib": unknown("GlobalMemoryStatusEx failed"),
                "commit_limit_mib": unknown("GlobalMemoryStatusEx failed"),
                "commit_avail_mib": unknown("GlobalMemoryStatusEx failed")}
    mib = 1 << 20
    return {
        "total_mib": field(round(m.ullTotalPhys / mib, 1), "os-api"),
        "avail_mib": field(round(m.ullAvailPhys / mib, 1), "os-api",
                           "instantaneous; the planner checks budgets and does not rely on this staying free"),
        "commit_limit_mib": field(round(m.ullTotalPageFile / mib, 1), "os-api",
                                  "commit limit = physical + pagefile; the bound a pageable host arena lives against"),
        "commit_avail_mib": field(round(m.ullAvailPageFile / mib, 1), "os-api"),
    }


_DRIVE_KIND = {0: "unknown", 1: "no-root", 2: "removable", 3: "fixed", 4: "remote", 5: "cdrom", 6: "ramdisk"}


def probe_disk(path: str) -> dict:
    try:
        root = os.path.splitdrive(os.path.abspath(path))[0] + os.sep
        kind = _DRIVE_KIND.get(ctypes.windll.kernel32.GetDriveTypeW(wintypes.LPCWSTR(root)), "unknown")
        usage = shutil.disk_usage(root)
        return {
            "volume": field(root, "os-api"),
            "kind": field(kind, "os-api", "GetDriveTypeW; rotational media is not probed"),
            "free_mib": field(round(usage.free / (1 << 20), 1), "os-api"),
            "total_mib": field(round(usage.total / (1 << 20), 1), "os-api"),
        }
    except Exception as exc:
        return {"volume": field(os.path.splitdrive(os.path.abspath(path))[0] or None, "os-api"),
                "kind": unknown(f"best effort failed: {exc}"),
                "free_mib": unknown(f"best effort failed: {exc}"),
                "total_mib": unknown(f"best effort failed: {exc}")}


_EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
_LAYER_RE = re.compile(r"^blk\.(\d+)\.")


def summarize_metadata(meta: dict, tensors: list) -> dict:
    """Pure GGUF summary: metadata dict + TensorInfo-like objects (name, type_name, expected_bytes)."""
    key = "qwen35moe."

    def num(name):
        v = meta.get(key + name)
        return v if isinstance(v, int) else None

    expert_count = num("expert_count")
    expert_used = num("expert_used_count")
    routed_total, routed_layers, role_bytes, role_types = 0, {}, {}, {}
    nextn_bytes, total_bytes = 0, 0
    for t in tensors:
        b = t.expected_bytes()
        if b is None:
            b = 0
        total_bytes += b
        if ".nextn." in t.name:
            nextn_bytes += b
        m = _EXPERT_RE.match(t.name)
        if m:
            layer, role = int(m.group(1)), "ffn_%s_exps" % m.group(2)
            routed_total += b
            routed_layers[layer] = routed_layers.get(layer, 0) + b
            role_bytes[role] = role_bytes.get(role, 0) + b
            role_types.setdefault(role, set()).add(t.type_name)
        elif _LAYER_RE.match(t.name):
            role = _LAYER_RE.sub("blk.N.", t.name)
            role_bytes[role] = role_bytes.get(role, 0) + b
            role_types.setdefault(role, set()).add(t.type_name)
        else:
            role_bytes[t.name] = role_bytes.get(t.name, 0) + b
            role_types.setdefault(t.name, set()).add(t.type_name)
    nextn_meta = num("nextn_predict_layers")
    layers = sorted(routed_layers)
    seq = [routed_layers[i] for i in range(layers[-1] + 1)] if layers else []
    return {
        "architecture": field(meta.get("general.architecture"), "gguf-meta"),
        "block_count": field(num("block_count"), "gguf-meta"),
        "context_length": field(num("context_length"), "gguf-meta"),
        "embedding_length": field(num("embedding_length"), "gguf-meta"),
        "expert_count": field(expert_count, "gguf-meta"),
        "expert_used_count": field(expert_used, "gguf-meta"),
        "expert_feed_forward_length": field(num("expert_feed_forward_length"), "gguf-meta"),
        "full_attention_interval": field(num("full_attention_interval"), "gguf-meta"),
        "attention": {
            "head_count": field(num("attention.head_count"), "gguf-meta"),
            "head_count_kv": field(num("attention.head_count_kv"), "gguf-meta"),
            "key_length": field(num("attention.key_length"), "gguf-meta"),
            "value_length": field(num("attention.value_length"), "gguf-meta"),
        },
        "ssm": {
            "conv_kernel": field(num("ssm.conv_kernel"), "gguf-meta"),
            "state_size": field(num("ssm.state_size"), "gguf-meta"),
            "group_count": field(num("ssm.group_count"), "gguf-meta"),
            "time_step_rank": field(num("ssm.time_step_rank"), "gguf-meta"),
            "inner_size": field(num("ssm.inner_size"), "gguf-meta"),
        },
        "nextn_predict_layers": field(nextn_meta if nextn_meta is not None else 0, "gguf-meta",
                                      "absent key reads as 0" if nextn_meta is None else None),
        "has_mtp_block": field(bool((nextn_meta or 0) > 0 or nextn_bytes > 0), "gguf-meta"),
        "nextn_tensor_bytes": field(nextn_bytes, "gguf-meta"),
        "tensor_count": field(len(tensors), "gguf-meta"),
        "tensor_bytes_total": field(total_bytes, "gguf-meta",
                                    "sum of declared shapes x block geometry; alignment padding is not included"),
        "routed_expert_bytes_total": field(routed_total, "gguf-meta"),
        "non_expert_bytes_total": field(total_bytes - routed_total, "gguf-meta",
                                        "everything that must be resident: attention/GDN/shared-expert/embed/head/nextn"),
        "routed_expert_layers": field(len(layers), "gguf-meta"),
        "routed_expert_layer_bytes": field(seq, "gguf-meta", "one entry per layer index; 0 means no routed experts that layer"),
        "role_quant": {r: field(sorted(ts), "gguf-meta") for r, ts in sorted(role_types.items())},
        "role_bytes": {r: field(b, "gguf-meta") for r, b in sorted(role_bytes.items())},
    }


def summarize_gguf(path: str) -> dict:
    try:
        import gguf_reader
    except Exception:
        return {"error": field(f"gguf_reader.py not importable from {TOOLS}", "unknown")}
    try:
        g = gguf_reader.GGUFFile(path)
    except Exception as exc:
        return {"error": field(f"GGUF header unreadable ({exc})", "unknown")}
    return summarize_metadata(g.metadata, g.tensors)


def probe_model(path: str, sha: str | None) -> dict:
    section = {"path": field(path, "ask")}
    try:
        st = os.stat(path)
        section["size_bytes"] = field(st.st_size, "fs")
        section["mtime_utc"] = field(_dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).isoformat(), "fs")
    except OSError as exc:
        section["size_bytes"] = unknown(f"stat failed: {exc}")
        section["mtime_utc"] = unknown(f"stat failed: {exc}")
    section["sha256"] = (field(sha.lower(), "ask", "registered value supplied by the caller; this tool never hashes the file")
                         if sha else unknown("no registered SHA passed; size+mtime stand in (full-file hashing is out of policy)"))
    section["gguf"] = summarize_gguf(path)
    return section


def build_fingerprint(model_path: str, workload: dict, sha: str | None, gpu_index: int = 0,
                      gpu_probe=probe_gpu, cpu_probe=probe_cpu, ram_probe=probe_ram,
                      disk_probe=probe_disk, model_probe=probe_model) -> dict:
    workload_out = {
        "model_path": field(model_path, "ask"),
        "context": field(workload.get("context"), "ask" if workload.get("context") is not None else "unknown"),
        "max_new": field(workload.get("max_new"), "ask" if workload.get("max_new") is not None else "unknown"),
        "batch": field(workload.get("batch", 1), "ask",
                       "the serve protocol's batching contract v1 is batch=1 (P2-7)"),
    }
    return {
        "schema": "strata-fingerprint/1",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source_legend": SOURCE_LEGEND,
        "host": {
            "os": field(platform.platform(), "os-api"),
            "python": field(platform.python_version(), "os-api"),
        },
        "gpu": gpu_probe(index=gpu_index),
        "cpu": cpu_probe(),
        "ram": ram_probe(),
        "disk": disk_probe(model_path) if model_path else {"volume": unknown("no model path given")},
        "model": model_probe(model_path, sha) if model_path else {"path": unknown("no model path given")},
        "workload": workload_out,
    }


def walk_sources(node, path="") -> list[tuple[str, dict]]:
    """Every {"value": ...} leaf found in a fingerprint document, with its dotted path."""
    out = []
    if isinstance(node, dict):
        if "value" in node and "source" in node:
            out.append((path, node))
            return out
        for k, v in node.items():
            if k == "source_legend":
                continue
            out += walk_sources(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += walk_sources(v, f"{path}[{i}]")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the GGUF file to fingerprint (header only; never hashed)")
    ap.add_argument("--context", type=int, help="workload context (tokens) declared by the caller")
    ap.add_argument("--max-new", type=int, dest="max_new", help="workload max_new (tokens)")
    ap.add_argument("--batch", type=int, default=1, help="workload batch (contract v1: 1)")
    ap.add_argument("--sha", help="a registered sha256 for the model; recorded as source=ask, never recomputed")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--out", help="write the JSON here (default: stdout)")
    a = ap.parse_args()
    if not os.path.isfile(a.model):
        sys.exit(f"strata-fingerprint: no such model file: {a.model}")
    doc = build_fingerprint(a.model, {"context": a.context, "max_new": a.max_new, "batch": a.batch},
                            a.sha, a.gpu_index)
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
