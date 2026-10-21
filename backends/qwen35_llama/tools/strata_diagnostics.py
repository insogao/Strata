"""Bounded read-only telemetry beside a leased experiment; no kernel tracing."""
import ctypes
import json
from pathlib import Path
import threading
import time


class Sampler:
    def __init__(self, native, period=.25):
        self.native, self.period = native, period
        self.ps, self.gpu = native.monitor.ps, native.monitor.gpu
        self.rows, self.errors = [], []
        self.done = threading.Event()
        self.ps.cpu_percent(percpu=True)

    def sample(self):
        gpu = self.gpu.read()
        gpu['sm_mhz'] = self.gpu._uint('nvmlDeviceGetClockInfo', ctypes.c_uint(1))
        gpu['memory_mhz'] = self.gpu._uint('nvmlDeviceGetClockInfo', ctypes.c_uint(2))
        gpu['pstate'] = self.gpu._uint('nvmlDeviceGetPerformanceState')
        reasons = ctypes.c_ulonglong()
        try:
            rc = self.gpu.lib.nvmlDeviceGetCurrentClocksThrottleReasons(self.gpu.dev, ctypes.byref(reasons))
            gpu['throttle_reasons'] = reasons.value if rc == 0 else None
        except (AttributeError, OSError):
            gpu['throttle_reasons'] = None
        cpus = self.ps.cpu_percent(percpu=True)
        freq = self.ps.cpu_freq()
        processes = []
        try:
            children = self.ps.Process(self.native.proc.pid).children(recursive=True)
            for p in children:
                try:
                    mem = p.memory_info()._asdict(); cpu = p.cpu_times()._asdict(); io = p.io_counters()._asdict()
                    processes.append({'pid': p.pid, 'name': p.name(), 'memory': mem, 'cpu': cpu, 'io': io})
                except self.ps.Error:
                    pass
        except self.ps.Error:
            pass
        self.rows.append({'at': time.monotonic(), 'gpu': gpu, 'cpu_busy_per_core': cpus,
            'cpu_mhz_proxy': freq.current if freq else None,
            'available_ram_bytes': self.ps.virtual_memory().available,
            'swap': self.ps.swap_memory()._asdict(),
            'disk': self.ps.disk_io_counters()._asdict(), 'processes': processes})

    def start(self):
        def run():
            while not self.done.is_set():
                try: self.sample()
                except Exception as exc: self.errors.append(repr(exc))
                self.done.wait(self.period)
        self.thread = threading.Thread(target=run, daemon=True); self.thread.start()

    def finish(self, path):
        self.done.set(); self.thread.join(3)
        result = {'period_s': self.period, 'scope': 'diagnostic; CPU frequency is an OS proxy, disk/pagefaults are not direct memory bandwidth',
                  'errors': self.errors, 'samples': self.rows}
        Path(path).write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result

DECODE_DIAG_SCHEMA = "strata-decode-diag-v1"
DECODE_DIAG_FINISH = {"stop", "length", "cancel", "nonfinite", "score", "error"}


def decode_diag_gate(value):
    """The C++ gate is exact: only the string '1' enables diagnostics."""
    return value == "1"


def counter_delta(now, before):
    """Cumulative counter delta; None when either side is unavailable or reset backwards."""
    if now is None or before is None:
        return None
    if isinstance(now, bool) or isinstance(before, bool):
        return None
    if now < before:
        return None
    return now - before


def validate_decode_diag_record(rec):
    """Validate one strata-decode-diag-v1 line; returns a list of contract errors."""
    errors = []

    def need(cond, message):
        if not cond:
            errors.append(message)

    need(isinstance(rec, dict), "record must be an object")
    if not isinstance(rec, dict):
        return errors
    need(rec.get("schema") == DECODE_DIAG_SCHEMA, "schema")
    need(rec.get("event") == "request_end", "event")
    for key in ("req", "serve_req", "prompt_tokens", "max_new", "base_pos", "produced"):
        value = rec.get(key)
        need(isinstance(value, int) and not isinstance(value, bool) and value >= 0, key)
    need(isinstance(rec.get("kind"), str) and rec["kind"] in ("gen", "score"), "kind")
    for key in ("cont", "session", "keep", "native_tier"):
        need(isinstance(rec.get(key), bool), key)
    need(rec.get("finish") in DECODE_DIAG_FINISH, "finish")
    for key in ("prompt_ms", "decode_ms"):
        value = rec.get(key)
        need(isinstance(value, (int, float)) and not isinstance(value, bool)
             and value == value and value not in (float("inf"), float("-inf")) and value >= 0, key)

    for phase in ("prefill", "decode"):
        block = rec.get(phase)
        need(isinstance(block, dict), phase)
        if not isinstance(block, dict):
            continue
        need(isinstance(block.get("ran"), bool), phase + ".ran")
        need(isinstance(block.get("counters_available"), bool), phase + ".counters_available")
        if block.get("ran") and block.get("counters_available"):
            counters = block.get("counter_delta")
            need(isinstance(counters, dict) and bool(counters), phase + ".counter_delta")
            if isinstance(counters, dict):
                for name, value in counters.items():
                    if value is None:
                        continue
                    need(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                         or isinstance(value, float) and value == value
                         and value not in (float("inf"), float("-inf")) and value >= 0,
                         phase + "." + name)
        else:
            need("counter_delta" not in block, phase + ".counter_delta when unavailable")

    h2d = rec.get("h2d")
    need(isinstance(h2d, dict), "h2d")
    if isinstance(h2d, dict):
        need(h2d.get("available") is False, "h2d.available")
        need(h2d.get("decode_ops") is None and h2d.get("decode_bytes") is None, "h2d null")
    legacy = rec.get("legacy_done")
    need(isinstance(legacy, dict), "legacy_done")
    if isinstance(legacy, dict):
        for key in ("hits", "lookups", "accepted", "offered"):
            need(legacy.get(key) is None, "legacy_done." + key)
    return errors
