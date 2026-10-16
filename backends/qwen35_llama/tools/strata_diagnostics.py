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
