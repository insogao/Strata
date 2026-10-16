"""Finite Windows native transport, owned exclusively by the canonical GPU lease.

The engine is the declared binary, never a Python wrapper. Named-pipe stdin
allows real STOP; original merged output stays in gpu-lock's child log.
Observation timestamps are bounded file-reader observations, NOT exact flushes.
Online run needs pywin32 (Windows-only optional experiment dependency) and psutil.
Offline planning/checking and unit tests do not import either dependency.
"""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import re


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def configuration_evidence(config, raw):
    compact=re.search(r'native expert tier \(compact\): (\d+) slots',raw)
    uniform=re.search(r'native expert tier: (\d+) slots',raw)
    match=compact or uniform
    actual={'compact':bool(compact),'slots':int(match.group(1)) if match else None,
            'startup_line':match.group(0) if match else None}
    if actual['compact']!=config['compact'] or actual['slots']!=config['slots']:
        raise ValueError('requested cache layout/slots not applied: '+str(actual))
    return actual


class ResourceMonitor:
    def __init__(self, limits):
        import psutil
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from serve.telemetry import _Nvml
        self.ps = psutil
        self.gpu = _Nvml(0)
        if not self.gpu.ok():
            raise ValueError('NVML unavailable: refusing unmonitored experiment')
        self.limits = limits
        self.samples = []
        self.violation = None
        self.done = threading.Event()
        self.proc = None

    def sample(self):
        m = self.gpu.Mem()
        if self.gpu.lib.nvmlDeviceGetMemoryInfo(self.gpu.dev, ctypes.byref(m)) != 0:
            raise ValueError('NVML memory reading failed')
        ram = self.ps.virtual_memory()
        row = {'at': time.monotonic(), 'gpu_used_mib': m.used / 2**20,
               'gpu_total_mib': m.total / 2**20, 'available_ram_mib': ram.available / 2**20,
               'rss_mib': 0}
        from strata_fingerprint import probe_ram
        row['commit_available_mib'] = probe_ram()['commit_avail_mib']['value']
        if self.proc is not None:
            try:
                children = self.ps.Process(self.proc.pid).children(recursive=True)
                row['rss_mib'] = sum(p.memory_info().rss for p in children) / 2**20
            except self.ps.Error:
                pass  # exited child is confirmed separately by the Job Object
        if row['gpu_used_mib'] > self.limits['max_gpu_used_mib']:
            self.violation = 'GPU memory budget exceeded'
        elif row['available_ram_mib'] < self.limits['min_available_ram_mib']:
            self.violation = 'host available-memory floor crossed'
        elif row['rss_mib'] > self.limits['max_host_rss_mib']:
            self.violation = 'host RSS budget exceeded'
        elif row['commit_available_mib'] is None or row['commit_available_mib'] < 1024:
            self.violation = 'host commit headroom floor crossed/unavailable'
        self.samples.append(row)
        return row

    def start(self, proc):
        self.proc = proc
        def work():
            while not self.done.is_set():
                try:
                    self.sample()
                except Exception as exc:
                    self.violation = 'resource monitor failed: ' + str(exc)
                self.done.wait(.1)
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def finish(self):
        self.done.set()
        if hasattr(self, 'thread'):
            self.thread.join(2)
        return {'period_s': .1, 'peak_kind': 'sampled whole-card high-water; not exact allocator peak',
                'gpu_peak_mib': max((s['gpu_used_mib'] for s in self.samples), default=None),
                'host_rss_peak_mib': max((s['rss_mib'] for s in self.samples), default=None),
                'available_ram_min_mib': min((s['available_ram_mib'] for s in self.samples), default=None),
                'violation': self.violation, 'samples': self.samples}


class LeasedNative:
    def __init__(self, spec, config, out, window_started):
        import pywintypes
        import win32event
        import win32file
        import win32pipe
        self.wt, self.we, self.wf, self.wp = pywintypes, win32event, win32file, win32pipe
        self.spec, self.out = spec, Path(out)
        self.out.mkdir(exist_ok=False)
        self.proc = self.pipe = None
        self.offset, self.partial, self.queue = 0, b'', []
        self.commands, self.responses = [], {}
        self.monitor = ResourceMonitor(spec['resources'])
        self.monitor.sample()
        if self.monitor.violation:
            raise ValueError(self.monitor.violation)
        left = spec['budget']['window_s'] - (time.time() - window_started)
        bound = spec['budget']['native_timeout_s']
        if left < bound + spec['budget']['wait_s']:
            raise ValueError('continuation window insufficient; no native started')
        self.job_id = 'strata-tune-' + self.out.name
        exe = str(Path(spec['engine']['path']).resolve())
        argv = [exe, '--qwen35-model', spec['model']['path'], '--qwen35-pack', spec['pack'],
                '--serve', '--max-context', str(config['ctx']), '--qwen35-threads', str(config['threads']),
                '--qwen35-cache-slots', str(config['slots']), '--qwen35-ubatch', str(config['ubatch'])]
        if spec.get('expert_profile'):
            argv += ['--qwen35-profile',spec['expert_profile']['path']]
        if not all(s.isascii() for s in argv):
            raise ValueError('engine argv must be ASCII (use relative model/pack paths)')
        job = {'authorized': True, 'job': self.job_id, 'task': 'opt-in single-GPU experiment',
               'source': 'explicit experiment manifest', 'binary': exe,
               'binary_sha256': spec['engine']['sha256'], 'commands': [argv],
               'max_launch': 1, 'runtime_limit_seconds': bound}
        save(self.out / 'job.json', job)
        env = {k: v for k, v in os.environ.items() if not
               (k.upper().startswith(('STRATA_', 'GGML_', 'LLAMA_GRAPH')) or k.upper() == 'CUDA_LAUNCH_BLOCKING')}
        env['PATH'] = spec['runtime']['cuda_bin'] + os.pathsep + env.get('PATH', '')
        env.update(spec['runtime']['env'])
        if config['compact']:
            env['STRATA_QWEN35_COMPACT_SLOTS'] = '1'
        env['STRATA_QWEN35_SESSION_ROOTS'] = str(self.out.resolve())
        pipe_name = r'\\.\pipe' + '\\' + self.job_id + '-' + str(os.getpid())
        self.pipe = win32pipe.CreateNamedPipe(pipe_name,
            win32pipe.PIPE_ACCESS_OUTBOUND | win32file.FILE_FLAG_OVERLAPPED,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT, 1, 65536, 65536, 0, None)
        ov = pywintypes.OVERLAPPED()
        ov.hEvent = win32event.CreateEvent(None, True, False, None)
        win32pipe.ConnectNamedPipe(self.pipe, ov)
        cmd = [sys.executable, '-X', 'utf8', '-B', spec['lease']['tool'], 'run',
               '--lock', spec['lease']['lock'], '--job', str(self.out / 'job.json'),
               '--audit', str(self.out / 'audit.json'), '--result', str(self.out / 'result.json'),
               '--ready-file', str(self.out / 'ready.json'), '--child-log-dir', str(self.out / 'raw'),
               '--cwd', spec['cwd'], '--wait-seconds', str(spec['budget']['wait_s']),
               '--retry-interval-seconds', '10', '--cleanup-wait-seconds', '15',
               '--expect-job', self.job_id, '--source', 'single-gpu-autotuner', '--stdin-file', pipe_name]
        save(self.out / 'preregistration.json', {'argv': cmd, 'remaining_window_s': left,
             'config': config, 'runtime_env': spec['runtime'], 'job': job})
        self.log = (self.out / 'lease.log').open('wb')
        self.started = time.monotonic()
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            end = time.monotonic() + spec['budget']['wait_s'] + 15
            while win32event.WaitForSingleObject(ov.hEvent, 100) != win32event.WAIT_OBJECT_0:
                if self.proc.poll() is not None or time.monotonic() >= end:
                    raise ValueError('lease refused/busy or pipe connect deadline; inspect lease.log')
            win32file.GetOverlappedResult(self.pipe, ov, False)
            self.monitor.start(self.proc)
            while not self.line(self.started + bound)[0].startswith('READY '):
                pass
            actual=configuration_evidence(config,(self.out/'raw/child_00.log').read_text(encoding='utf-8',errors='replace'))
            save(self.out/'actual-configuration.json',{'requested':config,'actual':actual})
            self.load_ready_s = time.monotonic() - self.started
        except BaseException:
            self.close(abort=True)
            raise
        finally:
            win32file.CloseHandle(ov.hEvent)

    def send(self, text):
        ov = self.wt.OVERLAPPED()
        ov.hEvent = self.we.CreateEvent(None, True, False, None)
        try:
            self.wf.WriteFile(self.pipe, (text + '\n').encode('ascii'), ov)
            if self.we.WaitForSingleObject(ov.hEvent, 5000) != self.we.WAIT_OBJECT_0:
                raise TimeoutError('stdin write deadline')
            self.wf.GetOverlappedResult(self.pipe, ov, False)
        finally:
            self.wf.CloseHandle(ov.hEvent)

    def line(self, deadline):
        raw = self.out / 'raw/child_00.log'
        while time.monotonic() < deadline:
            if self.monitor.violation:
                raise ValueError(self.monitor.violation)
            if self.queue:
                return self.queue.pop(0)
            if raw.exists():
                with raw.open('rb') as f:
                    f.seek(self.offset)
                    data = f.read()
                    self.offset += len(data)
                parts = (self.partial + data).split(b'\n')
                self.partial = parts.pop()
                now = time.monotonic()
                self.queue.extend((p.rstrip(b'\r').decode('utf-8', 'replace'), now) for p in parts)
                if self.queue:
                    continue
            if self.proc.poll() is not None:
                raise ValueError('native exited before protocol completion')
            time.sleep(.005)
        raise TimeoutError('bounded native response deadline')

    def ask(self, name, command, stop_after=None):
        import score_api
        if name in self.responses:
            raise ValueError('duplicate response name')
        self.commands.append({'name': name, 'command': command})
        start = time.monotonic()
        self.send(command)
        rows, tokens, stamps = [], [], []
        stopped = False
        while True:
            row, stamp = self.line(min(start + 60, self.started + self.spec['budget']['native_timeout_s']))
            if not row.startswith(('PP ', 'T ', 'DONE ', 'ERR ', 'STATE ', 'SAVED ', 'LOADED ',
                                   'CAPS ', 'LOGPROBS ', 'CANDLOGPROBS ')):
                continue
            rows.append(row)
            if row.startswith('T '):
                token = int(row.split()[1])
                if token < 0 or token > 2147483647:
                    raise ValueError('invalid generated token')
                tokens.append(token)
                stamps.append(stamp)
                if stop_after and len(tokens) >= stop_after and not stopped:
                    self.send('STOP')
                    stopped = True
            if row.startswith(('DONE ', 'ERR ', 'STATE ', 'SAVED ', 'LOADED ', 'CAPS ')):
                break
        answer = {'lines': rows, 'tokens': tokens, 'token_observation_s': stamps,
                  'started_s': start, 'finished_s': time.monotonic(), 'stop_sent': stopped}
        if row.startswith('DONE '):
            answer['done'] = score_api._parse_done(row)
            if command.startswith('GEN '):
                d = answer['done']; maximum = int(command.split()[1])
                if (d['finish'] not in {'length', 'stop', 'cancel'} or d['produced'] != len(tokens)
                        or len(tokens) > maximum or (d['finish'] == 'length' and len(tokens) != maximum)):
                    raise ValueError('invalid generation DONE/count/limit')
        self.responses[name] = answer
        save(self.out / 'transcript.json', {'commands': self.commands, 'responses': self.responses})
        return answer

    def close(self, abort=False):
        if self.pipe is not None:
            try:
                if abort:
                    self.send('STOP')
                self.send('QUIT')
            except Exception:
                pass
            self.wf.CloseHandle(self.pipe)
            self.pipe = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=25 if abort else 15)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(self.proc.pid), '/T', '/F'], capture_output=True)
                self.proc.wait(timeout=10)
            self.log.close()
        save(self.out / 'resources.json', self.monitor.finish())

    def receipt(self):
        result = json.loads((self.out / 'result.json').read_text())
        if (self.proc.returncode != 0 or not result['child_cleanup']['active_processes_zero']
                or result['marker']['retained']):
            raise ValueError('native/lease cleanup failure; fence kept, no retry')
        if self.monitor.violation:
            raise ValueError(self.monitor.violation)
        return result
