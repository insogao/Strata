#!/usr/bin/env python3
"""0-GPU unit tests for the P2-7 backend adapters, driven by two stub engines.

    python backends/qwen35_llama/tools/test_backend_adapters.py -v

One stub per protocol: tools/stub_engine.py (qwen35_llama: CAPS/SCORE/...) and
tools/stub_engine_official_q36.py (official_q36: READY/GEN/PP/T/DONE/STOP, no
CAPS/SCORE/SAVE/LOAD/STATE). No GPU, no engine, no lease: each test starts a stub
subprocess and an in-process score_api HTTP server on 127.0.0.1:0.
"""
import json
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import backend_adapters  # noqa: E402
import score_api  # noqa: E402

QWEN35_STUB = TOOLS / "stub_engine.py"
OFFICIAL_STUB = TOOLS / "stub_engine_official_q36.py"
PYTHON = sys.executable


def request(port, method, path, payload=None):
    url = "http://127.0.0.1:%d%s" % (port, path)
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8")), dict(response.headers)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = {"raw": body}
        return exc.code, parsed, dict(exc.headers)


class StubService:
    """A stub engine + score_api HTTP service, constructed the way the CLI starts one."""

    def __init__(self, backend, stub, vocab=64, timeout=30.0, drain_timeout=1.0):
        argv = [PYTHON, "-B", "-u", str(stub), "--vocab", str(vocab)]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     cwd=str(TOOLS))
        self.link = score_api.EngineLink(self.proc.stdin, self.proc.stdout, proc=self.proc)
        ready = self.link.wait_ready(15.0)
        adapter = score_api.get_adapter(backend)
        adapter.note_ready(ready)
        self.service = score_api.ScoreService(self.link, engine_path=str(stub), timeout=timeout,
                                              drain_timeout=drain_timeout, backend=adapter)
        self.httpd = score_api.start_http(self.service, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.link.close()

    def post(self, path, payload):
        return request(self.port, "POST", path, payload)

    def get(self, path):
        return request(self.port, "GET", path)


class SelectionTests(unittest.TestCase):
    def test_unknown_backend_valueerror(self):
        with self.assertRaises(ValueError) as caught:
            backend_adapters.get_adapter("native")
        self.assertIn("unknown backend", str(caught.exception))
        self.assertIn("official_q36", str(caught.exception))

    def test_config_field_and_cli_precedence(self):
        self.assertEqual(score_api.resolve_backend(None, {}).name, "qwen35_llama")
        self.assertEqual(score_api.resolve_backend(None, {"backend": "official_q36"}).name,
                         "official_q36")
        self.assertEqual(score_api.resolve_backend("qwen35_llama", {"backend": "official_q36"}).name,
                         "qwen35_llama")

    def test_unknown_config_backend_systemexit(self):
        with self.assertRaises(SystemExit) as caught:
            score_api.resolve_backend(None, {"backend": "native"})
        self.assertIn("unknown backend", str(caught.exception))

    def test_command_surfaces_differ(self):
        qwen = score_api.get_adapter("qwen35_llama")
        official = score_api.get_adapter("official_q36")
        for command in ("CAPS", "SCORE", "GEN", "SAVE", "LOAD", "STATE", "STOP", "QUIT"):
            self.assertIn(command, qwen.client_commands)
        for command in ("CAPS", "SCORE", "SAVE", "LOAD", "STATE"):
            self.assertNotIn(command, official.client_commands)
        for command in ("GEN", "STOP", "QUIT"):
            self.assertIn(command, official.client_commands)
        self.assertIs(qwen.supports_score, True)
        self.assertIs(official.supports_score, False)


class Qwen35BackendTests(unittest.TestCase):
    def setUp(self):
        self.env = StubService("qwen35_llama", QWEN35_STUB)
        self.addCleanup(self.env.close)

    def test_capabilities_engine_reported_plus_framework_fields(self):
        status, body, _ = self.env.get("/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(body["backend"], "qwen35_llama")
        self.assertEqual(body["protocol"], "strata-serve-v1")
        self.assertIs(body["logits_read"], True)
        self.assertIs(body["state_export"], False)
        self.assertIs(body["batching"], False)
        self.assertEqual(body["batching_contract"], "1")
        self.assertEqual(body["service"], "score_api")

    def test_score_and_decisions_still_green(self):
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1, 2, 3], "topk": 3})
        self.assertEqual(status, 200)
        self.assertEqual([pair["id"] for pair in body["logprobs"]], [0, 1, 2])
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "choice", "prompt_ids": [1], "candidates": [[0], [1]]})
        self.assertEqual(status, 200)
        self.assertEqual(body["kind"], "choice")


class OfficialBackendTests(unittest.TestCase):
    def setUp(self):
        self.env = StubService("official_q36", OFFICIAL_STUB)
        self.addCleanup(self.env.close)

    def test_capabilities_synthesized_not_qwen35_equivalent(self):
        # A 200 here already proves no CAPS command was sent: the official stub answers CAPS
        # with ERR, so the framework must have synthesized the table instead.
        status, body, _ = self.env.get("/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(body["backend"], "official_q36")
        self.assertEqual(body["protocol"], "official-q36-serve")
        self.assertEqual(body["capability_source"], "framework-synthesized-from-protocol-facts")
        self.assertIs(body["logits_read"], False)
        self.assertIs(body["state_export"], False)
        self.assertIs(body["batching"], False)
        self.assertEqual(body["batching_contract"], "1")
        self.assertIs(body["generation"], True)
        self.assertIs(body["continuous_requests"], True)
        self.assertIs(body["cancel"], True)
        self.assertEqual(body["context_max"], 4096)
        self.assertEqual(body["service"], "score_api")
        self.assertNotIn("mtp", body)
        self.assertNotIn("expert_formats", body)

    def test_score_501_explicit_no_generation(self):
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1, 2]})
        self.assertEqual(status, 501)
        self.assertEqual(body["error_source"], "service")
        self.assertIn("/v1/score", body["error"])
        self.assertIn("official_q36", body["error"])
        self.assertIn("SCORE", body["error"])
        self.assertNotIn("logprobs", body)
        self.assertNotIn("generated_tokens", body)

    def test_decisions_501_explicit(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "choice", "prompt_ids": [1], "candidates": [[0], [1]]})
        self.assertEqual(status, 501)
        self.assertEqual(body["error_source"], "service")
        self.assertIn("/v1/decisions", body["error"])
        self.assertIn("official_q36", body["error"])
        self.assertNotIn("candidates", body)


class OfficialStubProtocolTests(unittest.TestCase):
    """Protocol fidelity of the test-only official stub (raw pipes, no HTTP service)."""

    def setUp(self):
        argv = [PYTHON, "-B", "-u", str(OFFICIAL_STUB), "--vocab", "8"]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     cwd=str(TOOLS))
        self.link = score_api.EngineLink(self.proc.stdin, self.proc.stdout, proc=self.proc)
        self.addCleanup(self._close)
        self.assertEqual(self.link.wait_ready(15.0), "READY 4096 stop")

    def _close(self):
        if self.proc.poll() is None:
            try:
                self.link.send("QUIT")
            except score_api.ServiceError:
                pass
        self.link.close()

    def test_qwen35_commands_are_refused(self):
        for command in ("CAPS", "SCORE 1,2", "SAVE C:/x", "LOAD C:/x", "STATE"):
            self.link.send(command)
            answer = self.link.read_line(5.0)
            self.assertTrue(answer.startswith("ERR "), (command, answer))
            self.assertIn("expected", answer)

    def test_gen_stream_and_quit(self):
        self.link.send("GEN 2 1,2,3")
        self.assertEqual(self.link.read_line(5.0), "PP 3 3 1.0 100.0")
        self.assertEqual(self.link.read_line(5.0), "T 6")
        self.assertEqual(self.link.read_line(5.0), "T 7")
        done = self.link.read_line(5.0).split()
        self.assertEqual(done[0], "DONE")
        self.assertEqual(done[1], "2")
        self.assertEqual(done[2], "3")
        self.assertEqual(done[5], "length")
        self.link.send("STOP")
        self.link.send("GEN 1 0,0")
        self.assertEqual(self.link.read_line(5.0), "PP 2 2 1.0 100.0")
        self.assertEqual(self.link.read_line(5.0), "T 0")
        self.link.read_line(5.0)
        self.link.send("QUIT")
        self.assertEqual(self.proc.wait(timeout=10), 0)


class CliTests(unittest.TestCase):
    def _start_cli(self, backend, stub, vocab):
        command = [PYTHON, "-B", "-u", str(TOOLS / "score_api.py"), "--backend", backend,
                   "--start-engine", "--host", "127.0.0.1", "--port", "0",
                   "--engine", PYTHON, "--argv", str(stub), "--vocab", str(vocab), "--serve"]
        proc = subprocess.Popen(command, cwd=str(TOOLS), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        self.addCleanup(proc.stderr.close)
        self.addCleanup(proc.stdout.close)
        self.addCleanup(proc.wait, 15)
        self.addCleanup(proc.terminate)
        killer = threading.Timer(30.0, proc.kill)
        killer.start()
        try:
            ready_line = proc.stdout.readline()
            listen_line = proc.stdout.readline()
        finally:
            killer.cancel()
        self.assertIn("engine ready: READY 4096 stop", ready_line)
        self.assertIn("listening on http://127.0.0.1:", listen_line)
        return int(listen_line.rsplit(":", 1)[1].strip())

    def test_official_backend_cli_end_to_end(self):
        port = self._start_cli("official_q36", OFFICIAL_STUB, 8)
        status, caps, _ = request(port, "GET", "/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(caps["backend"], "official_q36")
        self.assertEqual(caps["context_max"], 4096)
        self.assertIs(caps["logits_read"], False)
        status, body, _ = request(port, "POST", "/v1/score", {"prompt_ids": [1]})
        self.assertEqual(status, 501)
        self.assertIn("official_q36", body["error"])

    def test_qwen35_backend_cli_end_to_end(self):
        port = self._start_cli("qwen35_llama", QWEN35_STUB, 8)
        status, caps, _ = request(port, "GET", "/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(caps["backend"], "qwen35_llama")
        self.assertIs(caps["logits_read"], True)
        status, body, _ = request(port, "POST", "/v1/score", {"prompt_ids": [1, 2], "topk": 3})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["logprobs"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
