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


SCORE_501_TEXT = (
    "/v1/score is not implemented on the official_q36 backend: its protocol (GEN/STOP/QUIT) has "
    "no SCORE command, so next-token logprobs cannot be read. This service refuses explicitly "
    "instead of substituting another operation (it never degrades /v1/score into generation). "
    "Use the qwen35_llama backend for /v1/score.")
DECISIONS_501_TEXT = (
    "/v1/decisions is not implemented on the official_q36 backend: its protocol (GEN/STOP/QUIT) "
    "has no SCORE command, so next-token logprobs cannot be read. This service refuses explicitly "
    "instead of substituting another operation (it never degrades /v1/decisions into generation). "
    "Use the qwen35_llama backend for /v1/decisions.")
CONTINUE_501_TEXT = (
    "/v1/continue is not implemented on the official_q36 backend: its protocol (GEN/STOP/QUIT) "
    "has no session commands (no STATE, no 'GEN ... cont=1'), so a session can be neither "
    "continued nor reported. This service refuses explicitly instead of substituting another "
    "operation. Use the qwen35_llama backend for /v1/continue.")
SESSION_501_TEXT = (
    "/v1/session is not implemented on the official_q36 backend: its protocol (GEN/STOP/QUIT) "
    "has no session commands (no STATE, no 'GEN ... cont=1'), so a session can be neither "
    "continued nor reported. This service refuses explicitly instead of substituting another "
    "operation. Use the qwen35_llama backend for /v1/session.")


class RoutingMatrixTests(unittest.TestCase):
    """P2R-5: the declarative (endpoint, backend) matrix and its query surface (no stubs)."""

    def test_matrix_covers_every_endpoint_and_backend(self):
        expected = {(endpoint, backend) for endpoint in backend_adapters.ENDPOINTS
                    for backend in backend_adapters.SUPPORTED_BACKENDS}
        self.assertEqual(set(backend_adapters.ROUTES), expected)
        self.assertEqual(len(backend_adapters.ROUTES), 2 * len(backend_adapters.ENDPOINTS))
        for cell in backend_adapters.ROUTES.values():
            for key in backend_adapters.ROUTE_CELL_KEYS:
                self.assertIn(key, cell)

    def test_every_cell_is_consistent(self):
        # Cells that legitimately carry no engine command: the framework-only /v1/routes and the
        # synthesized official_q36 capabilities table (no CAPS handshake).
        without_engine_command = {
            (backend_adapters.ROUTES_ENDPOINT, "qwen35_llama"),
            (backend_adapters.ROUTES_ENDPOINT, "official_q36"),
            (backend_adapters.CAPABILITIES_ENDPOINT, "official_q36"),
        }
        for (endpoint, backend), cell in backend_adapters.ROUTES.items():
            with self.subTest(endpoint=endpoint, backend=backend):
                self.assertIsInstance(cell["supported"], bool)
                self.assertIsInstance(cell["in_service"], bool)
                self.assertIsInstance(cell["notes"], str)
                self.assertTrue(cell["notes"])
                if cell["supported"]:
                    self.assertIsNone(cell["refusal_status"])
                    self.assertIsNone(cell["refusal_text_template"])
                    if (endpoint, backend) not in without_engine_command:
                        self.assertIsInstance(cell["engine_command"], str)
                        self.assertTrue(cell["engine_command"])
                elif cell["in_service"]:
                    # In-service refusals are 501s with a renderable text.
                    self.assertEqual(cell["refusal_status"], 501)
                    self.assertIsInstance(cell["refusal_text_template"], str)
                    self.assertTrue(cell["refusal_text_template"])
                else:
                    # Boundary rows (chat, engine SAVE/LOAD): facts, not HTTP refusals.
                    self.assertIsNone(cell["refusal_status"])
                    self.assertIsNone(cell["refusal_text_template"])

    def test_unknown_endpoint_and_backend_refused(self):
        with self.assertRaises(ValueError) as caught:
            backend_adapters.route("GET /v1/nope", "qwen35_llama")
        self.assertIn("unknown endpoint", str(caught.exception))
        self.assertIn("GET /v1/nope", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            backend_adapters.route(backend_adapters.SCORE_ENDPOINT, "native")
        self.assertIn("unknown backend", str(caught.exception))
        # refusal_text() only exists for refused cells.
        with self.assertRaises(ValueError) as caught:
            backend_adapters.refusal_text(backend_adapters.SCORE_ENDPOINT, "qwen35_llama")
        self.assertIn("not refused", str(caught.exception))

    def test_route_returns_a_readonly_copy(self):
        cell = backend_adapters.route(backend_adapters.SCORE_ENDPOINT, "qwen35_llama")
        cell["supported"] = False
        self.assertIs(backend_adapters.ROUTES[(backend_adapters.SCORE_ENDPOINT, "qwen35_llama")]["supported"],
                      True)

    def test_official_refusal_texts_byte_identical_to_pre_matrix(self):
        expected = {
            backend_adapters.SCORE_ENDPOINT: SCORE_501_TEXT,
            backend_adapters.SCORE_CANDIDATES_ENDPOINT: SCORE_501_TEXT,
            backend_adapters.SCORE_KEEP_ENDPOINT: SCORE_501_TEXT,
            backend_adapters.DECISIONS_ENDPOINT: DECISIONS_501_TEXT,
            backend_adapters.CONTINUE_ENDPOINT: CONTINUE_501_TEXT,
            backend_adapters.SESSION_ENDPOINT: SESSION_501_TEXT,
        }
        for endpoint, text in expected.items():
            with self.subTest(endpoint=endpoint):
                self.assertEqual(backend_adapters.refusal_text(endpoint, "official_q36"), text)

    def test_qwen35_supported_cells_and_engine_commands(self):
        expected = {
            backend_adapters.SCORE_ENDPOINT: ("SCORE", None),
            backend_adapters.SCORE_CANDIDATES_ENDPOINT: ("cand=", "score_explicit_candidate_ids"),
            backend_adapters.SCORE_KEEP_ENDPOINT: ("keep=1", "state_keep_score"),
            backend_adapters.DECISIONS_ENDPOINT: ("cand=", "score_explicit_candidate_ids"),
            backend_adapters.CONTINUE_ENDPOINT: ("cont=1", "session_continue"),
            backend_adapters.SESSION_ENDPOINT: ("STATE", None),
            backend_adapters.CAPABILITIES_ENDPOINT: ("CAPS", None),
            backend_adapters.ROUTES_ENDPOINT: (None, None),
        }
        for endpoint, (command_token, requires) in expected.items():
            with self.subTest(endpoint=endpoint):
                cell = backend_adapters.route(endpoint, "qwen35_llama")
                self.assertIs(cell["supported"], True)
                self.assertEqual(cell["requires"], requires)
                if command_token is None:
                    self.assertIsNone(cell["engine_command"])
                else:
                    self.assertIn(command_token, cell["engine_command"])

    def test_official_unsupported_cells_and_boundary_notes(self):
        for endpoint in (backend_adapters.SCORE_ENDPOINT, backend_adapters.SCORE_CANDIDATES_ENDPOINT,
                         backend_adapters.SCORE_KEEP_ENDPOINT, backend_adapters.DECISIONS_ENDPOINT,
                         backend_adapters.CONTINUE_ENDPOINT, backend_adapters.SESSION_ENDPOINT):
            with self.subTest(endpoint=endpoint):
                cell = backend_adapters.route(endpoint, "official_q36")
                self.assertIs(cell["supported"], False)
                self.assertIs(cell["in_service"], True)
                self.assertEqual(cell["refusal_status"], 501)
        state_io = backend_adapters.route(backend_adapters.ENGINE_STATE_ENDPOINT, "official_q36")
        self.assertIs(state_io["supported"], False)
        self.assertIn("RESTORE/SERR", state_io["notes"])
        chat = backend_adapters.route(backend_adapters.CHAT_ENDPOINT, "qwen35_llama")
        self.assertIs(chat["supported"], False)
        self.assertIs(chat["in_service"], False)
        self.assertIn("serve/server.py", chat["notes"])
        self.assertIn("404", chat["notes"])

    def test_capabilities_for_difference_items(self):
        qwen = backend_adapters.capabilities_for("qwen35_llama")
        official = backend_adapters.capabilities_for("official_q36")
        for key in ("score", "score_explicit_candidate_ids", "state_keep_score", "session_continue",
                    "STATE", "state_export"):
            with self.subTest(key=key):
                self.assertIs(qwen[key], True)
                self.assertIs(official[key], False)
        self.assertIs(qwen["caps_handshake"], True)
        self.assertIs(official["caps_handshake"], False)
        self.assertEqual(qwen["protocol"], "strata-serve-v1")
        self.assertEqual(official["protocol"], "official-q36-serve")
        self.assertEqual(len(qwen["routes"]), len(backend_adapters.ENDPOINTS))
        self.assertEqual(qwen["routes"][backend_adapters.CONTINUE_ENDPOINT], True)
        self.assertEqual(official["routes"][backend_adapters.CONTINUE_ENDPOINT], False)

    def test_routes_payload_is_full_matrix_and_serializable(self):
        payload = backend_adapters.routes_payload()
        self.assertEqual(len(payload), 2 * len(backend_adapters.ENDPOINTS))
        json.dumps(payload)  # must be JSON-serializable for GET /v1/routes
        by_pair = {(item["endpoint"], item["backend"]): item for item in payload}
        self.assertEqual(set(by_pair), set(backend_adapters.ROUTES))
        for (endpoint, backend), item in by_pair.items():
            with self.subTest(endpoint=endpoint, backend=backend):
                cell = backend_adapters.ROUTES[(endpoint, backend)]
                self.assertEqual(item["supported"], cell["supported"])
                self.assertEqual(item["notes"], cell["notes"])
                if cell["refusal_text_template"] is not None:
                    self.assertEqual(item["refusal_text"],
                                     backend_adapters.refusal_text(endpoint, backend))
                else:
                    self.assertIsNone(item["refusal_text"])


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
