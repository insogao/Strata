#!/usr/bin/env python3
"""0-GPU unit tests for score_api.py, driven by the test-only stub engine.

    python backends/qwen35_llama/tools/test_score_api.py -v

No GPU, no engine, no lease: each test starts tools/stub_engine.py (a subprocess that
speaks the serve line protocol) and an in-process score_api HTTP server on 127.0.0.1:0.
"""
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import score_api  # noqa: E402

STUB = TOOLS / "stub_engine.py"
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


def raw_post(port, path, data):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class StubService:
    def __init__(self, vocab=64, drop_scores=0, score_delay_ms=0, timeout=30.0,
                 drain_timeout=1.0, tokenizer_model=None, cand_mode=None, backend=None, stub=None):
        stub = stub or STUB
        argv = [PYTHON, "-B", "-u", str(stub), "--vocab", str(vocab)]
        if drop_scores:
            argv += ["--drop-scores", str(drop_scores)]
        if score_delay_ms:
            argv += ["--score-delay-ms", str(score_delay_ms)]
        if cand_mode:
            argv += ["--cand-mode", cand_mode]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=str(TOOLS))
        self.link = score_api.EngineLink(self.proc.stdin, self.proc.stdout, proc=self.proc)
        ready = self.link.wait_ready(15.0)
        adapter = None
        if backend is not None:
            adapter = score_api.get_adapter(backend)
            adapter.note_ready(ready)
        self.service = score_api.ScoreService(self.link, engine_path=str(stub), timeout=timeout,
                                              drain_timeout=drain_timeout, tokenizer_model=tokenizer_model,
                                              backend=adapter)
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


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.env = StubService()
        self.addCleanup(self.env.close)

    def test_score_normal(self):
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1, 2, 3], "topk": 3})
        self.assertEqual(status, 200)
        self.assertEqual(body["generated_tokens"], 0)
        self.assertIs(body["full_vocab_softmax"], True)
        self.assertEqual(body["pos"], 2)
        self.assertEqual([pair["id"] for pair in body["logprobs"]], [0, 1, 2])
        logprobs = [pair["logprob"] for pair in body["logprobs"]]
        self.assertEqual(logprobs, sorted(logprobs, reverse=True))
        for logprob in logprobs:
            self.assertLessEqual(logprob, 0.0)
        self.assertEqual(body["usage"]["prompt_tokens"], 3)
        self.assertEqual(body["usage"]["prompt_read"], 3)

    def test_score_topk_default_and_override(self):
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [5]})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["logprobs"]), score_api.DEFAULT_TOPK)
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [5], "topk": 2})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["logprobs"]), 2)

    def test_capabilities_passthrough(self):
        status, body, _ = self.env.get("/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(body["service"], "score_api")
        self.assertEqual(body["protocol"], "strata-serve-v1")
        self.assertEqual(body["context_max"], 4096)
        self.assertIs(body["batching"], False)
        self.assertIs(body["logits_read"], True)
        self.assertEqual(body["mtp"], {"available": False, "enabled": False, "default": False, "k": None})
        self.assertEqual(body["engine_sha256"], hashlib.sha256(STUB.read_bytes()).hexdigest())

    def test_engine_err_maps_422_with_text(self):
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [999999]})
        self.assertEqual(status, 422)
        self.assertEqual(body["error_source"], "engine")
        self.assertIn("a token id is outside the vocabulary", body["error"])

    def test_prompt_string_refused_without_tokenizer(self):
        status, body, _ = self.env.post("/v1/score", {"prompt": "hello"})
        self.assertEqual(status, 501)
        self.assertIn("tokenizer", body["error"])
        self.assertIn("prompt_ids", body["error"])

    def test_prompt_string_refused_when_tokenizer_unloadable(self):
        env = StubService(tokenizer_model=str(TOOLS / "_no_such_tokenizer.gguf"))
        self.addCleanup(env.close)
        status, body, _ = env.post("/v1/score", {"prompt": "hello"})
        self.assertEqual(status, 501)
        self.assertIn("tokenizer", body["error"])

    def test_bad_bodies_400(self):
        status, body = raw_post(self.env.port, "/v1/score", b"{oops")
        self.assertEqual(status, 400)
        self.assertIn("JSON", body["error"])
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1], "prompt": "x"})
        self.assertEqual(status, 400)
        self.assertIn("exactly one", body["error"])
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1], "temperature": 0.5})
        self.assertEqual(status, 400)
        self.assertIn("unknown field", body["error"])
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": []})
        self.assertEqual(status, 400)
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [True]})
        self.assertEqual(status, 400)
        self.assertIn("integer", body["error"])

    def test_unknown_path_404(self):
        status, body, _ = self.env.get("/v1/chat/completions")
        self.assertEqual(status, 404)
        self.assertIn("not found", body["error"])


class CandidateScoreTests(unittest.TestCase):
    """P2R-1: the optional explicit `candidates` field of /v1/score."""

    def setUp(self):
        self.env = StubService()
        self.addCleanup(self.env.close)

    def test_candidates_order_preserved_and_same_row(self):
        status, body, _ = self.env.post("/v1/score",
                                        {"prompt_ids": [1, 2], "topk": 64, "candidates": [5, 1, 3]})
        self.assertEqual(status, 200)
        self.assertEqual([candidate["id"] for candidate in body["candidates"]], [5, 1, 3])
        top = {pair["id"]: pair["logprob"] for pair in body["logprobs"]}
        for candidate in body["candidates"]:
            # the same row and the same full-vocab lse as LOGPROBS: the values are identical
            self.assertEqual(candidate["logprob"], top[candidate["id"]])

    def test_candidate_outside_topk_still_returned(self):
        status, body, _ = self.env.post("/v1/score",
                                        {"prompt_ids": [1], "topk": 2, "candidates": [50]})
        self.assertEqual(status, 200)
        self.assertEqual([candidate["id"] for candidate in body["candidates"]], [50])
        self.assertLess(body["candidates"][0]["logprob"], body["logprobs"][-1]["logprob"])

    def test_topk0_with_candidates_answers_only_candidates(self):
        status, body, _ = self.env.post("/v1/score",
                                        {"prompt_ids": [3], "topk": 0, "candidates": [2, 4]})
        self.assertEqual(status, 200)
        self.assertEqual(body["logprobs"], [])
        self.assertEqual([candidate["id"] for candidate in body["candidates"]], [2, 4])
        self.assertEqual(body["pos"], 0)

    def test_topk0_without_candidates_422(self):
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1], "topk": 0})
        self.assertEqual(status, 422)
        self.assertIn("candidates", body["error"])

    def test_candidates_and_topk_coexist(self):
        status, body, _ = self.env.post("/v1/score",
                                        {"prompt_ids": [1], "topk": 3, "candidates": [10]})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["logprobs"]), 3)
        self.assertEqual([candidate["id"] for candidate in body["candidates"]], [10])

    def test_candidates_duplicate_400(self):
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1], "candidates": [2, 2]})
        self.assertEqual(status, 400)
        self.assertIn("distinct", body["error"])

    def test_candidates_over_limit_400(self):
        status, body, _ = self.env.post("/v1/score",
                                        {"prompt_ids": [1], "candidates": list(range(2049))})
        self.assertEqual(status, 400)
        self.assertIn("2048", body["error"])

    def test_candidates_outside_vocabulary_engine_422(self):
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1], "candidates": [999999]})
        self.assertEqual(status, 422)
        self.assertEqual(body["error_source"], "engine")
        self.assertIn("outside the vocabulary", body["error"])


class CandidateProtocolViolationTests(unittest.TestCase):
    """P2R-1: a malformed CANDLOGPROBS answer is a protocol violation, never passed on."""

    def _env(self, mode):
        env = StubService(cand_mode=mode)
        self.addCleanup(env.close)
        return env

    def test_count_mismatch_502(self):
        env = self._env("count-bump")
        status, body, _ = env.post("/v1/score", {"prompt_ids": [1], "candidates": [3]})
        self.assertEqual(status, 502)
        self.assertIn("CANDLOGPROBS", body["error"])

    def test_unsolicited_line_502(self):
        env = self._env("unsolicited")
        status, body, _ = env.post("/v1/score", {"prompt_ids": [1]})
        self.assertEqual(status, 502)
        self.assertIn("unrequested", body["error"])

    def test_two_lines_502(self):
        env = self._env("twice")
        status, body, _ = env.post("/v1/score", {"prompt_ids": [1], "candidates": [3]})
        self.assertEqual(status, 502)
        self.assertIn("two CANDLOGPROBS", body["error"])

    def test_wrong_id_order_502(self):
        env = self._env("swap")
        status, body, _ = env.post("/v1/score", {"prompt_ids": [1], "candidates": [3, 9]})
        self.assertEqual(status, 502)
        self.assertIn("order", body["error"])


class DecisionsTests(unittest.TestCase):
    def setUp(self):
        self.env = StubService()
        self.addCleanup(self.env.close)

    def test_unknown_kind_rejected(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "risk", "prompt_ids": [1], "candidates": [[0], [1]]})
        self.assertEqual(status, 422)
        self.assertIn("kind must be one of", body["error"])

    def test_multi_token_candidate_rejected(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "choice", "prompt_ids": [1], "candidates": [[1, 2], [3]]})
        self.assertEqual(status, 422)
        self.assertIn("multi-token", body["error"])

    def test_duplicate_candidate_rejected(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "choice", "prompt_ids": [1], "candidates": [[2], [2]]})
        self.assertEqual(status, 422)
        self.assertIn("distinct", body["error"])

    def test_choice_normal_label_mass_always_returned(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "choice", "prompt_ids": [1, 2], "candidates": [[0], [1], [2]]})
        self.assertEqual(status, 200)
        self.assertEqual(body["kind"], "choice")
        self.assertEqual(body["generated_tokens"], 0)
        self.assertIs(body["full_vocab_softmax"], True)
        self.assertEqual(body["pos"], 1)
        self.assertEqual([candidate["token_ids"] for candidate in body["candidates"]], [[0], [1], [2]])
        for candidate in body["candidates"]:
            self.assertAlmostEqual(candidate["full_vocab_probability"], math.exp(candidate["logprob"]),
                                   places=12)
            self.assertAlmostEqual(candidate["probability"], candidate["full_vocab_probability"], places=15)
            self.assertLessEqual(candidate["logprob"], 0.0)
        mass = body["label_mass"]
        self.assertAlmostEqual(mass, sum(candidate["full_vocab_probability"] for candidate in body["candidates"]),
                               places=12)
        self.assertFalse(body["label_mass_covers_distribution"])
        self.assertEqual(body["label_mass_tolerance"], score_api.LABEL_MASS_TOLERANCE)
        self.assertIs(body["calibrated"], False)
        self.assertNotIn("label_mass_omitted", body)
        for candidate in body["candidates"]:
            self.assertAlmostEqual(candidate["conditional_probability"],
                                   candidate["full_vocab_probability"] / mass, places=12)
        scopes = body["probability_scopes"]
        self.assertEqual(set(scopes), {"full_vocab_probability", "conditional_probability",
                                       "business_confidence"})
        for text in scopes.values():
            self.assertIsInstance(text, str)
            self.assertTrue(text)

    def test_low_mass_has_no_confidence_fields(self):
        # mass ~= 0.27 for ids 1 and 5 in the stub's fixed distribution: clearly below the tolerance,
        # and the payload must say so without inventing any confidence or threshold.
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "choice", "prompt_ids": [1], "candidates": [[1], [5]]})
        self.assertEqual(status, 200)
        self.assertLess(body["label_mass"], 0.5)
        self.assertGreater(body["label_mass"], 0.0)
        self.assertFalse(body["label_mass_covers_distribution"])
        self.assertIs(body["calibrated"], False)
        for candidate in body["candidates"]:
            self.assertIsNotNone(candidate["conditional_probability"])
            self.assertGreater(candidate["conditional_probability"], candidate["full_vocab_probability"])
        for key in body:
            self.assertNotIn("confidence", key)
            self.assertNotIn("threshold", key)
        for candidate in body["candidates"]:
            self.assertNotIn("confidence", " ".join(candidate))

    def test_score_normal(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "score", "prompt_ids": [3], "candidates": [[0], [2]]})
        self.assertEqual(status, 200)
        self.assertEqual(body["kind"], "score")
        self.assertEqual(len(body["candidates"]), 2)
        self.assertEqual(body["generated_tokens"], 0)
        self.assertIn("label_mass", body)
        self.assertFalse(body["label_mass_covers_distribution"])
        self.assertNotIn("label_mass_omitted", body)

    def test_yes_no_payload_mass_and_scopes(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "yes_no", "prompt_ids": [7], "candidates": [[0], [1]]})
        self.assertEqual(status, 200)
        self.assertEqual([candidate["token_ids"] for candidate in body["candidates"]], [[0], [1]])
        self.assertIn("label_mass", body)
        self.assertFalse(body["label_mass_covers_distribution"])
        self.assertIs(body["calibrated"], False)
        self.assertEqual(len(body["probability_scopes"]), 3)

    def test_candidate_outside_vocabulary_422(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "choice", "prompt_ids": [1],
                                         "candidates": [[999999], [1]]})
        self.assertEqual(status, 422)
        self.assertEqual(body["error_source"], "engine")
        self.assertIn("outside the vocabulary", body["error"])

    def test_yes_no_normal_and_arity(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "yes_no", "prompt_ids": [7], "candidates": [[0], [1]]})
        self.assertEqual(status, 200)
        self.assertEqual(body["kind"], "yes_no")
        self.assertEqual([candidate["token_ids"] for candidate in body["candidates"]], [[0], [1]])
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "yes_no", "prompt_ids": [7], "candidates": [[0], [1], [2]]})
        self.assertEqual(status, 422)
        self.assertIn("exactly two", body["error"])

    def test_candidate_outside_topk_still_scored(self):
        # P2R-1 semantic change: explicit candidate scoring has no top-k dependency, so ids outside the
        # distribution's top-k are answered (the old "outside the engine's returned top-N" 422 is gone).
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "choice", "prompt_ids": [1], "candidates": [[7], [8]]})
        self.assertEqual(status, 200)
        self.assertEqual([candidate["token_ids"] for candidate in body["candidates"]], [[7], [8]])
        self.assertIn("label_mass", body)


class FullCoverTests(unittest.TestCase):
    def setUp(self):
        self.env = StubService(vocab=4)
        self.addCleanup(self.env.close)

    def test_label_mass_on_closed_cover(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "choice", "prompt_ids": [1], "topk": 4,
                                         "candidates": [[0], [1], [2], [3]]})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["candidates"]), 4)
        self.assertIn("label_mass", body)
        self.assertNotIn("label_mass_omitted", body)
        self.assertLessEqual(abs(body["label_mass"] - 1.0), score_api.LABEL_MASS_TOLERANCE)
        self.assertIs(body["label_mass_covers_distribution"], True)
        total_conditional = sum(candidate["conditional_probability"] for candidate in body["candidates"])
        self.assertAlmostEqual(total_conditional, 1.0, places=9)


class BusyTests(unittest.TestCase):
    def test_concurrent_request_gets_503(self):
        env = StubService(score_delay_ms=2000)
        self.addCleanup(env.close)
        first = {}

        def slow_call():
            first["result"] = env.post("/v1/score", {"prompt_ids": [1, 2]})

        thread = threading.Thread(target=slow_call)
        thread.start()
        try:
            time.sleep(0.4)
            status, body, headers = env.post("/v1/score", {"prompt_ids": [3, 4]})
            self.assertEqual(status, 503)
            self.assertEqual(body["error_source"], "service")
            self.assertIn("in flight", body["error"])
            self.assertEqual(headers.get("Retry-After"), "1")
        finally:
            thread.join(30)
        self.assertEqual(first["result"][0], 200)


class TimeoutTests(unittest.TestCase):
    def test_engine_no_response_504_then_dead_502(self):
        env = StubService(drop_scores=1, timeout=1.0, drain_timeout=0.5)
        self.addCleanup(env.close)
        started = time.monotonic()
        status, body, _ = env.post("/v1/score", {"prompt_ids": [1]})
        self.assertEqual(status, 504)
        self.assertIn("did not answer", body["error"])
        self.assertLess(time.monotonic() - started, 5.0)
        status, body, _ = env.post("/v1/score", {"prompt_ids": [1]})
        self.assertEqual(status, 502)
        self.assertIn("dead", body["error"])


class CliTests(unittest.TestCase):
    def test_cli_start_engine_end_to_end(self):
        command = [PYTHON, "-B", "-u", str(TOOLS / "score_api.py"),
                   "--start-engine", "--host", "127.0.0.1", "--port", "0",
                   "--engine", PYTHON, "--argv", str(STUB), "--vocab", "8", "--serve"]
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
        port = int(listen_line.rsplit(":", 1)[1].strip())
        status, caps, _ = request(port, "GET", "/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(caps["service"], "score_api")
        self.assertEqual(caps["engine_sha256"], hashlib.sha256(Path(PYTHON).read_bytes()).hexdigest())
        status, body, _ = request(port, "POST", "/v1/score", {"prompt_ids": [1, 2, 3], "topk": 5})
        self.assertEqual(status, 200)
        self.assertEqual(body["generated_tokens"], 0)
        self.assertEqual(len(body["logprobs"]), 5)
        self.assertEqual(body["pos"], 2)


class ContinueSessionTests(unittest.TestCase):
    """P2R-3c: keep=1 state-keeping score, /v1/continue and GET /v1/session (qwen35 stub)."""

    def setUp(self):
        self.env = StubService()
        self.addCleanup(self.env.close)

    def arm(self, ids=(1, 2, 3)):
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": list(ids), "topk": 2, "keep": True})
        self.assertEqual(status, 200, body)
        return body

    def session(self):
        status, body, _ = self.env.get("/v1/session")
        self.assertEqual(status, 200, body)
        return body

    def test_score_keep_true_arms_session(self):
        body = self.arm([1, 2, 3])
        self.assertIs(body["keep"], True)
        session = self.session()
        self.assertIs(session["armed"], True)
        self.assertIs(session["engine_state"]["session"], True)
        self.assertEqual(session["engine_state"]["origin"], "score-keep")
        self.assertEqual(session["engine_state"]["pos"], 3)
        self.assertIsNone(session["engine_state"]["path"])
        self.assertIsNone(session["engine_state"]["file_bytes"])

    def test_score_without_keep_clears_session(self):
        self.arm()
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1], "topk": 1})
        self.assertEqual(status, 200)
        self.assertIs(body["keep"], False)
        session = self.session()
        self.assertIs(session["armed"], False)
        self.assertIs(session["engine_state"]["session"], False)
        self.assertIsNone(session["engine_state"]["origin"])
        self.assertEqual(session["engine_state"]["pos"], 0)

    def test_score_keep_requires_boolean(self):
        for value in (1, 0, "true", None, [True]):
            status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1], "keep": value})
            self.assertEqual(status, 400, value)
            self.assertIn("keep", body["error"])

    def test_decisions_keep_true_keeps_payload_and_arms(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "yes_no", "prompt_ids": [1, 2],
                                         "candidates": [[0], [1]], "keep": True})
        self.assertEqual(status, 200)
        self.assertIs(body["keep"], True)
        self.assertIn("label_mass", body)
        self.assertIs(body["calibrated"], False)
        self.assertEqual(set(body["probability_scopes"]),
                         {"full_vocab_probability", "conditional_probability", "business_confidence"})
        for candidate in body["candidates"]:
            self.assertIn("full_vocab_probability", candidate)
            self.assertIn("conditional_probability", candidate)
            self.assertIn("probability", candidate)
        session = self.session()
        self.assertIs(session["armed"], True)
        self.assertEqual(session["engine_state"]["origin"], "score-keep")
        self.assertEqual(session["engine_state"]["pos"], 2)

    def test_decisions_without_keep_clears_session(self):
        self.arm()
        status, _, _ = self.env.post("/v1/decisions",
                                     {"kind": "choice", "prompt_ids": [1], "candidates": [[0], [1]]})
        self.assertEqual(status, 200)
        self.assertIs(self.session()["armed"], False)

    def test_decisions_keep_requires_boolean(self):
        status, body, _ = self.env.post("/v1/decisions",
                                        {"kind": "choice", "prompt_ids": [1],
                                         "candidates": [[0], [1]], "keep": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("keep", body["error"])

    def test_continue_pure_and_text_null_without_tokenizer(self):
        self.arm([1, 2, 3])
        status, body, _ = self.env.post("/v1/continue", {"max_new": 5})
        self.assertEqual(status, 200)
        self.assertEqual(body["tokens"], [0, 1, 2, 3, 4])
        self.assertIsNone(body["text"])
        self.assertEqual(body["finish"], "length")
        self.assertEqual(body["reused"], 3)
        self.assertEqual(body["prompt_read"], 0)
        self.assertEqual(body["prompt_tokens"], 0)
        self.assertEqual(body["generated_tokens"], 5)
        self.assertEqual(body["usage"]["prompt_read"], 0)
        self.assertIs(self.session()["armed"], True)

    def test_continue_with_ids_appends_at_cursor(self):
        self.arm([1, 2, 3])
        status, body, _ = self.env.post("/v1/continue", {"max_new": 2, "ids": [5, 6]})
        self.assertEqual(status, 200)
        self.assertEqual(body["tokens"], [0, 1])
        self.assertEqual(body["reused"], 3)
        self.assertEqual(body["prompt_read"], 2)
        self.assertEqual(body["prompt_tokens"], 2)
        session = self.session()
        self.assertIs(session["armed"], True)
        self.assertEqual(session["engine_state"]["pos"], 3 + 2 + 2)

    def test_continue_without_armed_session_is_422_verbatim(self):
        status, body, _ = self.env.post("/v1/continue", {"max_new": 1})
        self.assertEqual(status, 422)
        self.assertEqual(body["error_source"], "engine")
        self.assertEqual(body["error"], "cont=1 needs an armed session")

    def test_continue_multi_round_stays_armed(self):
        self.arm([1, 2, 3])
        status, first, _ = self.env.post("/v1/continue", {"max_new": 4})
        self.assertEqual(status, 200)
        self.assertEqual(first["reused"], 3)
        status, second, _ = self.env.post("/v1/continue", {"max_new": 1})
        self.assertEqual(status, 200)
        self.assertEqual(second["reused"], 3 + 4)
        self.assertIs(self.session()["armed"], True)

    def test_continue_max_new_validation_400(self):
        for payload in ({}, {"max_new": 0}, {"max_new": -1}, {"max_new": "4"}, {"max_new": True}):
            status, body, _ = self.env.post("/v1/continue", payload)
            self.assertEqual(status, 400, payload)
            self.assertIn("max_new", body["error"])

    def test_continue_sampling_fields_400(self):
        for name, value in (("temperature", 0.7), ("seed", 1), ("top_p", 0.9), ("min_p", 0.1),
                            ("top_k", 5)):
            status, body, _ = self.env.post("/v1/continue", {"max_new": 1, name: value})
            self.assertEqual(status, 400, name)
            self.assertIn(name, body["error"])
            self.assertIn("not implemented", body["error"])

    def test_continue_unknown_field_400(self):
        status, body, _ = self.env.post("/v1/continue", {"max_new": 1, "n": 2})
        self.assertEqual(status, 400)
        self.assertIn("unknown field", body["error"])

    def test_continue_ids_validation_400(self):
        for value in ("5", [True], [-1], {}):
            status, body, _ = self.env.post("/v1/continue", {"max_new": 1, "ids": value})
            self.assertEqual(status, 400, value)

    def test_session_readonly_and_fresh_state(self):
        fresh = self.session()
        self.assertIs(fresh["armed"], False)
        self.assertIs(fresh["engine_state"]["session"], False)
        self.assertEqual(fresh["engine_state"]["pos"], 0)
        self.assertEqual(self.session(), fresh)
        self.arm([7, 8])
        before = self.session()
        self.assertEqual(self.session(), before)
        status, body, _ = self.env.post("/v1/continue", {"max_new": 1})
        self.assertEqual(status, 200)
        self.assertEqual(body["reused"], 2)

    def test_capabilities_expose_p2r3a_engine_fields(self):
        status, body, _ = self.env.get("/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertIs(body["state_keep_score"], True)
        self.assertIs(body["session_continue"], True)


class OfficialBackendSessionTests(unittest.TestCase):
    """P2R-3c: the official q36 protocol has no session commands -> explicit 501."""

    def setUp(self):
        self.env = StubService(backend="official_q36", stub=OFFICIAL_STUB)
        self.addCleanup(self.env.close)

    def test_continue_501(self):
        status, body, _ = self.env.post("/v1/continue", {"max_new": 1})
        self.assertEqual(status, 501)
        self.assertEqual(body["error_source"], "service")
        self.assertIn("/v1/continue", body["error"])
        self.assertIn("official_q36", body["error"])
        self.assertIn("STATE", body["error"])

    def test_session_501(self):
        status, body, _ = self.env.get("/v1/session")
        self.assertEqual(status, 501)
        self.assertEqual(body["error_source"], "service")
        self.assertIn("/v1/session", body["error"])
        self.assertIn("official_q36", body["error"])

    def test_score_keep_501(self):
        status, body, _ = self.env.post("/v1/score", {"prompt_ids": [1], "keep": True})
        self.assertEqual(status, 501)
        self.assertEqual(body["error_source"], "service")
        self.assertIn("/v1/score", body["error"])
        self.assertIn("official_q36", body["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
