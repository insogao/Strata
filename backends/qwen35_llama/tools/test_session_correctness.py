"""Session ownership and hostile protocol regressions over real localhost HTTP/pipes.

No GPU: stub tests certify the service guard, not native numerical parity. Native
split-token/state parity is a separately leased hardware test (see delivery report).
"""
import json
import threading
import unittest
from unittest.mock import patch

import score_api as api
from test_score_api import StubService, raw_post


class SessionOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.env = StubService()
        self.addCleanup(self.env.close)

    def arm(self):
        status, data, _ = self.env.post("/v1/score", {"prompt_ids": [1, 2, 3], "keep": True})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["session"]["revision"], 1)
        return data["session"]

    def state(self):
        status, data, _ = self.env.get("/v1/session")
        self.assertEqual(status, 200)
        return data

    def test_other_caller_cannot_append_clear_or_continue(self):
        owner = self.arm()
        before = self.state()
        calls = [("/v1/score", {"prompt_ids": [9], "keep": True}),
                 ("/v1/score", {"prompt_ids": [9]}),
                 ("/v1/decisions", {"kind": "choice", "prompt_ids": [9], "candidates": [[0], [1]]}),
                 ("/v1/continue", {"max_new": 1}),
                 ("/v1/continue", {"max_new": 1, "session": {"id": "wrong", "revision": 1}})]
        for path, body in calls:
            with patch.object(self.env.link, "send", wraps=self.env.link.send) as sent:
                self.assertEqual(self.env.post(path, body)[0], 409)
                sent.assert_not_called()
            self.assertEqual(self.state(), before)
        status, data, _ = self.env.post("/v1/continue", {"max_new": 1, "session": owner})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["reused"], 3)
        self.assertEqual(data["session"], {"id": owner["id"], "revision": 2})
        self.assertNotIn(owner["id"], json.dumps(before))  # STATE never discloses the bearer secret

    def test_stale_revision_cannot_replay_even_for_same_owner(self):
        owner = self.arm()
        body = {"max_new": 2, "session": owner}
        self.assertEqual(self.env.post("/v1/continue", body)[0], 200)
        before = self.state()
        self.assertEqual(self.env.post("/v1/continue", body)[0], 409)
        self.assertEqual(self.state(), before)

    def test_owner_append_release_and_new_identity(self):
        owner = self.arm()
        status, data, _ = self.env.post("/v1/score", {"prompt_ids": [4], "keep": True, "session": owner})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["pos"], 0)  # LOGPROBS pos is local to this one-token append
        self.assertEqual(self.state()["engine_state"]["pos"], 4)
        self.assertEqual(data["session"]["revision"], 2)
        status, released, _ = self.env.post("/v1/score", {"prompt_ids": [9], "session": data["session"]})
        self.assertEqual(status, 200, released)
        self.assertIsNone(released["session"])
        self.assertFalse(self.state()["armed"])
        new = self.arm()
        self.assertNotEqual(new["id"], owner["id"])
        self.assertEqual(self.env.post("/v1/continue", {"max_new": 1, "session": owner})[0], 409)

    def test_one_revision_has_only_one_successful_mutation(self):
        owner = self.arm()
        barrier = threading.Barrier(3)
        results = []
        def request():
            barrier.wait()
            results.append(self.env.post("/v1/continue", {"max_new": 1, "session": owner})[0])
        threads = [threading.Thread(target=request) for _ in range(2)]
        for t in threads:
            t.start()
        barrier.wait()
        for t in threads:
            t.join(10)
            self.assertFalse(t.is_alive())
        self.assertEqual(results.count(200), 1, results)
        self.assertIn(next(code for code in results if code != 200), (409, 503))
        self.assertEqual(self.state()["engine_state"]["pos"], 4)

    def test_failed_mutation_requires_owned_reset(self):
        owner = self.arm()
        status, _, _ = self.env.post("/v1/continue", {"max_new": 1, "ids": [64], "session": owner})
        self.assertEqual(status, 422)
        self.assertEqual(self.env.post("/v1/continue", {"max_new": 1, "session": owner})[0], 409)
        self.assertEqual(self.env.post("/v1/score", {"prompt_ids": [9]})[0], 409)
        status, data, _ = self.env.post("/v1/score", {"prompt_ids": [9], "session": owner})
        self.assertEqual(status, 200, data)
        self.assertIsNone(data["session"])

    def test_bad_handle_shape_and_unbounded_length_are_rejected_without_mutation(self):
        owner = self.arm()
        before = self.state()
        for handle in ({}, {"id": owner["id"], "revision": True},
                       {"id": "中文", "revision": 1}, {"id": owner["id"], "revision": 0}):
            self.assertEqual(self.env.post("/v1/continue", {"max_new": 1, "session": handle})[0], 400)
        for value in (2**31, 2**63 - 1, 2**63, 10**100):
            self.assertEqual(self.env.post("/v1/continue", {"max_new": value, "session": owner})[0], 400)
        self.assertEqual(self.state(), before)


class ProbabilityBoundaryTests(unittest.TestCase):
    def test_illegal_rows_rejected_in_both_protocol_families(self):
        for name, parser in (("LOGPROBS", api._parse_logprobs), ("CANDLOGPROBS", api._parse_candlogprobs)):
            for row in ("0 1 1:nan", "0 1 1:inf", "0 1 1:-inf", "0 1 1:0.5",
                        "-1 1 1:-1", "0 1 -1:-1", "0 2 1:-1 1:-2", "0 -1 1:-1",
                        "0 2 1:-0.1 2:-0.1"):
                with self.subTest(name=name, row=row), self.assertRaises(api.EngineProtocolError):
                    parser(name + " " + row)
            self.assertEqual(parser(name + " 0 1 1:-0.0000")[1][0]["logprob"], 0)

    def test_invalid_done_fields_fail_protocol_boundary(self):
        parts = "DONE 1 0 0.0 1.0 length 0 0 0 0 0 0 0 0.0 0 0".split()
        for index, value in ((1, "-1"), (3, "nan"), (4, "inf"), (5, "nonfinite")):
            broken = parts.copy()
            broken[index] = value
            with self.subTest(index=index), self.assertRaises(api.EngineProtocolError):
                api._parse_done(" ".join(broken))

    def test_bad_probabilities_quarantine_real_pipe_and_never_return_nan(self):
        for mode in ("nan-logprobs", "positive-logprobs", "nan-candidates"):
            with self.subTest(mode=mode):
                env = StubService(fault_mode=mode)
                try:
                    path = "/v1/decisions" if mode == "nan-candidates" else "/v1/score"
                    payload = ({"kind": "choice", "prompt_ids": [1], "candidates": [[0], [1]]}
                               if path.endswith("decisions") else {"prompt_ids": [1], "topk": 1})
                    status, data, _ = env.post(path, payload)
                    self.assertEqual(status, 502, data)
                    self.assertTrue(data["quarantined"])
                    json.dumps(data, allow_nan=False)
                    self.assertTrue(env.service._dead)
                    self.assertEqual(env.post("/v1/score", {"prompt_ids": [1]})[0], 502)
                finally:
                    env.close()

    def test_nonfinite_request_json_is_400(self):
        env = StubService()
        try:
            self.assertEqual(raw_post(env.port, "/v1/continue", b'{"max_new":NaN}')[0], 400)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
