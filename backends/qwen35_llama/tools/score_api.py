#!/usr/bin/env python3
"""score_api.py - thin HTTP scoring service over a serve line protocol engine.

This is a *subset* service, not a fully protocol-compatible server. It drives one existing
engine process through a documented stdin/stdout protocol and exposes:

    GET  /v1/capabilities   the backend's capability table (engine CAPS passthrough for
                            qwen35_llama; framework-synthesized for official_q36) + service fields
    GET  /v1/routes         P2R-5: read-only service discovery - the declarative routing matrix
                            (every endpoint x backend: supported / engine command / refusal /
                            notes) + active_backend + backend_source; no engine command
    POST /v1/score          read-only top-k log-softmax of one prompt (optional keep=true:
                            state-keeping score, no reset, session stays armed)
    POST /v1/decisions      minimal choice / score / yes_no candidate scoring (optional
                            keep=true: a decision and its explanation share one prefill)
    POST /v1/continue       P2R-3c: continue an armed session (engine `GEN <max_new> cont=1
                            [<ids>]`), no re-prefill of the carried prefix
    GET  /v1/session        P2R-3c: read-only engine `STATE` passthrough + the service's
                            `armed` view; never mutates (no reset/arm)

Two backends are selectable with --backend (or the config field "backend"; P2-7, see
tools/backend_adapters.py): `qwen35_llama` (the in-repo backend: CAPS/SCORE/GEN/SAVE/LOAD/
STATE/STOP/QUIT, docs/backends/qwen35_llama_serve_protocol.md) and `official_q36` (the
official q36 engine's own --serve protocol: READY/GEN/PP/T/DONE/STOP). The official protocol
has no CAPS/SCORE/SAVE/LOAD, so its capability table is synthesized (logits_read and
state_export are false) and /v1/score + /v1/decisions answer an explicit 501 there - never a
silent fallback to generation. Design and protocol differences:
docs/backends/qwen35_llama_backend_adapters.md.

P2R-5 note (routing matrix): every endpoint gate is the declarative (endpoint, backend) matrix
in backend_adapters.py (ROUTES) - the service no longer decides support by hand. The 501
statuses and texts on official_q36 are byte-identical to the pre-matrix ones (pinned by
tools/test_backend_adapters.py and tools/test_score_api.py), and `GET /v1/routes` publishes
the matrix read-only (`active_backend`, `backend_source`, every cell with supported /
engine_command / refusal / notes). Single-entry deployment (chat via the official server +
score/decisions/session via this service) is a documented recipe, not a merged service - see
docs/backends/qwen35_llama_backend_adapters.md.

One engine, one in-flight request: a concurrent request is answered 503 (no queue).
Stateful workflows additionally carry an opaque session id and current revision:
keep=true issues the handle; subsequent score/decision/continue mutations without
the current handle answer409 before touching the pipe. See
docs/SESSION_CORRECTNESS_REVIEW_20261015.md. There is
no chat/completions, no streaming and no batching. String
`prompt` input needs an optional tokenizer (--tokenizer-model); without it the request is
refused with an explanation instead of being guessed at - except `/v1/continue`, whose `text`
is `null` when no tokenizer is available (never an error).

P2R-3c note (session endpoints): `keep` / `/v1/continue` / `GET /v1/session` speak the
P2R-3a engine contract (`docs/backends/qwen35_llama_session_state.md` §3b). They exist only
on a backend whose command surface has the session commands (`STATE`); the official q36
protocol has none of this, so `/v1/continue` and `/v1/session` answer an explicit 501 there,
and `/v1/score` (the whole family, `keep` included) is refused by the same matrix gate - never
a silent substitution.

P2R-4b note (link quarantine): an `EngineProtocolError` is not just answered 502 - the link is
poisoned and the service quarantines it. The failing request's 502 keeps the original error
text and adds `"quarantined": true`; the engine process is terminated (when the service
started it); every later engine-touching endpoint answers 502 (dead/quarantined) and the pipe
is never read again. Recovery is a service restart - never an automatic resync, because this
unframed line protocol has no reliable resynchronization point. A legitimate engine `ERR`
(a business rejection) is *not* a protocol error: it stays a 422 passthrough with no
quarantine.

Tests (0 GPU, stub engines): tools/test_score_api.py (qwen35 protocol) and
tools/test_backend_adapters.py (both backends); test-only stubs: tools/stub_engine.py,
tools/stub_engine_official_q36.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from backend_adapters import (DEFAULT_BACKEND, SUPPORTED_BACKENDS, get_adapter, refusal_text,
                              route, routes_payload)

SERVICE_NAME = "score_api"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8097
DEFAULT_TOPK = 5
DEFAULT_TIMEOUT = 600.0
DEFAULT_START_TIMEOUT = 600.0
DEFAULT_DRAIN_TIMEOUT = 10.0
LABEL_MASS_TOLERANCE = 1e-3
MAX_BODY_BYTES = 1 << 20
MAX_EXPLICIT_CANDIDATES = 2048
DECISION_KINDS = ("choice", "score", "yes_no")
# Sampling parameters that exist on `GEN` but are NOT implemented on /v1/continue: their
# presence is an explicit 400 refusal (never a silent ignore).
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seed")

_EOF = object()


class ServiceError(Exception):
    status = 500
    source = "service"
    # P2R-4b: set on the exception instance when the service quarantined the engine link
    # (a protocol error). The HTTP layer mirrors it as the machine-readable
    # "quarantined": true response field.
    quarantined = False


class BadRequest(ServiceError):
    status = 400


class PayloadTooLarge(ServiceError):
    status = 413


class NotImplementedFeature(ServiceError):
    status = 501


class Unprocessable(ServiceError):
    status = 422


class Busy(ServiceError):
    status = 503


class EngineError(ServiceError):
    status = 422
    source = "engine"


class EngineTimeout(ServiceError):
    status = 504


class EngineDead(ServiceError):
    status = 502


class EngineProtocolError(ServiceError):
    # P2R-4b: the engine violated the line protocol. The violation is engine-side, and
    # _engine_call reacts to it with link quarantine (see _engine_call).
    status = 502
    source = "engine"


class SessionConflict(ServiceError):
    status = 409


def _timeout_message(timeout: float) -> str:
    return "the engine did not answer in time (request timeout %.1f s)" % timeout


@dataclass(frozen=True)
class ScoreResult:
    pos: int
    pairs: list
    done: dict
    cand_pairs: list = None


@dataclass(frozen=True)
class GenerationResult:
    tokens: list
    done: dict


def _parse_probability_row(line: str, name: str):
    parts = line.split()
    if len(parts) < 4:
        raise EngineProtocolError("short %s line: %r" % (name, line))
    try:
        pos = int(parts[1])
        count = int(parts[2])
        pairs = []
        seen = set()
        for token in parts[3:]:
            token_id, separator, logprob = token.partition(":")
            token_id, logprob = int(token_id), float(logprob)
            if not separator or token_id < 0 or not math.isfinite(logprob) or logprob > 0:
                raise ValueError("requires a nonnegative id and finite logprob <= 0")
            if token_id in seen:
                raise ValueError("duplicate token id")
            seen.add(token_id)
            pairs.append({"id": token_id, "logprob": logprob})
        if pos < 0 or count < 1:
            raise ValueError("requires nonnegative position and positive count")
    except ValueError as exc:
        raise EngineProtocolError("malformed %s line: %r (%s)" % (name, line, exc))
    if len(pairs) != count:
        raise EngineProtocolError("%s declares %d pairs but has %d: %r" % (name, count, len(pairs), line))
    if math.fsum(math.exp(pair["logprob"]) for pair in pairs) > 1.0 + LABEL_MASS_TOLERANCE:
        raise EngineProtocolError("%s probabilities exceed full-vocabulary mass" % name)
    return pos, pairs


def _parse_logprobs(line: str):
    return _parse_probability_row(line, "LOGPROBS")


def _parse_candlogprobs(line: str):
    return _parse_probability_row(line, "CANDLOGPROBS")


def _parse_done(line: str) -> dict:
    parts = line.split()
    if len(parts) < 16:
        raise EngineProtocolError("short DONE line: %r" % line)
    try:
        result = {
            "produced": int(parts[1]),
            "prompt_tokens": int(parts[2]),
            "prompt_ms": float(parts[3]),
            "decode_ms": float(parts[4]),
            "finish": parts[5],
            "reused": int(parts[8]),
            "prompt_read": int(parts[14]),
        }
        if (any(result[key] < 0 for key in ("produced", "prompt_tokens", "reused", "prompt_read"))
                or any(not math.isfinite(result[key]) or result[key] < 0 for key in ("prompt_ms", "decode_ms"))
                or result["finish"] not in {"score", "length", "stop", "cancel"}):
            raise ValueError("invalid counters, timing or finish reason")
        return result
    except ValueError as exc:
        raise EngineProtocolError("malformed DONE line: %r (%s)" % (line, exc))


class EngineLink:
    """Owns one engine process's stdin/stdout; one logical request at a time."""

    def __init__(self, writer, reader, proc=None):
        self._writer = writer
        self._reader = reader
        self._closed = False
        self.proc = proc
        self._queue: "queue.Queue" = queue.Queue()
        self._dead = threading.Event()
        threading.Thread(target=self._pump, args=(reader,), name="engine-reader", daemon=True).start()

    @property
    def dead(self) -> bool:
        return self._dead.is_set()

    def _pump(self, reader) -> None:
        try:
            for raw in reader:
                self._queue.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
        except Exception:
            pass
        finally:
            self._dead.set()
            self._queue.put(_EOF)

    def send(self, line: str) -> None:
        if self._dead.is_set():
            raise EngineDead("the engine stdin/stdout link is closed")
        try:
            self._writer.write((line + "\n").encode("utf-8"))
            self._writer.flush()
        except (OSError, ValueError) as exc:
            self._dead.set()
            raise EngineDead("cannot write to the engine: %s" % exc)

    def _take_until(self, deadline: float, timeout: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EngineTimeout(_timeout_message(timeout))
        try:
            item = self._queue.get(timeout=remaining)
        except queue.Empty:
            raise EngineTimeout(_timeout_message(timeout))
        if item is _EOF:
            raise EngineDead("the engine stdin/stdout link is closed (process exited?)")
        return item

    def wait_ready(self, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        line = self._take_until(deadline, timeout)
        if not line.startswith("READY "):
            raise EngineProtocolError("expected a READY line from the engine, got %r" % line)
        return line

    def read_line(self, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        return self._take_until(deadline, timeout)

    def caps(self, timeout: float) -> dict:
        self.send("CAPS")
        deadline = time.monotonic() + timeout
        while True:
            line = self._take_until(deadline, timeout)
            if line.startswith("CAPS "):
                try:
                    payload = json.loads(line[5:])
                except ValueError as exc:
                    raise EngineProtocolError("CAPS line is not JSON: %s" % exc)
                if not isinstance(payload, dict):
                    raise EngineProtocolError("CAPS payload is not a JSON object")
                return payload
            if line.startswith("ERR "):
                raise EngineError(line[4:])
            raise EngineProtocolError("unexpected line while waiting for CAPS: %r" % line)

    def score(self, ids, topk, timeout, candidates=None, keep=False) -> ScoreResult:
        command = "SCORE " + ",".join(str(int(i)) for i in ids)
        if topk is not None:
            command += " topk=%d" % topk
        if candidates is not None:
            command += " cand=" + ",".join(str(int(c)) for c in candidates)
        if keep:
            command += " keep=1"
        self.send(command)
        deadline = time.monotonic() + timeout
        pos = None
        pairs = None
        cand_pairs = None
        while True:
            line = self._take_until(deadline, timeout)
            if line.startswith("PP "):
                continue
            if line.startswith("LOGPROBS "):
                if pairs is not None:
                    raise EngineProtocolError("two LOGPROBS lines for one SCORE")
                if topk is not None and topk == 0:
                    raise EngineProtocolError("LOGPROBS for a topk=0 SCORE (only explicit candidates "
                                              "were requested, none may be answered)")
                pos, pairs = _parse_logprobs(line)
                continue
            if line.startswith("CANDLOGPROBS "):
                if candidates is None:
                    raise EngineProtocolError("unrequested CANDLOGPROBS line (no cand= was sent): %r" % line)
                if cand_pairs is not None:
                    raise EngineProtocolError("two CANDLOGPROBS lines for one SCORE")
                cand_pos, cand_pairs = _parse_candlogprobs(line)
                if pos is None:
                    pos = cand_pos
                elif cand_pos != pos:
                    raise EngineProtocolError("CANDLOGPROBS pos %d differs from LOGPROBS pos %d"
                                              % (cand_pos, pos))
                continue
            if line.startswith("DONE "):
                done = _parse_done(line)
                if (topk != 0) and pairs is None:
                    raise EngineProtocolError("SCORE finished without a LOGPROBS line")
                if candidates is not None:
                    if cand_pairs is None:
                        raise EngineProtocolError("SCORE finished without the requested CANDLOGPROBS line")
                    if [pair["id"] for pair in cand_pairs] != [int(c) for c in candidates]:
                        raise EngineProtocolError("CANDLOGPROBS ids %s do not match the requested candidates %s "
                                                  "(the answer must be in request order)"
                                                  % ([pair["id"] for pair in cand_pairs],
                                                     [int(c) for c in candidates]))
                if done["produced"] != 0 or done["finish"] != "score":
                    raise EngineProtocolError("SCORE ended with produced=%d finish=%s"
                                              % (done["produced"], done["finish"]))
                return ScoreResult(pos=pos, pairs=pairs, done=done, cand_pairs=cand_pairs)
            if line.startswith("ERR "):
                raise EngineError(line[4:])
            if line.startswith("T "):
                raise EngineProtocolError("SCORE generated a token (%r); the engine contract forbids it" % line)
            raise EngineProtocolError("unexpected line during SCORE: %r" % line)

    def generate_continue(self, max_new, ids, timeout) -> GenerationResult:
        """P2R-3c: engine `GEN <max_new> cont=1 [<ids>]` - continue an armed session."""
        command = "GEN %d cont=1" % max_new
        if ids:
            command += " " + ",".join(str(int(i)) for i in ids)
        self.send(command)
        deadline = time.monotonic() + timeout
        tokens = []
        while True:
            line = self._take_until(deadline, timeout)
            if line.startswith("PP "):
                continue
            if line.startswith("T "):
                try:
                    token = int(line[2:].strip())
                    if token < 0 or token > 2147483647:
                        raise ValueError("token id outside int32 range")
                    tokens.append(token)
                except ValueError:
                    raise EngineProtocolError("malformed T line during GEN cont=1: %r" % line)
                if len(tokens) > max_new:
                    raise EngineProtocolError("GEN cont=1 emitted more than max_new=%d T lines" % max_new)
                continue
            if line.startswith("DONE "):
                done = _parse_done(line)
                if done["finish"] not in {"length", "stop", "cancel"}:
                    raise EngineProtocolError("GEN cont=1 ended with invalid finish=%s" % done["finish"])
                if done["produced"] > max_new:
                    raise EngineProtocolError("GEN cont=1 finished with produced=%d exceeding max_new=%d"
                                              % (done["produced"], max_new))
                if done["produced"] != len(tokens):
                    raise EngineProtocolError("GEN cont=1 finished with produced=%d but %d T lines"
                                              % (done["produced"], len(tokens)))
                if done["finish"] == "length" and done["produced"] != max_new:
                    raise EngineProtocolError("GEN cont=1 finished length with produced=%d but max_new=%d"
                                              % (done["produced"], max_new))
                if done["finish"] == "stop" and done["produced"] == 0:
                    raise EngineProtocolError("GEN cont=1 finished stop without an EOS token")
                return GenerationResult(tokens=tokens, done=done)
            if line.startswith("ERR "):
                raise EngineError(line[4:])
            raise EngineProtocolError("unexpected line during GEN cont=1: %r" % line)

    def state(self, timeout) -> dict:
        """P2R-5/P2R-3c: one `STATE` round-trip; the engine's bookkeeping JSON, parsed."""
        self.send("STATE")
        deadline = time.monotonic() + timeout
        while True:
            line = self._take_until(deadline, timeout)
            if line.startswith("STATE "):
                try:
                    payload = json.loads(line[6:])
                except ValueError as exc:
                    raise EngineProtocolError("STATE line is not JSON: %s" % exc)
                if not isinstance(payload, dict):
                    raise EngineProtocolError("STATE payload is not a JSON object")
                return payload
            if line.startswith("ERR "):
                raise EngineError(line[4:])
            raise EngineProtocolError("unexpected line while waiting for STATE: %r" % line)

    def cancel_and_drain(self, timeout: float) -> bool:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _EOF:
                return False
            if item.startswith(("DONE ", "ERR ")):
                return True
        try:
            self.send("STOP")
        except EngineDead:
            return False
        deadline = time.monotonic() + timeout
        while True:
            try:
                item = self._take_until(deadline, timeout)
            except (EngineTimeout, EngineDead):
                return False
            if item.startswith(("DONE ", "ERR ")):
                return True

    def quarantine(self) -> str:
        """P2R-4b: kill this link after a protocol error; the queue is never read again.

        Marks the link dead, terminates the engine process when this service owns it
        (attach mode cannot), discards whatever answer bytes are already buffered and
        returns an honest note about termination ("" when the owned process was
        terminated).
        """
        self._dead.set()
        note = ""
        if self.proc is None:
            note = ("this service does not own the engine process (attach mode), so it could "
                    "not terminate it")
        else:
            try:
                self.proc.terminate()
            except Exception as exc:
                note = "terminating the engine process failed (%s: %s)" % (type(exc).__name__, exc)
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _EOF:
                break
        return note

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.close()
        except Exception:
            pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        try:
            self._reader.close()
        except Exception:
            pass


def _require_object(body):
    if not isinstance(body, dict):
        raise BadRequest("the request body must be a JSON object")
    return body


def _reject_unknown_fields(body, allowed):
    unknown = sorted(set(body) - set(allowed))
    if unknown:
        raise BadRequest("unknown field(s) %s; this endpoint accepts %s"
                         % (", ".join(repr(u) for u in unknown), ", ".join(sorted(allowed))))


def _positive_int(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2147483647:
        raise BadRequest("%s must be a positive integer <= 2147483647" % what)
    return value


def _nonnegative_int(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BadRequest("%s must be a non-negative integer" % what)
    return value


def _parse_explicit_candidates(value):
    if not isinstance(value, list) or not value:
        raise BadRequest("candidates must be a non-empty list of token ids")
    if len(value) > MAX_EXPLICIT_CANDIDATES:
        raise BadRequest("candidates has %d ids; the engine's explicit-candidate limit is %d"
                         % (len(value), MAX_EXPLICIT_CANDIDATES))
    ids = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise BadRequest("candidates[%d] is not an integer" % index)
        if item < 0:
            raise BadRequest("candidates[%d]=%d is negative" % (index, item))
        ids.append(item)
    if len(set(ids)) != len(ids):
        duplicates = sorted({t for t in ids if ids.count(t) > 1})
        raise BadRequest("candidates ids must be pairwise distinct; duplicated: %s"
                         % ", ".join(str(t) for t in duplicates))
    return ids


def _parse_prompt_ids(value):
    if not isinstance(value, list):
        raise BadRequest("prompt_ids must be a list of token ids")
    if not value:
        raise BadRequest("prompt_ids must not be empty")
    ids = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise BadRequest("prompt_ids[%d] is not an integer" % index)
        if item < 0:
            raise BadRequest("prompt_ids[%d]=%d is negative" % (index, item))
        ids.append(item)
    return ids


def _parse_continue_ids(value):
    if not isinstance(value, list):
        raise BadRequest("ids must be a list of token ids (omit it or pass [] for a pure continuation)")
    ids = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise BadRequest("ids[%d] is not an integer" % index)
        if item < 0:
            raise BadRequest("ids[%d]=%d is negative" % (index, item))
        ids.append(item)
    return ids


def _parse_candidates(value):
    if not isinstance(value, list) or not value:
        raise BadRequest("candidates must be a non-empty list of token-id lists")
    flat = []
    for index, candidate in enumerate(value):
        if not isinstance(candidate, list):
            raise BadRequest("candidates[%d] must be a list of token ids" % index)
        if not candidate:
            raise BadRequest("candidates[%d] is empty" % index)
        if len(candidate) != 1:
            raise Unprocessable("candidates[%d] has %d tokens; multi-token candidates are not implemented "
                                "in this subset - pass exactly one token id per candidate" % (index, len(candidate)))
        token_id = candidate[0]
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise BadRequest("candidates[%d][0] is not an integer" % index)
        if token_id < 0:
            raise BadRequest("candidates[%d][0]=%d is negative" % (index, token_id))
        flat.append(token_id)
    if len(set(flat)) != len(flat):
        duplicates = sorted({t for t in flat if flat.count(t) > 1})
        raise Unprocessable("candidate ids must be pairwise distinct (mutually exclusive labels); duplicated: %s"
                            % ", ".join(str(t) for t in duplicates))
    if len(flat) > MAX_EXPLICIT_CANDIDATES:
        raise Unprocessable("candidates has %d ids; the engine's explicit-candidate limit is %d"
                            % (len(flat), MAX_EXPLICIT_CANDIDATES))
    return flat


def _usage(done: dict) -> dict:
    return {"prompt_tokens": done["prompt_tokens"], "prompt_ms": done["prompt_ms"],
            "prompt_read": done["prompt_read"]}


class ScoreService:
    def __init__(self, link: EngineLink, engine_path=None, timeout=DEFAULT_TIMEOUT,
                 drain_timeout=DEFAULT_DRAIN_TIMEOUT, tokenizer_model=None, backend=None,
                 backend_source=None):
        self.link = link
        self.engine_path = Path(engine_path) if engine_path else None
        self.timeout = float(timeout)
        self.drain_timeout = float(drain_timeout)
        self.tokenizer_model = tokenizer_model
        if backend is None:
            backend = get_adapter(DEFAULT_BACKEND)
        elif isinstance(backend, str):
            backend = get_adapter(backend)
        self.backend = backend
        # --backend (cli) / the config field / the default; "unspecified" when a service is
        # constructed directly (the library/unit-test path). Reported by GET /v1/routes.
        self.backend_source = backend_source or "unspecified"
        self._busy = threading.Lock()
        self._dead = False
        self._quarantined = False
        self._sha = None
        self._tokenizer = None
        self._tokenizer_error = None
        self._tokenizer_loaded = False
        self._session_lease = None
        self._session_uncertain = False

    def _owned_call(self, body, call, keep):
        """Lease check, engine mutation and revision publication share the pipe lock."""
        provided = body.get("session")
        if provided is not None:
            if (not isinstance(provided, dict) or set(provided) != {"id", "revision"}
                    or not isinstance(provided["id"], str) or not provided["id"].isascii()
                    or not 1 <= len(provided["id"]) <= 128
                    or type(provided["revision"]) is not int or provided["revision"] < 1):
                raise BadRequest("session must contain an opaque id and positive integer revision")

        def owned():
            if self._session_lease is None:
                if provided is not None:
                    raise SessionConflict("session is absent or expired")
            elif (provided is None
                  or not secrets.compare_digest(provided["id"], self._session_lease["id"])
                  or provided["revision"] != self._session_lease["revision"]):
                raise SessionConflict("session ownership or revision conflict; no engine command was sent")
            if self._session_uncertain and keep:
                raise SessionConflict("session state is uncertain after a failed request; use an owned "
                                      "stateless score/decision to reset it, or restart the service")
            try:
                result = call()
            except ServiceError:
                if self._session_lease is not None:
                    self._session_uncertain = True
                raise
            if keep:
                previous = self._session_lease
                self._session_lease = {"id": previous["id"] if previous else secrets.token_urlsafe(32),
                                       "revision": previous["revision"] + 1 if previous else 1}
            else:
                self._session_lease = None
            self._session_uncertain = False
            return result, dict(self._session_lease) if self._session_lease else None

        return self._engine_call(owned)

    def _quarantine_note(self) -> str:
        note = self.link.quarantine()
        if note:
            return ("; the engine link was quarantined after this protocol error, but %s - restart "
                    "the service to get a fresh engine" % note)
        return ("; the engine link was quarantined after this protocol error and the engine was "
                "terminated - restart the service to get a fresh engine")

    def _engine_call(self, call):
        if self._dead:
            if self._quarantined:
                exc = EngineDead("the engine link is dead: it was quarantined after a protocol error; "
                                 "restart the service to get a fresh engine")
                exc.quarantined = True
                raise exc
            raise EngineDead("the engine link is dead; this service must be restarted")
        if not self._busy.acquire(blocking=False):
            raise Busy("a request is already in flight; this service serializes requests on one engine "
                       "(no queue, retry later)")
        try:
            return call()
        except EngineTimeout:
            if not self.link.cancel_and_drain(self.drain_timeout):
                self._dead = True
            raise
        except EngineProtocolError as exc:
            # P2R-4b: a protocol error poisons the whole link. Bytes of this answer may still be
            # unread in the pipe, and this line protocol has no framing and no resynchronization
            # point - a later request could read those stale bytes as its own answer (cross-talk).
            # The honest action is a quarantine: kill the engine, mark the link dead, never read
            # it again; only a service restart provides a fresh engine.
            self._dead = True
            self._quarantined = True
            exc.args = ("%s%s" % (exc, self._quarantine_note()),)
            exc.quarantined = True
            raise
        except EngineDead:
            self._dead = True
            raise
        finally:
            self._busy.release()

    def _engine_sha256(self):
        if self.engine_path is None or not self.engine_path.is_file():
            return None
        if self._sha is None:
            digest = hashlib.sha256()
            with open(self.engine_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            self._sha = digest.hexdigest()
        return self._sha

    def _get_tokenizer(self):
        if self._tokenizer is None and not self._tokenizer_loaded:
            self._tokenizer_loaded = True
            if not self.tokenizer_model:
                self._tokenizer_error = "this service was started without --tokenizer-model"
            else:
                try:
                    sys.path.insert(0, str(Path(__file__).resolve().parent))
                    import strata_tokenizer
                    self._tokenizer = strata_tokenizer.Tokenizer.from_gguf(self.tokenizer_model)
                except Exception as exc:
                    self._tokenizer = None
                    self._tokenizer_error = "the tokenizer could not be loaded from %r (%s: %s)" % (
                        self.tokenizer_model, type(exc).__name__, exc)
        if self._tokenizer is None:
            raise NotImplementedFeature("string prompt input is unavailable: %s; send prompt_ids instead"
                                        % self._tokenizer_error)
        return self._tokenizer

    def _require_route(self, endpoint):
        # P2R-5: support/refusal is decided by the declarative matrix, never by hand. An
        # unsupported in-service cell raises NotImplementedFeature with that cell's own
        # refusal text (byte-identical on official_q36 to the pre-matrix hand-written text).
        cell = route(endpoint, self.backend.name)
        if not cell["supported"]:
            raise NotImplementedFeature(refusal_text(endpoint, self.backend.name))

    @staticmethod
    def _keep_field(body):
        keep = body.get("keep", False)
        if not isinstance(keep, bool):
            raise BadRequest("keep must be a boolean (true/false)")
        return keep

    def score_request(self, body):
        body = _require_object(body)
        self._require_route("POST /v1/score")
        _reject_unknown_fields(body, {"prompt_ids", "prompt", "topk", "candidates", "keep", "session"})
        if ("prompt_ids" in body) == ("prompt" in body):
            raise BadRequest("exactly one of prompt_ids or prompt is required")
        topk = _nonnegative_int(body.get("topk", DEFAULT_TOPK), "topk")
        candidates = _parse_explicit_candidates(body["candidates"]) if "candidates" in body else None
        if candidates is not None:
            # No-op on qwen35_llama (supported); unreachable on official_q36 (the base cell above
            # already refused). Consulted so the matrix's candidates form is part of the gate.
            self._require_route("POST /v1/score:candidates")
        keep = self._keep_field(body)
        if keep:
            # Same for the keep form: qwen35_llama supports it (its engine must declare
            # state_keep_score); the official_q36 base gate already refused the family.
            self._require_route("POST /v1/score:keep")
        if topk == 0 and candidates is None:
            raise Unprocessable("topk=0 answers only explicit candidates; pass candidates (or topk >= 1)")
        if "prompt" in body:
            text = body["prompt"]
            if not isinstance(text, str) or not text:
                raise BadRequest("prompt must be a non-empty string")
            ids = [int(i) for i in self._get_tokenizer().encode(text)]
            if not ids:
                raise Unprocessable("the prompt encoded to zero tokens")
        else:
            ids = _parse_prompt_ids(body["prompt_ids"])
        result, lease = self._owned_call(body, lambda: self.backend.score(self.link, ids, topk, self.timeout,
                                                                         candidates=candidates, keep=keep), keep)
        payload = {"logprobs": result.pairs or [], "pos": result.pos, "full_vocab_softmax": True,
                   "generated_tokens": 0, "usage": _usage(result.done), "keep": keep}
        if candidates is not None:
            payload["candidates"] = result.cand_pairs
        if keep or "session" in body:
            payload["session"] = lease
        return 200, payload

    def capabilities_request(self):
        payload = dict(self._engine_call(lambda: self.backend.capabilities(self.link, self.timeout)))
        payload["service"] = SERVICE_NAME
        payload["engine_sha256"] = self._engine_sha256()
        payload["active_backend"] = self.backend.name
        payload["service_session_contract"] = "opaque-lease-revision/1"
        return 200, payload

    def routes_request(self):
        # P2R-5: read-only service discovery, rendered from the static matrix. No engine command
        # and no busy lock: the declared contract is available even when the link is dead or
        # quarantined (that state is a runtime fact about the engine, not about the routes).
        return 200, {"active_backend": self.backend.name,
                     "backend_source": self.backend_source,
                     "routes": routes_payload()}

    def decisions_request(self, body):
        body = _require_object(body)
        self._require_route("POST /v1/decisions")
        _reject_unknown_fields(body, {"kind", "prompt_ids", "candidates", "topk", "keep", "session"})
        kind = body.get("kind")
        if kind not in DECISION_KINDS:
            raise Unprocessable("kind must be one of %s" % ", ".join(DECISION_KINDS))
        ids = _parse_prompt_ids(body.get("prompt_ids"))
        flat = _parse_candidates(body.get("candidates"))
        if kind == "yes_no" and len(flat) != 2:
            raise Unprocessable("yes_no requires exactly two candidate ids (the caller's yes/no token pair), "
                                "got %d" % len(flat))
        if kind in ("choice", "score") and len(flat) < 2:
            raise Unprocessable("%s requires at least two candidates, got %d" % (kind, len(flat)))
        topk = body.get("topk")
        if topk is not None:
            # Accepted for backward compatibility only: scoring is explicit-id now, so topk no longer
            # constrains anything (the old "topk must cover all candidates" refusal is gone).
            _positive_int(topk, "topk")
        keep = self._keep_field(body)
        # Same engine command family as /v1/score: `SCORE <ids> topk=0 cand=<ids> [keep=1]` - with
        # keep=1 the decision and a later /v1/continue share this one prefill (see score_request).
        result, lease = self._owned_call(body, lambda: self.backend.score(self.link, ids, 0, self.timeout,
                                                                         candidates=flat, keep=keep), keep)
        by_id = {pair["id"]: pair["logprob"] for pair in result.cand_pairs}
        logprobs = [by_id[token_id] for token_id in flat]
        full_probs = [math.exp(logprob) for logprob in logprobs]
        mass = sum(full_probs)
        candidates = []
        for token_id, logprob, full_prob in zip(flat, logprobs, full_probs):
            candidates.append({
                "token_ids": [token_id],
                "logprob": logprob,
                # Full-vocabulary probability: exp(logprob) under the log-softmax over ALL n_vocab logits.
                "full_vocab_probability": full_prob,
                # Conditional on this candidate set only: meaningful within the set, never across sets.
                "conditional_probability": (full_prob / mass) if mass > 0 else None,
                # Deprecated alias of full_vocab_probability (kept for downstream compatibility).
                "probability": full_prob,
            })
        payload = {
            "kind": kind,
            "candidates": candidates,
            "pos": result.pos,
            "full_vocab_softmax": True,
            "generated_tokens": 0,
            "usage": _usage(result.done),
            "keep": keep,
            # Always returned, whatever the mass: the candidate set's share of the full distribution.
            "label_mass": mass,
            "label_mass_covers_distribution": abs(mass - 1.0) <= LABEL_MASS_TOLERANCE,
            "label_mass_tolerance": LABEL_MASS_TOLERANCE,
            # These numbers are uncalibrated; nothing here is a business confidence.
            "calibrated": False,
            "probability_scopes": {
                "full_vocab_probability": "exp(logprob) under the full-vocabulary softmax: comparable "
                                          "across requests; the sum over this candidate set is at most 1.",
                "conditional_probability": "full_vocab_probability / label_mass: meaningful only within "
                                           "this candidate set; not comparable across different candidate sets.",
                "business_confidence": "not provided: no calibration exists, so these numbers are not "
                                       "confidences; e.g. a label_mass of 0.31 means about 69% of the "
                                       "distribution mass lies outside the candidate set.",
            },
        }
        if keep or "session" in body:
            payload["session"] = lease
        return 200, payload

    def _try_decode(self, tokens):
        # /v1/continue `text`: decoded only when a tokenizer is available; no tokenizer (or a
        # decode failure) answers `null` - never an error for this optional field.
        try:
            tokenizer = self._get_tokenizer()
        except NotImplementedFeature:
            return None
        try:
            return tokenizer.decode(tokens)
        except Exception:
            return None

    def continue_request(self, body):
        body = _require_object(body)
        self._require_route("POST /v1/continue")
        sampling = sorted(set(body) & set(SAMPLING_FIELDS))
        if sampling:
            raise BadRequest("sampling is not implemented on /v1/continue: %s; this endpoint always "
                             "decodes greedily from the armed session - remove the field(s) instead "
                             "of relying on them being ignored"
                             % ", ".join(repr(name) for name in sampling))
        _reject_unknown_fields(body, {"max_new", "ids", "session"})
        if "max_new" not in body:
            raise BadRequest("max_new is required (a positive integer)")
        max_new = _positive_int(body["max_new"], "max_new")
        ids = _parse_continue_ids(body.get("ids", []))
        result, lease = self._owned_call(body, lambda: self.link.generate_continue(max_new, ids, self.timeout), True)
        payload = {
            "tokens": result.tokens,
            "text": self._try_decode(result.tokens),
            "finish": result.done["finish"],
            "reused": result.done["reused"],
            "prompt_read": result.done["prompt_read"],
            "prompt_tokens": result.done["prompt_tokens"],
            "generated_tokens": len(result.tokens),
            "usage": _usage(result.done),
            "session": lease,
        }
        return 200, payload

    def session_request(self):
        # Read-only: one STATE query, no reset, no arm. The engine's own `session` field is the
        # arming fact (docs/backends/qwen35_llama_session_state.md); a STATE payload without it
        # is a protocol error, never an invented value. The check runs inside the engine call so
        # that this protocol error goes through the same quarantine as every other one (P2R-4b).
        self._require_route("GET /v1/session")

        def state_call():
            state = self.link.state(self.timeout)
            if not isinstance(state.get("session"), bool):
                raise EngineProtocolError("the engine STATE JSON has no boolean 'session' field: %r"
                                          % sorted(state))
            return state

        state = self._engine_call(state_call)
        return 200, {"engine_state": state, "armed": state["session"]}


class _Handler(BaseHTTPRequestHandler):
    service: ScoreService
    server_version = "score_api/1.0"

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/v1/capabilities":
            self._dispatch(lambda: self.service.capabilities_request())
        elif path == "/v1/routes":
            self._dispatch(lambda: self.service.routes_request())
        elif path == "/v1/session":
            self._dispatch(lambda: self.service.session_request())
        else:
            self._send_json(404, {"error": "not found: %s; this service implements GET "
                                           "/v1/capabilities, GET /v1/routes, GET /v1/session, "
                                           "POST /v1/score, POST /v1/decisions and POST "
                                           "/v1/continue" % path,
                                  "error_source": "service"})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/v1/score":
            self._dispatch(lambda: self.service.score_request(self._read_json()))
        elif path == "/v1/decisions":
            self._dispatch(lambda: self.service.decisions_request(self._read_json()))
        elif path == "/v1/continue":
            self._dispatch(lambda: self.service.continue_request(self._read_json()))
        else:
            self._send_json(404, {"error": "not found: %s; this service implements GET "
                                           "/v1/capabilities, GET /v1/routes, GET /v1/session, "
                                           "POST /v1/score, POST /v1/decisions and POST "
                                           "/v1/continue" % path,
                                  "error_source": "service"})

    def _read_json(self):
        header = self.headers.get("Content-Length")
        try:
            length = int(header) if header else 0
        except ValueError:
            raise BadRequest("invalid Content-Length header: %r" % header)
        if length < 0:
            raise BadRequest("negative Content-Length header")
        if length > MAX_BODY_BYTES:
            raise PayloadTooLarge("request body exceeds %d bytes" % MAX_BODY_BYTES)
        raw = self.rfile.read(length) if length else b""
        try:
            def bad_constant(value):
                raise ValueError("non-finite JSON number: %s" % value)
            return json.loads(raw.decode("utf-8"), parse_constant=bad_constant)
        except (UnicodeDecodeError, ValueError) as exc:
            raise BadRequest("request body is not valid UTF-8 JSON: %s" % exc)

    def _dispatch(self, action):
        try:
            status, payload = action()
        except ServiceError as exc:
            headers = {"Retry-After": "1"} if isinstance(exc, Busy) else None
            payload = {"error": str(exc), "error_source": exc.source}
            if exc.quarantined:
                payload["quarantined"] = True
            self._send_json(exc.status, payload, headers)
        except Exception as exc:
            self._send_json(500, {"error": "internal error: %s: %s" % (type(exc).__name__, exc),
                                  "error_source": "service"})
        else:
            self._send_json(status, payload)

    def _send_json(self, status, payload, headers=None):
        try:
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        except ValueError:
            status = 500
            data = b'{"error":"non-finite response refused","error_source":"service"}\n'
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        sys.stderr.write("score_api: %s - %s\n" % (self.address_string(), fmt % args))
        sys.stderr.flush()


def start_http(service: ScoreService, host=DEFAULT_HOST, port=DEFAULT_PORT) -> ThreadingHTTPServer:
    class Handler(_Handler):
        pass

    Handler.service = service
    return ThreadingHTTPServer((host, port), Handler)


def _default_engine_path() -> Path:
    return Path(__file__).resolve().parent.parent / "build" / "strata.exe"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="thin /v1/score + /v1/capabilities + /v1/decisions service over the serve line protocol",
        epilog="service options must come before --argv; everything after --argv goes to the engine")
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=None,
                        help="engine backend/protocol to drive (default: %s; the config field "
                             "'backend' is also accepted). official_q36 drives the official serve "
                             "protocol (READY/GEN/PP/T/DONE/STOP): its capability table is "
                             "synthesized and /v1/score + /v1/decisions answer an explicit 501"
                             % DEFAULT_BACKEND)
    parser.add_argument("--engine", metavar="EXE",
                        help="engine executable (default: backends/qwen35_llama/build/strata.exe)")
    parser.add_argument("--config", metavar="JSON", help="official server config (exe/cwd/args/env)")
    parser.add_argument("--argv", nargs=argparse.REMAINDER, metavar="...",
                        help="engine arguments, e.g. --argv --qwen35-model ... --serve (--serve is required)")
    parser.add_argument("--cwd", metavar="DIR", help="working directory for a started engine")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="extra engine environment (repeatable)")
    parser.add_argument("--start-engine", action="store_true", help="spawn the engine and wait for READY")
    parser.add_argument("--attach-stdin", metavar="PIPE", help="write engine commands to this pipe (attach mode)")
    parser.add_argument("--attach-stdout", metavar="PIPE", help="read engine answers from this pipe (attach mode)")
    parser.add_argument("--engine-log", metavar="PATH", help="append the started engine's stderr here")
    parser.add_argument("--tokenizer-model", metavar="GGUF",
                        help="GGUF with a byte-level BPE tokenizer, for string prompt input; without it "
                             "string prompts are refused")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="per-request engine timeout in seconds (default %(default)s)")
    parser.add_argument("--start-timeout", type=float, default=DEFAULT_START_TIMEOUT,
                        help="READY wait / attach handshake timeout in seconds (default %(default)s)")
    return parser


def _load_config(path):
    if not path:
        return {}
    try:
        config = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit("score_api: cannot read --config %r: %s" % (path, exc))
    if not isinstance(config, dict):
        raise SystemExit("score_api: --config must contain a JSON object")
    return config


def resolve_backend(cli_value, config):
    name = cli_value or config.get("backend") or DEFAULT_BACKEND
    if name not in SUPPORTED_BACKENDS:
        raise SystemExit("score_api: unknown backend %r; supported backends: %s"
                         % (name, ", ".join(SUPPORTED_BACKENDS)))
    return get_adapter(name)


def _resolve_start_command(args, config):
    exe = args.engine or config.get("exe") or str(_default_engine_path())
    cwd = args.cwd or config.get("cwd") or None
    engine_argv = list(args.argv) if args.argv else list(config.get("args") or [])
    if not engine_argv:
        raise SystemExit("score_api: --start-engine needs engine arguments (--config or --argv) including --serve")
    if "--serve" not in engine_argv:
        raise SystemExit("score_api: the engine argv must include --serve (this service drives the serve protocol)")
    env = dict(os.environ)
    config_env = config.get("env") or {}
    if not isinstance(config_env, dict):
        raise SystemExit("score_api: config 'env' must be an object")
    env.update({str(k): str(v) for k, v in config_env.items()})
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit("score_api: --env wants KEY=VALUE, got %r" % item)
        env[key] = value
    exe_path = Path(exe)
    if not exe_path.is_absolute():
        exe_path = (Path(cwd) / exe_path) if cwd else exe_path.resolve()
    return str(exe_path), engine_argv, cwd, env


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    attach = bool(args.attach_stdin or args.attach_stdout)
    if args.start_engine == attach:
        raise SystemExit("score_api: choose exactly one of --start-engine or --attach-stdin/--attach-stdout")
    if attach and not (args.attach_stdin and args.attach_stdout):
        raise SystemExit("score_api: attach mode needs both --attach-stdin and --attach-stdout")
    config = _load_config(args.config)
    backend = resolve_backend(args.backend, config)
    # Where the selection came from (reported by GET /v1/routes as backend_source): the CLI
    # flag wins over the config field; otherwise the default backend is in use.
    if args.backend is not None:
        backend_source = "cli"
    elif config.get("backend") is not None:
        backend_source = "config"
    else:
        backend_source = "default"
    proc = None
    if args.start_engine:
        exe, engine_argv, cwd, env = _resolve_start_command(args, config)
        stderr = open(args.engine_log, "ab") if args.engine_log else None
        try:
            proc = subprocess.Popen([exe] + engine_argv, cwd=cwd, env=env,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
        except OSError as exc:
            raise SystemExit("score_api: cannot start %r: %s" % (exe, exc))
        link = EngineLink(proc.stdin, proc.stdout, proc=proc)
        ready = link.wait_ready(args.start_timeout)
        backend.note_ready(ready)
        engine_path = exe
        sys.stdout.write("score_api engine ready: %s\n" % ready)
    else:
        try:
            writer = open(args.attach_stdin, "wb", buffering=0)
            reader = open(args.attach_stdout, "rb", buffering=0)
        except OSError as exc:
            raise SystemExit("score_api: cannot open the attach pipes: %s" % exc)
        link = EngineLink(writer, reader)
        if backend.caps_handshake:
            caps = link.caps(args.start_timeout)
            sys.stdout.write("score_api attached to engine (protocol %s)\n" % caps.get("protocol", "?"))
        else:
            sys.stdout.write("score_api attached to engine (backend %s: this protocol has no CAPS "
                             "handshake)\n" % backend.name)
        engine_path = args.engine
    service = ScoreService(link, engine_path=engine_path, timeout=args.timeout,
                           tokenizer_model=args.tokenizer_model, backend=backend,
                           backend_source=backend_source)
    httpd = start_http(service, args.host, args.port)
    host, port = httpd.server_address[:2]
    sys.stdout.write("score_api listening on http://%s:%d\n" % (host, port))
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if proc is not None:
            try:
                link.send("QUIT")
            except ServiceError:
                pass
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
