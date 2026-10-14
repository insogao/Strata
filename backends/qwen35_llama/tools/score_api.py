#!/usr/bin/env python3
"""score_api.py - thin HTTP scoring service over the qwen35_llama serve line protocol.

This is a *subset* service, not a fully protocol-compatible server. It drives one existing
engine process through the documented stdin/stdout protocol (`CAPS` / `SCORE`, see
docs/backends/qwen35_llama_serve_protocol.md) and exposes:

    GET  /v1/capabilities   engine CAPS passthrough + this service's own fields
    POST /v1/score          read-only top-k log-softmax of one prompt
    POST /v1/decisions      minimal choice / score / yes_no candidate scoring

One engine, one in-flight request: a concurrent request is answered 503 (no queue). There is
no chat/completions, no streaming, no batching, and the engine source is untouched. String
`prompt` input needs an optional tokenizer (--tokenizer-model); without it the request is
refused with an explanation instead of being guessed at.

Tests (0 GPU, stub engine): tools/test_score_api.py; test-only stub: tools/stub_engine.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

SERVICE_NAME = "score_api"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8097
DEFAULT_TOPK = 5
DEFAULT_TIMEOUT = 600.0
DEFAULT_START_TIMEOUT = 600.0
DEFAULT_DRAIN_TIMEOUT = 10.0
LABEL_MASS_TOLERANCE = 1e-3
MAX_BODY_BYTES = 1 << 20
DECISION_KINDS = ("choice", "score", "yes_no")

_EOF = object()


class ServiceError(Exception):
    status = 500
    source = "service"


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
    status = 502


def _timeout_message(timeout: float) -> str:
    return "the engine did not answer in time (request timeout %.1f s)" % timeout


@dataclass(frozen=True)
class ScoreResult:
    pos: int
    pairs: list
    done: dict


def _parse_logprobs(line: str):
    parts = line.split()
    if len(parts) < 4:
        raise EngineProtocolError("short LOGPROBS line: %r" % line)
    try:
        pos = int(parts[1])
        count = int(parts[2])
        pairs = []
        for token in parts[3:]:
            token_id, _, logprob = token.partition(":")
            pairs.append({"id": int(token_id), "logprob": float(logprob)})
    except ValueError as exc:
        raise EngineProtocolError("malformed LOGPROBS line: %r (%s)" % (line, exc))
    if len(pairs) != count:
        raise EngineProtocolError("LOGPROBS declares %d pairs but has %d: %r" % (count, len(pairs), line))
    return pos, pairs


def _parse_done(line: str) -> dict:
    parts = line.split()
    if len(parts) < 16:
        raise EngineProtocolError("short DONE line: %r" % line)
    try:
        return {
            "produced": int(parts[1]),
            "prompt_tokens": int(parts[2]),
            "prompt_ms": float(parts[3]),
            "decode_ms": float(parts[4]),
            "finish": parts[5],
            "reused": int(parts[8]),
            "prompt_read": int(parts[14]),
        }
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

    def score(self, ids, topk, timeout: float) -> ScoreResult:
        command = "SCORE " + ",".join(str(int(i)) for i in ids)
        if topk is not None:
            command += " topk=%d" % topk
        self.send(command)
        deadline = time.monotonic() + timeout
        pos = None
        pairs = None
        while True:
            line = self._take_until(deadline, timeout)
            if line.startswith("PP "):
                continue
            if line.startswith("LOGPROBS "):
                if pairs is not None:
                    raise EngineProtocolError("two LOGPROBS lines for one SCORE")
                pos, pairs = _parse_logprobs(line)
                continue
            if line.startswith("DONE "):
                done = _parse_done(line)
                if pairs is None:
                    raise EngineProtocolError("SCORE finished without a LOGPROBS line")
                if done["produced"] != 0 or done["finish"] != "score":
                    raise EngineProtocolError("SCORE ended with produced=%d finish=%s"
                                              % (done["produced"], done["finish"]))
                return ScoreResult(pos=pos, pairs=pairs, done=done)
            if line.startswith("ERR "):
                raise EngineError(line[4:])
            if line.startswith("T "):
                raise EngineProtocolError("SCORE generated a token (%r); the engine contract forbids it" % line)
            raise EngineProtocolError("unexpected line during SCORE: %r" % line)

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
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BadRequest("%s must be a positive integer" % what)
    return value


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
    return flat


def _usage(done: dict) -> dict:
    return {"prompt_tokens": done["prompt_tokens"], "prompt_ms": done["prompt_ms"],
            "prompt_read": done["prompt_read"]}


class ScoreService:
    def __init__(self, link: EngineLink, engine_path=None, timeout=DEFAULT_TIMEOUT,
                 drain_timeout=DEFAULT_DRAIN_TIMEOUT, tokenizer_model=None):
        self.link = link
        self.engine_path = Path(engine_path) if engine_path else None
        self.timeout = float(timeout)
        self.drain_timeout = float(drain_timeout)
        self.tokenizer_model = tokenizer_model
        self._busy = threading.Lock()
        self._dead = False
        self._sha = None
        self._tokenizer = None
        self._tokenizer_error = None
        self._tokenizer_loaded = False

    def _engine_call(self, call):
        if self._dead:
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

    def score_request(self, body):
        body = _require_object(body)
        _reject_unknown_fields(body, {"prompt_ids", "prompt", "topk"})
        if ("prompt_ids" in body) == ("prompt" in body):
            raise BadRequest("exactly one of prompt_ids or prompt is required")
        topk = _positive_int(body.get("topk", DEFAULT_TOPK), "topk")
        if "prompt" in body:
            text = body["prompt"]
            if not isinstance(text, str) or not text:
                raise BadRequest("prompt must be a non-empty string")
            ids = [int(i) for i in self._get_tokenizer().encode(text)]
            if not ids:
                raise Unprocessable("the prompt encoded to zero tokens")
        else:
            ids = _parse_prompt_ids(body["prompt_ids"])
        result = self._engine_call(lambda: self.link.score(ids, topk, self.timeout))
        return 200, {"logprobs": result.pairs, "pos": result.pos, "full_vocab_softmax": True,
                     "generated_tokens": 0, "usage": _usage(result.done)}

    def capabilities_request(self):
        caps = self._engine_call(lambda: self.link.caps(self.timeout))
        payload = dict(caps)
        payload["service"] = SERVICE_NAME
        payload["engine_sha256"] = self._engine_sha256()
        return 200, payload

    def decisions_request(self, body):
        body = _require_object(body)
        _reject_unknown_fields(body, {"kind", "prompt_ids", "candidates", "topk"})
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
            topk = _positive_int(topk, "topk")
            if topk < len(flat):
                raise Unprocessable("topk=%d cannot cover all %d candidates" % (topk, len(flat)))
        engine_topk = max(DEFAULT_TOPK, len(flat), topk or 0)
        result = self._engine_call(lambda: self.link.score(ids, engine_topk, self.timeout))
        by_id = {pair["id"]: pair["logprob"] for pair in result.pairs}
        missing = [token_id for token_id in flat if token_id not in by_id]
        if missing:
            raise Unprocessable("candidate id(s) %s are outside the engine's returned top-%d logprobs; this "
                                "subset scores only ids that appear in the distribution's top-k (the SCORE "
                                "command has no explicit-id scoring yet) - raise topk if the ids are plausible"
                                % (", ".join(str(t) for t in missing), engine_topk))
        candidates = [{"token_ids": [token_id], "logprob": by_id[token_id],
                       "probability": math.exp(by_id[token_id])} for token_id in flat]
        mass = sum(candidate["probability"] for candidate in candidates)
        payload = {"kind": kind, "candidates": candidates, "pos": result.pos, "full_vocab_softmax": True,
                   "generated_tokens": 0, "usage": _usage(result.done)}
        if abs(mass - 1.0) <= LABEL_MASS_TOLERANCE:
            payload["label_mass"] = mass
        else:
            payload["label_mass_omitted"] = (
                "the candidate set is not a closed, mutually exclusive cover of the next-token distribution "
                "(total candidate probability %.6f, tolerance %g); the per-candidate probabilities are returned "
                "un-renormalized" % (mass, LABEL_MASS_TOLERANCE))
        return 200, payload


class _Handler(BaseHTTPRequestHandler):
    service: ScoreService
    server_version = "score_api/1.0"

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/v1/capabilities":
            self._dispatch(lambda: self.service.capabilities_request())
        else:
            self._send_json(404, {"error": "not found: %s; this service implements GET /v1/capabilities, "
                                           "POST /v1/score and POST /v1/decisions" % path,
                                  "error_source": "service"})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/v1/score":
            self._dispatch(lambda: self.service.score_request(self._read_json()))
        elif path == "/v1/decisions":
            self._dispatch(lambda: self.service.decisions_request(self._read_json()))
        else:
            self._send_json(404, {"error": "not found: %s; this service implements GET /v1/capabilities, "
                                           "POST /v1/score and POST /v1/decisions" % path,
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
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BadRequest("request body is not valid UTF-8 JSON: %s" % exc)

    def _dispatch(self, action):
        try:
            status, payload = action()
        except ServiceError as exc:
            headers = {"Retry-After": "1"} if isinstance(exc, Busy) else None
            self._send_json(exc.status, {"error": str(exc), "error_source": exc.source}, headers)
        except Exception as exc:
            self._send_json(500, {"error": "internal error: %s: %s" % (type(exc).__name__, exc),
                                  "error_source": "service"})
        else:
            self._send_json(status, payload)

    def _send_json(self, status, payload, headers=None):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
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


def _resolve_start_command(args):
    config = {}
    if args.config:
        try:
            config = json.loads(Path(args.config).read_bytes().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit("score_api: cannot read --config %r: %s" % (args.config, exc))
        if not isinstance(config, dict):
            raise SystemExit("score_api: --config must contain a JSON object")
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
    proc = None
    if args.start_engine:
        exe, engine_argv, cwd, env = _resolve_start_command(args)
        stderr = open(args.engine_log, "ab") if args.engine_log else None
        try:
            proc = subprocess.Popen([exe] + engine_argv, cwd=cwd, env=env,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
        except OSError as exc:
            raise SystemExit("score_api: cannot start %r: %s" % (exe, exc))
        link = EngineLink(proc.stdin, proc.stdout, proc=proc)
        ready = link.wait_ready(args.start_timeout)
        engine_path = exe
        sys.stdout.write("score_api engine ready: %s\n" % ready)
    else:
        try:
            writer = open(args.attach_stdin, "wb", buffering=0)
            reader = open(args.attach_stdout, "rb", buffering=0)
        except OSError as exc:
            raise SystemExit("score_api: cannot open the attach pipes: %s" % exc)
        link = EngineLink(writer, reader)
        caps = link.caps(args.start_timeout)
        engine_path = args.engine
        sys.stdout.write("score_api attached to engine (protocol %s)\n" % caps.get("protocol", "?"))
    service = ScoreService(link, engine_path=engine_path, timeout=args.timeout,
                           tokenizer_model=args.tokenizer_model)
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
