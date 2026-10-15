#!/usr/bin/env python3
"""backend_adapters.py - framework-level backend selection for score_api.py (P2-7).

The service drives one engine process over a stdin/stdout line protocol. Two backends are
selectable, each with its own command surface:

    qwen35_llama   the in-repo migrated backend (docs/backends/qwen35_llama_serve_protocol.md)
                   client -> engine: CAPS / SCORE / GEN / SAVE / LOAD / STATE / STOP / QUIT
                   engine -> client: READY / PP / T / LOGPROBS / DONE / CAPS / SAVED / LOADED /
                                    STATE / ERR

    official_q36   the official q36 engine's own `--serve` protocol, the one `serve/server.py`
                   drives (client -> engine: GEN / STOP / QUIT; engine -> client:
                   READY / PP / T / DONE)

The capability table is composed per backend and must not pretend the two are equivalent.
The official protocol has no CAPS command, so its table is synthesized here from protocol
facts - logits_read and state_export are false (no SCORE, no SAVE/LOAD/STATE) - while
qwen35_llama's table is the engine's own CAPS passthrough plus the framework fields added
here. Model-specific computation stays in the backends; selection, capability reporting and
refusals live in this layer.

Refusal discipline: an operation the selected backend has no command for raises
UnsupportedBackendOperation (score_api maps it to HTTP 501 with the reason). Nothing is
silently substituted - in particular /v1/score is never degraded into generation.

P2R-5 additions (routing and capability contract): `ROUTES` is the declarative
(endpoint, backend) -> cell matrix that replaces the hand-written `require_*` gates:

    cell = {"supported": bool,               # can this backend do it?
            "engine_command": str | None,    # the protocol command line it maps to
            "refusal_status": 501 | None,    # status of the refusal when unsupported
            "refusal_text_template": str|None,  # {path}/{backend}/{commands} template
            "requires": str | None,          # CAPS field the supported contract needs
            "in_service": bool,              # is it an HTTP route of score_api at all?
            "notes": str}                    # what the cell does and does not cover

An in-service cell that is unsupported on a backend answers 501 with that cell's own
rendered text (byte-identical to the hand-written pre-P2R-5 texts on official_q36, pinned
by tests); `in_service=false` rows are boundary facts only (the chat endpoint is driven by
the official `serve/server.py`; engine-level SAVE/LOAD are commands, not HTTP routes).
`route()`, `capabilities_for()` and `refusal_text()` are the query surface;
`routes_payload()` renders the whole matrix for score_api's read-only `GET /v1/routes`.
score_api.py consults the matrix for every endpoint; it never decides support by hand.
`requires` is a precondition on the engine's own CAPS declaration (the engine contract),
not a runtime check this round performs.

Batching: neither backend is driven concurrently in this phase; both capability tables
report `batching=false` and carry `batching_contract` = the version of
docs/backends/qwen35_llama_batching_contract.md. See that document for the contract a
future implementation must satisfy before batching can be reported true.

Tests (0 GPU, one stub engine per protocol): tools/test_backend_adapters.py.
"""
from __future__ import annotations

BATCHING_CONTRACT_VERSION = "1"
DEFAULT_BACKEND = "qwen35_llama"
SUPPORTED_BACKENDS = ("qwen35_llama", "official_q36")


class UnsupportedBackendOperation(Exception):
    """The selected backend's protocol has no command for the requested operation."""


class BackendAdapter:
    name = ""
    protocol = ""
    client_commands = ()
    engine_lines = ()
    supports_score = False
    caps_handshake = False

    def note_ready(self, ready_line):
        """Observe the engine's READY line (start mode); a backend may record facts from it."""

    def capabilities(self, link, timeout):
        raise NotImplementedError

    def score(self, link, ids, topk, timeout, candidates=None, keep=False):
        raise UnsupportedBackendOperation("the %s backend has no SCORE command" % self.name)


class Qwen35LlamaAdapter(BackendAdapter):
    name = "qwen35_llama"
    protocol = "strata-serve-v1"
    client_commands = ("CAPS", "SCORE", "GEN", "SAVE", "LOAD", "STATE", "STOP", "QUIT")
    engine_lines = ("READY", "PP", "T", "LOGPROBS", "CANDLOGPROBS", "DONE", "CAPS", "SAVED",
                    "LOADED", "STATE", "ERR")
    supports_score = True
    caps_handshake = True

    def capabilities(self, link, timeout):
        # The engine's own CAPS table, passed through unchanged; the framework adds only
        # the backend identity and, while batching is false, the contract version.
        payload = dict(link.caps(timeout))
        payload["backend"] = self.name
        if payload.get("batching") is False:
            payload["batching_contract"] = BATCHING_CONTRACT_VERSION
        return payload

    def score(self, link, ids, topk, timeout, candidates=None, keep=False):
        return link.score(ids, topk, timeout, candidates=candidates, keep=keep)


class OfficialQ36Adapter(BackendAdapter):
    name = "official_q36"
    protocol = "official-q36-serve"
    client_commands = ("GEN", "STOP", "QUIT")
    engine_lines = ("READY", "PP", "T", "DONE")
    supports_score = False
    caps_handshake = False

    def __init__(self):
        self.context_max = None

    def note_ready(self, ready_line):
        # READY <ctx> stop - the protocol's only source of the context size.
        parts = ready_line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            self.context_max = int(parts[1])

    def capabilities(self, link, timeout):
        # Synthesized by the framework: this protocol has no CAPS command, so there is no
        # engine-reported table to pass through. Every value below is a protocol fact
        # (GEN/PP/T/DONE/STOP exist) or an explicit non-claim (false) - it is not the
        # qwen35_llama table. Fields this protocol cannot report (prefix_cache,
        # state_isolation, mtp, expert_formats, ...) are deliberately absent: absence is
        # not a claim in either direction, and the official engine's own extensions
        # (GENI, BGEN, RESTORE/cache, INFO batch_slots) are outside the surface this
        # adapter drives.
        return {
            "backend": self.name,
            "protocol": self.protocol,
            "capability_source": "framework-synthesized-from-protocol-facts",
            "generation": True,
            "continuous_requests": True,
            "cancel": True,
            "prefill_progress": True,
            "token_stream": True,
            "context_max": self.context_max,
            "batching": False,
            "batching_contract": BATCHING_CONTRACT_VERSION,
            "logits_read": False,
            "state_export": False,
        }


_ADAPTERS = {adapter.name: adapter for adapter in (Qwen35LlamaAdapter, OfficialQ36Adapter)}


def get_adapter(name):
    if name not in _ADAPTERS:
        raise ValueError("unknown backend %r; supported backends: %s"
                         % (name, ", ".join(sorted(_ADAPTERS))))
    return _ADAPTERS[name]()


# ---------------------------------------------------------------------------------------------
# P2R-5: declarative routing/capability matrix.
#
# One matrix row per (endpoint, backend). score_api.py never decides support by hand: every
# endpoint gate is route(endpoint, active_backend), and an unsupported in-service cell raises
# NotImplementedFeature with this cell's rendered refusal_text (byte-identical on official_q36
# to the pre-P2R-5 hand-written texts). `requires` names the engine CAPS field the supported
# contract rests on (the engine-side precondition, documented; this layer does not enforce it -
# see docs/backends/qwen35_llama_backend_adapters.md).

SCORE_ENDPOINT = "POST /v1/score"
SCORE_CANDIDATES_ENDPOINT = "POST /v1/score:candidates"
SCORE_KEEP_ENDPOINT = "POST /v1/score:keep"
DECISIONS_ENDPOINT = "POST /v1/decisions"
CONTINUE_ENDPOINT = "POST /v1/continue"
SESSION_ENDPOINT = "GET /v1/session"
CAPABILITIES_ENDPOINT = "GET /v1/capabilities"
ROUTES_ENDPOINT = "GET /v1/routes"
ENGINE_STATE_ENDPOINT = "engine SAVE/LOAD"
CHAT_ENDPOINT = "chat /v1/chat/completions"

ENDPOINTS = (SCORE_ENDPOINT, SCORE_CANDIDATES_ENDPOINT, SCORE_KEEP_ENDPOINT, DECISIONS_ENDPOINT,
             CONTINUE_ENDPOINT, SESSION_ENDPOINT, CAPABILITIES_ENDPOINT, ROUTES_ENDPOINT,
             ENGINE_STATE_ENDPOINT, CHAT_ENDPOINT)

# The HTTP path a matrix endpoint id maps to (None: not an HTTP route of this service).
ENDPOINT_PATHS = {
    SCORE_ENDPOINT: "/v1/score",
    SCORE_CANDIDATES_ENDPOINT: "/v1/score",
    SCORE_KEEP_ENDPOINT: "/v1/score",
    DECISIONS_ENDPOINT: "/v1/decisions",
    CONTINUE_ENDPOINT: "/v1/continue",
    SESSION_ENDPOINT: "/v1/session",
    CAPABILITIES_ENDPOINT: "/v1/capabilities",
    ROUTES_ENDPOINT: "/v1/routes",
    ENGINE_STATE_ENDPOINT: None,
    CHAT_ENDPOINT: "/v1/chat/completions",
}

SCORE_REFUSAL_TEMPLATE = (
    "{path} is not implemented on the {backend} backend: its protocol ({commands}) has no SCORE "
    "command, so next-token logprobs cannot be read. This service refuses explicitly instead of "
    "substituting another operation (it never degrades {path} into generation). Use the "
    "qwen35_llama backend for {path}.")
SESSION_REFUSAL_TEMPLATE = (
    "{path} is not implemented on the {backend} backend: its protocol ({commands}) has no session "
    "commands (no STATE, no 'GEN ... cont=1'), so a session can be neither continued nor reported. "
    "This service refuses explicitly instead of substituting another operation. Use the "
    "qwen35_llama backend for {path}.")

_SCORE_NOTES = ("no SCORE command in this protocol: logprobs cannot be read; refused explicitly, "
                "never degraded into generation")
_SESSION_NOTES = ("no session commands (no STATE, no 'GEN ... cont=1'): a session can be neither "
                  "continued nor reported")

_ROUTE_TABLE = {
    SCORE_ENDPOINT: {
        "qwen35_llama": {
            "supported": True,
            "engine_command": "SCORE <id,id,...> topk=N",
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": None,
            "in_service": True,
            "notes": "read-only top-k log-softmax over the prompt; never generates tokens",
        },
        "official_q36": {
            "supported": False,
            "engine_command": None,
            "refusal_status": 501,
            "refusal_text_template": SCORE_REFUSAL_TEMPLATE,
            "requires": None,
            "in_service": True,
            "notes": _SCORE_NOTES,
        },
    },
    SCORE_CANDIDATES_ENDPOINT: {
        "qwen35_llama": {
            "supported": True,
            "engine_command": "SCORE <id,id,...> topk=N cand=<id,id,...>",
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": "score_explicit_candidate_ids",
            "in_service": True,
            "notes": "explicit candidate ids are answered whatever their top-k rank; topk=0 answers "
                     "candidates only; needs the engine's score_explicit_candidate_ids",
        },
        "official_q36": {
            "supported": False,
            "engine_command": None,
            "refusal_status": 501,
            "refusal_text_template": SCORE_REFUSAL_TEMPLATE,
            "requires": None,
            "in_service": True,
            "notes": _SCORE_NOTES,
        },
    },
    SCORE_KEEP_ENDPOINT: {
        "qwen35_llama": {
            "supported": True,
            "engine_command": "SCORE <id,id,...> topk=N [cand=<id,id,...>] keep=1",
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": "state_keep_score",
            "in_service": True,
            "notes": "state-keeping score: no reset, appends at an armed session's cursor and stays "
                     "armed; needs the engine's state_keep_score",
        },
        "official_q36": {
            "supported": False,
            "engine_command": None,
            "refusal_status": 501,
            "refusal_text_template": SCORE_REFUSAL_TEMPLATE,
            "requires": None,
            "in_service": True,
            "notes": _SCORE_NOTES,
        },
    },
    DECISIONS_ENDPOINT: {
        "qwen35_llama": {
            "supported": True,
            "engine_command": "SCORE <id,id,...> topk=0 cand=<id,id,...> [keep=1]",
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": "score_explicit_candidate_ids",
            "in_service": True,
            "notes": "minimal choice/score/yes_no candidate scoring; the optional keep=1 additionally "
                     "needs the engine's state_keep_score",
        },
        "official_q36": {
            "supported": False,
            "engine_command": None,
            "refusal_status": 501,
            "refusal_text_template": SCORE_REFUSAL_TEMPLATE,
            "requires": None,
            "in_service": True,
            "notes": _SCORE_NOTES,
        },
    },
    CONTINUE_ENDPOINT: {
        "qwen35_llama": {
            "supported": True,
            "engine_command": "GEN <max_new> cont=1 [<id,id,...>]",
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": "session_continue",
            "in_service": True,
            "notes": "greedy continuation of an armed session without re-prefilling the carried "
                     "prefix; needs the engine's session_continue and an armed session at run time",
        },
        "official_q36": {
            "supported": False,
            "engine_command": None,
            "refusal_status": 501,
            "refusal_text_template": SESSION_REFUSAL_TEMPLATE,
            "requires": None,
            "in_service": True,
            "notes": _SESSION_NOTES,
        },
    },
    SESSION_ENDPOINT: {
        "qwen35_llama": {
            "supported": True,
            "engine_command": "STATE",
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": None,
            "in_service": True,
            "notes": "read-only session bookkeeping (armed/pos/origin); one STATE query, never resets "
                     "or arms",
        },
        "official_q36": {
            "supported": False,
            "engine_command": None,
            "refusal_status": 501,
            "refusal_text_template": SESSION_REFUSAL_TEMPLATE,
            "requires": None,
            "in_service": True,
            "notes": _SESSION_NOTES,
        },
    },
    CAPABILITIES_ENDPOINT: {
        "qwen35_llama": {
            "supported": True,
            "engine_command": "CAPS",
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": None,
            "in_service": True,
            "notes": "engine CAPS passthrough plus the framework fields; the service adds "
                     "service/engine_sha256/active_backend",
        },
        "official_q36": {
            "supported": True,
            "engine_command": None,
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": None,
            "in_service": True,
            "notes": "supported, but synthesized by the framework: this protocol has no CAPS "
                     "handshake, so there is no engine table to pass through",
        },
    },
    ROUTES_ENDPOINT: {
        "qwen35_llama": {
            "supported": True,
            "engine_command": None,
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": None,
            "in_service": True,
            "notes": "framework-only: rendered from this matrix, no engine command is sent, "
                     "read-only even on a dead link",
        },
        "official_q36": {
            "supported": True,
            "engine_command": None,
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": None,
            "in_service": True,
            "notes": "framework-only: rendered from this matrix, no engine command is sent, "
                     "read-only even on a dead link",
        },
    },
    ENGINE_STATE_ENDPOINT: {
        "qwen35_llama": {
            "supported": True,
            "engine_command": "SAVE <path> / LOAD <path>",
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": None,
            "in_service": False,
            "notes": "engine-level single-session export/import (container v1, magic STRATAST; see "
                     "docs/backends/qwen35_llama_session_state.md); not an HTTP route of this service",
        },
        "official_q36": {
            "supported": False,
            "engine_command": None,
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": None,
            "in_service": False,
            "notes": "unsupported on the surface this adapter drives; the official engine's own "
                     "RESTORE/SERR session grammar is a different protocol surface, not driven or "
                     "changed this round",
        },
    },
    CHAT_ENDPOINT: {
        "qwen35_llama": {
            "supported": False,
            "engine_command": None,
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": None,
            "in_service": False,
            "notes": "not an HTTP route of this service: chat is driven by the official "
                     "serve/server.py over the same engine exe (single-entry recipe in "
                     "docs/backends/qwen35_llama_backend_adapters.md); this service answers 404",
        },
        "official_q36": {
            "supported": False,
            "engine_command": None,
            "refusal_status": None,
            "refusal_text_template": None,
            "requires": None,
            "in_service": False,
            "notes": "not an HTTP route of this service: the official server is the chat entry "
                     "(built-in path) for this protocol (single-entry recipe); this service answers "
                     "404",
        },
    },
}

ROUTES = {(endpoint, backend): dict(cell)
          for endpoint, backends in _ROUTE_TABLE.items()
          for backend, cell in backends.items()}
ROUTE_CELL_KEYS = ("supported", "engine_command", "refusal_status", "refusal_text_template",
                   "requires", "in_service", "notes")


def route(endpoint, backend):
    """The matrix cell for (endpoint, backend); unknown endpoints/backends are refused."""
    if endpoint not in ENDPOINT_PATHS:
        raise ValueError("unknown endpoint %r; known endpoints: %s"
                         % (endpoint, ", ".join(ENDPOINTS)))
    get_adapter(backend)  # unknown backend -> ValueError with the supported list
    return dict(ROUTES[(endpoint, backend)])


def refusal_text(endpoint, backend):
    """Render the cell's 501 text for an unsupported (endpoint, backend) pair."""
    cell = route(endpoint, backend)
    if cell["refusal_text_template"] is None:
        raise ValueError("endpoint %r is not refused on backend %r (supported=%r): it has no "
                         "refusal text"
                         % (endpoint, backend, cell["supported"]))
    adapter = get_adapter(backend)
    return cell["refusal_text_template"].format(
        path=ENDPOINT_PATHS.get(endpoint) or endpoint,
        backend=adapter.name,
        commands="/".join(adapter.client_commands))


def capabilities_for(backend):
    """The static capability contract this layer declares for `backend` (no engine involved).

    The keys mirror the engine-CAPS vocabulary where the matrix can answer it statically:
    score / score_explicit_candidate_ids / state_keep_score / session_continue / STATE /
    state_export. This is the declarative sibling of BackendAdapter.capabilities(), which
    needs a live engine link.
    """
    adapter = get_adapter(backend)
    cells = {endpoint: route(endpoint, adapter.name) for endpoint in ENDPOINTS}
    return {
        "backend": adapter.name,
        "protocol": adapter.protocol,
        "caps_handshake": adapter.caps_handshake,
        "client_commands": list(adapter.client_commands),
        "score": cells[SCORE_ENDPOINT]["supported"],
        "score_explicit_candidate_ids": cells[SCORE_CANDIDATES_ENDPOINT]["supported"],
        "state_keep_score": cells[SCORE_KEEP_ENDPOINT]["supported"],
        "session_continue": cells[CONTINUE_ENDPOINT]["supported"],
        "STATE": "STATE" in adapter.client_commands,
        "state_export": cells[ENGINE_STATE_ENDPOINT]["supported"],
        "routes": {endpoint: cell["supported"] for endpoint, cell in cells.items()},
    }


def routes_payload():
    """The full matrix as a JSON-serializable list, one item per (endpoint, backend) cell."""
    payload = []
    for endpoint in ENDPOINTS:
        for backend in SUPPORTED_BACKENDS:
            cell = ROUTES[(endpoint, backend)]
            payload.append({
                "endpoint": endpoint,
                "path": ENDPOINT_PATHS.get(endpoint),
                "backend": backend,
                "in_service": cell["in_service"],
                "supported": cell["supported"],
                "engine_command": cell["engine_command"],
                "requires": cell["requires"],
                "refusal_status": cell["refusal_status"],
                "refusal_text": refusal_text(endpoint, backend)
                                if cell["refusal_text_template"] is not None else None,
                "notes": cell["notes"],
            })
    return payload
