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

    def require_score(self, endpoint):
        if not self.supports_score:
            raise UnsupportedBackendOperation(
                "%s is not implemented on the %s backend: its protocol (%s) has no SCORE "
                "command, so next-token logprobs cannot be read. This service refuses "
                "explicitly instead of substituting another operation (it never degrades %s "
                "into generation). Use the qwen35_llama backend for %s."
                % (endpoint, self.name, "/".join(self.client_commands), endpoint, endpoint))

    def capabilities(self, link, timeout):
        raise NotImplementedError

    def score(self, link, ids, topk, timeout, candidates=None):
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

    def score(self, link, ids, topk, timeout, candidates=None):
        return link.score(ids, topk, timeout, candidates=candidates)


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
