"""One opt-in local API, one executor and one ownership lock. Text-only MVP.

Chat is stateless over the same engine, not cross-engine prefix reuse. Existing
Score/Decisions/Session contracts are inherited. Streaming/batching are refused.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import uuid
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

import score_api as api


class UnifiedService(api.ScoreService):
    model_alias = 'strata-qwen35'

    def chat_ready(self):
        return self.backend.supports_score and bool(self.tokenizer_model or self._tokenizer)

    def chat_request(self, body):
        body = api._require_object(body)
        api._reject_unknown_fields(body, {'model', 'messages', 'prompt_ids', 'max_tokens',
                                         'stream', 'temperature', 'session'})
        if not self.chat_ready():
            raise api.NotImplementedFeature('unified Chat needs qwen35_llama and its GGUF tokenizer; no backend substitution')
        if body.get('model', self.model_alias) != self.model_alias:
            raise api.Unprocessable('unknown model; this process owns exactly one model')
        if type(body.get('stream', False)) is not bool:
            raise api.BadRequest('stream must be a boolean')
        if body.get('stream'):
            raise api.NotImplementedFeature('streaming Chat is not implemented in this unified MVP')
        if type(body.get('temperature', 0)) not in (int, float) or body.get('temperature', 0) != 0:
            raise api.Unprocessable('unified Chat currently supports greedy temperature=0 only')
        maximum = api._positive_int(body.get('max_tokens', 128), 'max_tokens')
        if ('messages' in body) == ('prompt_ids' in body):
            raise api.BadRequest('provide exactly one of messages or prompt_ids')
        tokenizer = self._get_tokenizer()
        if 'prompt_ids' in body:
            ids = api._parse_prompt_ids(body['prompt_ids'])
        else:
            messages = body['messages']
            if not isinstance(messages, list) or not messages:
                raise api.BadRequest('messages must be a nonempty text-message array')
            for m in messages:
                if (not isinstance(m, dict) or set(m) != {'role', 'content'} or
                        m['role'] not in {'system', 'user', 'assistant'} or not isinstance(m['content'], str)):
                    raise api.BadRequest('only system/user/assistant text messages are supported')
            ids = self.encode_messages(messages, tokenizer)
        if not ids:
            raise api.Unprocessable('empty encoded prompt')

        def call():
            # Clear any explicit old workflow only after the ownership/revision
            # check. Private SCORE→GEN→reset holds the same busy lock throughout.
            try:
                self.backend.score(self.link, [0], 1, self.timeout)
                scored = self.backend.score(self.link, ids, 1, self.timeout, keep=True)
                generated = self.link.generate_continue(maximum, [], self.timeout)
                self.backend.score(self.link, [0], 1, self.timeout)
                return scored, generated
            except api.EngineProtocolError:
                raise  # never resynchronize a corrupted pipe
            except api.EngineTimeout as exc:
                raise api.EngineProtocolError('Chat timed out with private state uncertain: ' + str(exc))
            except api.EngineError:
                # ERR is a complete framed refusal. Clear private armed state before
                # returning it; a failed reset is unsafe and quarantines the link.
                try:
                    self.backend.score(self.link, [0], 1, self.timeout)
                except api.ServiceError as exc:
                    raise api.EngineProtocolError('Chat cleanup failed: ' + str(exc))
                raise
        (scored, generated), lease = self._owned_call(body, call, False)
        tokens = generated.tokens
        # Engine EOS is included in produced; do not render that control literal.
        visible = tokens[:-1] if generated.done['finish'] == 'stop' else tokens
        return 200, {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion',
            'created': int(time.time()), 'model': self.model_alias,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': tokenizer.decode(visible)},
                         'finish_reason': generated.done['finish']}],
            'usage': {'prompt_tokens': len(ids), 'completion_tokens': len(tokens),
                      'total_tokens': len(ids) + len(tokens)},
            'session': lease, 'prefix_reused': False,
            'engine_timing': {'prefill_ms': scored.done['prompt_ms'], 'decode_ms': generated.done['decode_ms']}}

    def encode_messages(self, messages, tokenizer):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from gguf_reader import GGUFFile
        if not hasattr(self, '_chat_template'):
            source = GGUFFile(Path(self.tokenizer_model)).metadata.get('tokenizer.chat_template')
            if not isinstance(source, str):
                raise api.NotImplementedFeature('GGUF contains no chat template')
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                               extensions=['jinja2.ext.loopcontrols'])
            def refuse(message):
                raise api.Unprocessable('chat template refused input: ' + str(message))
            env.globals['raise_exception'] = refuse
            self._chat_template = env.from_string(source)
        text = self._chat_template.render(messages=messages, tools=None, add_generation_prompt=True,
                                          enable_thinking=False)
        # Control literals written by callers are rejected rather than silently
        # turning user text into extra template roles (text-only MVP).
        controls = tokenizer.control_tokens
        if any(c and c in m['content'] for c in controls for m in messages):
            raise api.BadRequest('literal model control markers in message content are not supported')
        return tokenizer.encode(text, parse_special=True)

    def capabilities_request(self):
        status, payload = super().capabilities_request()
        payload.update({'service': 'strata-unified-mvp/1', 'chat_completions': self.chat_ready(),
                        'chat_streaming': False, 'chat_disconnect_cancel': False,
                        'chat_prefix_reuse': False, 'shared_executor': True,
                        'model_alias': self.model_alias})
        return status, payload

    def routes_request(self):
        status, payload = super().routes_request()
        for row in payload['routes']:
            if row['path'] == '/v1/chat/completions':
                supported = row['backend'] == self.backend.name and self.chat_ready()
                row.update({'in_service': True, 'supported': supported,
                    'engine_command': 'SCORE keep=1; GEN cont=1; reset' if supported else None,
                    'refusal_status': None if supported else 501,
                    'refusal_text': None if supported else 'unified Chat backend/tokenizer unavailable',
                    'notes': 'one executor; stateless Chat; active score workflow requires its current owner handle'})
        payload['service'] = 'strata-unified-mvp/1'
        return status, payload


def start_http(service, host='127.0.0.1', port=8080):
    if host not in {'127.0.0.1', 'localhost'}:
        raise ValueError('unified MVP has no authentication; loopback only')
    class Handler(api._Handler):
        def do_POST(self):
            if urlsplit(self.path).path == '/v1/chat/completions':
                self._dispatch(lambda: self.service.chat_request(self._read_json()))
            else:
                super().do_POST()
        def do_GET(self):
            if urlsplit(self.path).path == '/v1/models':
                self._send_json(200, {'object': 'list', 'data': [{'id': self.service.model_alias,
                    'object': 'model', 'owned_by': 'local-strata'}]})
            else:
                super().do_GET()
    Handler.service = service
    return ThreadingHTTPServer((host, port), Handler)


def leased_link(native):
    """EngineLink sees only protocol rows; original stdout+stderr is never altered."""
    class Writer:
        def write(self, data): native.send(data.decode('ascii').rstrip('\n'))
        def flush(self): pass
        def close(self):
            if native.pipe is not None:
                native.send('STOP'); native.send('QUIT')
    class Reader:
        closed = False
        def __iter__(self):
            while not self.closed:
                try:
                    row, _ = native.line(native.started + native.spec['budget']['native_timeout_s'])
                except (ValueError, TimeoutError):
                    return
                if row.startswith(('READY ', 'CAPS ', 'STATE ', 'SAVED ', 'LOADED ', 'PP ', 'T ',
                                   'LOGPROBS ', 'CANDLOGPROBS ', 'DONE ', 'ERR ')):
                    yield (row + '\n').encode('utf-8')
                elif row and not row.startswith(('strata:', 'strata qwen35:', 'serve:',
                                                   'llama_', 'ggml_', '~llama_context:')):
                    # Only known native diagnostics are filtered from merged stderr.
                    # Unknown bytes reach EngineLink and fail its protocol boundary.
                    yield (row + '\n').encode('utf-8')
        def close(self): self.closed = True
    return api.EngineLink(Writer(), Reader(), proc=native.proc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--lifetime', type=int, default=300, help='finite prototype lifetime,30..900 seconds')
    a = ap.parse_args(argv)
    if a.host not in {'127.0.0.1', 'localhost'} or not 30 <= a.lifetime <= 900:
        ap.error('loopback only; finite lifetime30..900 required')
    import strata_experiment as ex
    from leased_native import LeasedNative
    spec = ex.st.read_json(a.spec); ex.validate(spec); fp = ex.fingerprint(spec)
    if not ex.admission(spec, fp, spec['baseline']['config'])['admitted']:
        raise ValueError('runtime resource budget refused')
    # Explicit foreground lifetime; no infinite daemon or silent profile install.
    spec['budget']['native_timeout_s'] = a.lifetime
    native = LeasedNative(spec, spec['baseline']['config'], a.out, time.time())
    link = None
    httpd = None
    try:
        link = leased_link(native)
        service = UnifiedService(link, engine_path=spec['engine']['path'], timeout=60,
                tokenizer_model=str(Path(spec['cwd']) / spec['model']['path']), backend='qwen35_llama',
                backend_source='unified-config')
        httpd = start_http(service, a.host, a.port)
        httpd.timeout = .5
        print(json.dumps({'url': 'http://%s:%d' % httpd.server_address, 'shared_executor': True,
                          'lease_job': native.job_id, 'lifetime_s': a.lifetime}), flush=True)
        until = native.started + a.lifetime - 10
        while time.monotonic() < until and not link.dead:
            httpd.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        if httpd is not None:
            httpd.server_close()
        try:
            native.close(abort=True)
        finally:
            if link is not None:
                link.close()
    native.receipt()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
