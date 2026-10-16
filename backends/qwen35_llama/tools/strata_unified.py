"""One opt-in local API, one executor and one ownership lock. Text-only MVP.

Chat is stateless over the same engine, not cross-engine prefix reuse. Existing
Score/Decisions/Session contracts are inherited. Cooperative SSE is opt-in;
multi-sequence batching is refused.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import uuid
import select
import socket
import signal
import traceback
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

import score_api as api
from strata_streaming import CooperativeLink, ScoreCancelled


class UnifiedService(api.ScoreService):
    model_alias = 'strata-qwen35'

    def __init__(self,*args,**kwargs):
        self._http_context=threading.local()
        super().__init__(*args,**kwargs)

    def _engine_call(self,call):
        def bounded():
            deadline=time.monotonic()+self.timeout
            if getattr(self,'service_deadline',None) is not None:
                deadline=min(deadline,self.service_deadline-10)
            if deadline<=time.monotonic():raise api.Busy('service is draining; restart after lease completion')
            self.link.request_deadline=deadline
            peer=getattr(self._http_context,'peer',None)
            mutating=getattr(self._http_context,'mutating',False)
            self.link.request_cancelled=peer if mutating else None
            def discard_abandoned():
                # Still inside the same pipe/busy lock, after ownership checks.
                # A response-lost keep=1 must not strand an unreachable lease.
                self.link.request_cancelled=None
                try:self.backend.score(self.link,[0],1,self.timeout)
                except api.ServiceError as exc:
                    raise api.EngineProtocolError('abandoned request reset failed: '+str(exc))
                self._session_lease=None;self._session_uncertain=False
            try:
                try:result=call()
                except ScoreCancelled:
                    if mutating and peer and peer():discard_abandoned()
                    raise
                if mutating and peer and peer():discard_abandoned()
                return result
            finally:
                self.link.request_deadline=None;self.link.request_cancelled=None
        return super()._engine_call(bounded)

    def chat_ready(self):
        return self.backend.supports_score and bool(self.tokenizer_model or self._tokenizer)

    def chat_request(self, body, on_text=None, cancelled=None):
        body = api._require_object(body)
        api._reject_unknown_fields(body, {'model', 'messages', 'prompt_ids', 'max_tokens',
                                         'stream', 'temperature', 'session'})
        if not self.chat_ready():
            raise api.NotImplementedFeature('unified Chat needs qwen35_llama and its GGUF tokenizer; no backend substitution')
        if body.get('model', self.model_alias) != self.model_alias:
            raise api.Unprocessable('unknown model; this process owns exactly one model')
        if type(body.get('stream', False)) is not bool:
            raise api.BadRequest('stream must be a boolean')
        if body.get('stream') and on_text is None:
            raise api.BadRequest('streaming requires the HTTP SSE entry')
        if body.get('stream') and not isinstance(self.link,CooperativeLink):
            raise api.NotImplementedFeature('this attached link has no cooperative streaming/cancel support')
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

        emitted, streamed = '', []
        peer_cancelled = cancelled or (lambda:False)
        def cancelled():
            return peer_cancelled() or bool(getattr(self,'stopping',None) and self.stopping.is_set())
        def token_callback(token):
            nonlocal emitted
            controls = {getattr(tokenizer,'special_tokens',{}).get(s) for s in getattr(tokenizer,'control_tokens',[])}
            if token not in controls: streamed.append(token)
            text = tokenizer.decode(streamed,errors='ignore')
            if not text.startswith(emitted):
                raise api.EngineProtocolError('incremental tokenizer rewrote emitted text')
            delta=text[len(emitted):];emitted=text
            if delta and on_text is not None:on_text(delta)
        def call():
            # Clear any explicit old workflow only after the ownership/revision
            # check. Private SCORE→GEN→reset holds the same busy lock throughout.
            try:
                self.backend.score(self.link, [0], 1, self.timeout)
                try:
                    scored = self.link.score_interruptible(ids,self.timeout,cancelled) if isinstance(self.link,CooperativeLink) else self.backend.score(self.link, ids, 1, self.timeout, keep=True)
                except ScoreCancelled:
                    self.backend.score(self.link,[0],1,self.timeout)
                    return None,None
                if cancelled():
                    self.backend.score(self.link,[0],1,self.timeout)
                    return None,None
                generated = self.link.generate_interruptible(maximum,self.timeout,token_callback if on_text else None,cancelled) if isinstance(self.link,CooperativeLink) else self.link.generate_continue(maximum, [], self.timeout)
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
        if generated is None:
            return 200, {'cancelled':True,'session':lease}
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
                        'chat_streaming': isinstance(self.link,CooperativeLink),
                        'chat_disconnect_cancel': isinstance(self.link,CooperativeLink),
                        'queue_policy':'no queue; busy503 Retry-After1',
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
        def setup(self):
            super().setup();self.connection.settimeout(5)

        def disconnected(self):
            try:
                readable,_,_=select.select([self.connection],[],[],0)
                return bool(readable and self.connection.recv(1,socket.MSG_PEEK)==b'')
            except OSError:
                return True

        def do_POST(self):
            if urlsplit(self.path).path == '/v1/chat/completions':
                self._dispatch(self.chat)
            else:
                super().do_POST()

        def chat(self):
            body=self._read_json()
            if not isinstance(body,dict) or body.get('stream') is not True:
                return self.service.chat_request(body,cancelled=self.disconnected)
            begun=False;identifier='chatcmpl-'+uuid.uuid4().hex
            def frame(payload):
                nonlocal begun
                if not begun:
                    self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8')
                    self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers()
                    self.close_connection=True;begun=True
                self.wfile.write(('data: '+json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n\n').encode('utf-8'));self.wfile.flush()
            def text(delta):
                frame({'id':identifier,'object':'chat.completion.chunk','model':self.service.model_alias,
                       'choices':[{'index':0,'delta':{'content':delta},'finish_reason':None}]})
            try:
                status,payload=self.service.chat_request(body,on_text=text,cancelled=self.disconnected)
            except api.ServiceError as exc:
                if not begun:raise
                try:frame({'error':str(exc),'error_source':exc.source,'quarantined':exc.quarantined})
                except OSError:pass
                return None
            if not self.disconnected() and not payload.get('cancelled'):
                try:
                    frame({'id':identifier,'object':'chat.completion.chunk','model':self.service.model_alias,
                        'choices':[{'index':0,'delta':{},'finish_reason':payload['choices'][0]['finish_reason']}],
                        'usage':payload['usage']})
                    self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
                except OSError:pass
            return None

        def _dispatch(self, action):
            # SSE owns its response after the first frame; do not append JSON.
            def wrapped():
                self.service._http_context.peer=self.disconnected
                self.service._http_context.mutating=(self.command=='POST' and urlsplit(self.path).path in {'/v1/score','/v1/decisions','/v1/continue'})
                try:result=action()
                except api.ServiceError:raise
                except Exception:
                    traceback.print_exc(file=sys.stderr);raise
                finally:
                    self.service._http_context.peer=None
                    self.service._http_context.mutating=False
                return result if result is not None else (None,None)
            super()._dispatch(wrapped)

        def _send_json(self,status,payload,headers=None):
            if status is None:return
            try:super()._send_json(status,payload,headers)
            except OSError:
                lease=payload.get('session') if isinstance(payload,dict) else None
                if lease is not None and self.command=='POST':
                    try:
                        self.service._owned_call({'session':lease},
                            lambda:self.service.backend.score(self.service.link,[0],1,self.service.timeout),False)
                    except api.ServiceError:
                        # A busy or advanced revision belongs to another operation;
                        # never reset it using this stale response's authority.
                        pass
        def do_GET(self):
            if urlsplit(self.path).path == '/v1/models':
                self._send_json(200, {'object': 'list', 'data': [{'id': self.service.model_alias,
                    'object': 'model', 'owned_by': 'local-strata'}]})
            else:
                super().do_GET()
    Handler.service = service
    httpd=ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads=False
    return httpd


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
    return CooperativeLink(Writer(), Reader(), proc=native.proc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--lifetime', type=int, default=300, help='explicit finite foreground lease,30..86400 seconds')
    a = ap.parse_args(argv)
    if a.host not in {'127.0.0.1', 'localhost'} or not 30 <= a.lifetime <= 86400:
        ap.error('loopback only; explicit finite lifetime30..86400 required')
    import strata_experiment as ex
    from leased_native import LeasedNative
    spec = ex.st.read_json(a.spec); ex.validate(spec); fp = ex.fingerprint(spec)
    if not ex.admission(spec, fp, spec['baseline']['config'])['admitted']:
        raise ValueError('runtime resource budget refused')
    # Explicit foreground lifetime; no infinite daemon or silent profile install.
    spec['budget']['native_timeout_s'] = a.lifetime
    spec['budget']['window_s'] = a.lifetime+spec['budget']['wait_s']+30
    native = LeasedNative(spec, spec['baseline']['config'], a.out, time.time())
    link = None
    httpd = None
    try:
        link = leased_link(native)
        service = UnifiedService(link, engine_path=spec['engine']['path'], timeout=60,
                tokenizer_model=str(Path(spec['cwd']) / spec['model']['path']), backend='qwen35_llama',
                backend_source='unified-config')
        service.service_deadline=native.started+a.lifetime
        httpd = start_http(service, a.host, a.port)
        httpd.timeout = .5
        print(json.dumps({'url': 'http://%s:%d' % httpd.server_address, 'shared_executor': True,
                          'lease_job': native.job_id, 'lifetime_s': a.lifetime}), flush=True)
        until = native.started + a.lifetime - 10
        stopping=threading.Event();service.stopping=stopping;previous_signals={}
        def stop(signum,frame):stopping.set()
        for name in ('SIGTERM','SIGBREAK'):
            sig=getattr(signal,name,None)
            if sig is not None:
                previous_signals[sig]=signal.signal(sig,stop)
        while time.monotonic() < until and not link.dead and not stopping.is_set():
            httpd.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        for sig,previous in locals().get('previous_signals',{}).items():
            signal.signal(sig,previous)
        # Chat cooperatively STOPs/drains. Other SCORE endpoints finish within
        # their request deadline; never inject a foreign cancel into their parser.
        if 'service' in locals():
            if getattr(service,'stopping',None) is not None:service.stopping.set()
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
