"""Single-entry routing/isolation over a real CPU stub pipe, no GPU."""
import threading
import time
import unittest
import json
import urllib.request
import http.client
from unittest.mock import patch, MagicMock

import strata_unified as unified
from test_score_api import StubService, OFFICIAL_STUB


class Tokenizer:
    def decode(self, ids,errors='replace'): return 'reply:' + ','.join(map(str, ids))


class UnifiedTests(unittest.TestCase):
    def env(self, **kw):
        with patch.object(unified.api,'EngineLink',unified.CooperativeLink):
            env = StubService(**kw)
        env.httpd.shutdown(); env.httpd.server_close()
        env.service = unified.UnifiedService(env.link, engine_path=env.proc.args[3],
                                            backend=kw.get('backend', 'qwen35_llama'))
        env.service._tokenizer = Tokenizer(); env.service._tokenizer_loaded = True
        env.service.encode_messages = lambda messages, tokenizer: [1, 2, 3]
        env.httpd = unified.start_http(env.service, port=0)
        self.assertFalse(env.httpd.daemon_threads)
        env.port = env.httpd.server_address[1]
        threading.Thread(target=env.httpd.serve_forever, daemon=True).start()
        self.addCleanup(env.close)
        return env

    @staticmethod
    def chat(**extra):
        return {'model': 'strata-qwen35', 'messages': [{'role': 'user', 'content': 'hello'}],
                'max_tokens': 2, **extra}

    def test_one_entry_discovers_routes_and_chat_clears_private_state(self):
        env = self.env(); pid = env.proc.pid
        self.assertTrue(env.get('/v1/capabilities')[1]['chat_completions'])
        routes = env.get('/v1/routes')[1]['routes']
        row = next(x for x in routes if x['path'] == '/v1/chat/completions' and x['backend'] == 'qwen35_llama')
        self.assertTrue(row['supported']); self.assertTrue(row['in_service'])
        self.assertEqual(env.get('/v1/models')[1]['data'][0]['id'], 'strata-qwen35')
        status, body, _ = env.post('/v1/chat/completions', self.chat())
        self.assertEqual(status, 200, body)
        self.assertEqual(body['choices'][0]['message']['content'], 'reply:0,1')
        self.assertFalse(body['prefix_reused']); self.assertIsNone(body['session'])
        self.assertFalse(env.get('/v1/session')[1]['armed'])
        self.assertEqual(env.proc.pid, pid)
        self.assertEqual(env.post('/v1/score', {'prompt_ids': [1]})[0], 200)

    def test_chat_cannot_steal_score_owner_and_explicit_owner_reset_releases(self):
        env = self.env()
        owner = env.post('/v1/score', {'prompt_ids': [1, 2], 'keep': True})[1]['session']
        with patch.object(env.link, 'send', wraps=env.link.send) as send:
            self.assertEqual(env.post('/v1/chat/completions', self.chat())[0], 409)
            send.assert_not_called()
        status, body, _ = env.post('/v1/chat/completions', self.chat(session=owner))
        self.assertEqual(status, 200, body); self.assertIsNone(body['session'])
        self.assertEqual(env.post('/v1/continue', {'max_new': 1, 'session': owner})[0], 409)

    def test_unsupported_features_do_not_mutate_engine(self):
        env = self.env()
        for extra, status in [({'stream': 'false'}, 400),
                              ({'model': 'other'}, 422), ({'temperature': .8}, 422)]:
            with self.subTest(extra=extra), patch.object(env.link, 'send', wraps=env.link.send) as send:
                self.assertEqual(env.post('/v1/chat/completions', self.chat(**extra))[0], status)
                send.assert_not_called()
        with self.assertRaises(ValueError): unified.start_http(env.service, '0.0.0.0', 0)

    def test_invalid_prompt_refusal_cleans_up_before_next_request(self):
        env = self.env()
        bad = {'prompt_ids': [64], 'max_tokens': 2}
        self.assertEqual(env.post('/v1/chat/completions', bad)[0], 422)
        self.assertFalse(env.get('/v1/session')[1]['armed'])
        self.assertEqual(env.post('/v1/chat/completions', self.chat())[0], 200)

    def test_corrupt_generation_quarantines_same_engine_for_all_routes(self):
        env = self.env(fault_mode='gen-score-done')
        status, body, _ = env.post('/v1/chat/completions', self.chat())
        self.assertEqual(status, 502); self.assertTrue(body['quarantined'])
        self.assertEqual(env.post('/v1/score', {'prompt_ids': [1]})[0], 502)
        self.assertEqual(env.get('/v1/session')[0], 502)

    def test_one_busy_lock_serializes_chat_and_score(self):
        env = self.env(score_delay_ms=120)
        result = []
        thread = threading.Thread(target=lambda: result.append(env.post('/v1/chat/completions', self.chat())[0]))
        thread.start()
        end = time.monotonic() + 2
        while not env.service._busy.locked() and time.monotonic() < end: time.sleep(.01)
        self.assertTrue(env.service._busy.locked())
        self.assertEqual(env.post('/v1/score', {'prompt_ids': [1]})[0], 503)
        thread.join(5); self.assertFalse(thread.is_alive()); self.assertEqual(result, [200])
        self.assertEqual(env.post('/v1/score', {'prompt_ids': [1]})[0], 200)

    def test_official_backend_is_not_silently_substituted(self):
        env = self.env(backend='official_q36', stub=OFFICIAL_STUB)
        self.assertFalse(env.get('/v1/capabilities')[1]['chat_completions'])
        self.assertEqual(env.post('/v1/chat/completions', self.chat())[0], 501)

    def test_server_bind_failure_releases_owned_native_and_link(self):
        import leased_native
        import strata_experiment as ex
        native, link = MagicMock(), MagicMock()
        spec = {'budget': {'wait_s':1}, 'baseline': {'config': {}}, 'engine': {'path': 'fake'},
                'cwd': '.', 'model': {'path': 'fake.gguf'}}
        with patch.object(ex.st, 'read_json', return_value=spec), \
                patch.object(ex, 'validate'), patch.object(ex, 'fingerprint'), \
                patch.object(ex, 'admission', return_value={'admitted': True}), \
                patch.object(leased_native, 'LeasedNative', return_value=native), \
                patch.object(unified, 'leased_link', return_value=link), \
                patch.object(unified, 'UnifiedService'), \
                patch.object(unified, 'start_http', side_effect=OSError('port occupied')):
            with self.assertRaisesRegex(OSError, 'port occupied'):
                unified.main(['--spec', 'unused', '--out', 'unused'])
        native.close.assert_called_once_with(abort=True)
        link.close.assert_called_once()

    def test_sse_chunks_finish_and_next_request_uses_clean_engine(self):
        env=self.env();body=self.chat(stream=True)
        req=urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions'%env.port,
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=5) as r:
            self.assertTrue(r.headers['Content-Type'].startswith('text/event-stream'))
            lines=[x[6:] for x in r.read().decode().splitlines() if x.startswith('data: ')]
        self.assertEqual(lines[-1],'[DONE]')
        chunks=[json.loads(x) for x in lines[:-1]]
        self.assertEqual(''.join(x['choices'][0]['delta'].get('content','') for x in chunks),'reply:0,1')
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'],'length')
        self.assertEqual(env.post('/v1/score',{'prompt_ids':[1]})[0],200)
        self.assertFalse(env.get('/v1/session')[1]['armed'])

    def test_stream_ownership_refusal_is_json_before_sse_headers(self):
        env=self.env();env.post('/v1/score',{'prompt_ids':[1],'keep':True})
        code,body,headers=env.post('/v1/chat/completions',self.chat(stream=True))
        self.assertEqual(code,409);self.assertTrue(headers['Content-Type'].startswith('application/json'))

    def test_disconnect_during_prefill_drains_and_next_request_recovers(self):
        env=self.env(score_delay_ms=300)
        c=http.client.HTTPConnection('127.0.0.1',env.port,timeout=5)
        c.request('POST','/v1/chat/completions',json.dumps(self.chat(stream=True)),{'Content-Type':'application/json'})
        end=time.monotonic()+2
        while not env.service._busy.locked() and time.monotonic()<end:time.sleep(.01)
        self.assertTrue(env.service._busy.locked());c.close()
        end=time.monotonic()+4
        while env.service._busy.locked() and time.monotonic()<end:time.sleep(.01)
        self.assertFalse(env.service._busy.locked())
        self.assertEqual(env.post('/v1/score',{'prompt_ids':[1]})[0],200)
        self.assertFalse(env.get('/v1/session')[1]['armed'])

    def test_draining_service_refuses_before_pipe_write(self):
        env=self.env();env.service.service_deadline=time.monotonic()
        with patch.object(env.link,'send',wraps=env.link.send) as send:
            self.assertEqual(env.post('/v1/chat/completions',self.chat())[0],503)
            send.assert_not_called()

    def test_stream_invalid_done_reports_error_without_success_terminator(self):
        env=self.env(fault_mode='gen-score-done')
        req=urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions'%env.port,
            data=json.dumps(self.chat(stream=True)).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=5) as response:data=response.read().decode()
        self.assertIn('"error"',data);self.assertNotIn('[DONE]',data)
        self.assertEqual(env.get('/v1/session')[0],502)

    def test_disconnected_keep_score_cannot_strand_unknown_owner(self):
        env=self.env(score_delay_ms=300)
        c=http.client.HTTPConnection('127.0.0.1',env.port,timeout=5)
        c.request('POST','/v1/score',json.dumps({'prompt_ids':[1,2],'keep':True}),{'Content-Type':'application/json'})
        end=time.monotonic()+2
        while not env.service._busy.locked() and time.monotonic()<end:time.sleep(.01)
        self.assertTrue(env.service._busy.locked());c.close()
        end=time.monotonic()+4
        while env.service._busy.locked() and time.monotonic()<end:time.sleep(.01)
        self.assertFalse(env.service._busy.locked())
        self.assertFalse(env.get('/v1/session')[1]['armed'])
        self.assertIsNone(env.service._session_lease)
        self.assertEqual(env.post('/v1/score',{'prompt_ids':[1]})[0],200)


if __name__ == '__main__': unittest.main()
