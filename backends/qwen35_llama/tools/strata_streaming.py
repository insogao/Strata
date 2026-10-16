"""Cancellation hooks around the existing strict EngineLink parsers."""
import time
import score_api as api


class ScoreCancelled(api.EngineError):
    pass


class CooperativeLink(api.EngineLink):
    def send(self,line):
        peer=getattr(self,'request_cancelled',None)
        if peer is not None and getattr(self,'_cooperative',None) is None:
            if line.startswith(('SCORE ','GEN ')):
                self._cooperative={'kind':'score' if line.startswith('SCORE ') else 'gen',
                    'callback':None,'cancelled':peer,'stop_sent':False,'count':0,
                    'maximum':int(line.split()[1]) if line.startswith('GEN ') else 0,'automatic':True}
        return super().send(line)

    def _operation(self, kind, callback, cancelled, maximum, run):
        self._cooperative = {'kind':kind,'callback':callback,'cancelled':cancelled,
                             'stop_sent':False,'count':0,'maximum':maximum}
        try:
            return run()
        finally:
            self._cooperative = None

    def score_interruptible(self, ids, timeout, cancelled):
        return self._operation('score', None, cancelled, 0,
            lambda: super(CooperativeLink,self).score(ids,1,timeout,keep=True))

    def generate_interruptible(self, maximum, timeout, callback, cancelled):
        return self._operation('gen',callback,cancelled,maximum,
            lambda: super(CooperativeLink,self).generate_continue(maximum,[],timeout))

    def _take_until(self, deadline, timeout):
        if getattr(self,'request_deadline',None) is not None:
            deadline=min(deadline,self.request_deadline)
        op = getattr(self,'_cooperative',None)
        if op is None:
            return super()._take_until(deadline,timeout)
        while True:
            if not op['stop_sent'] and op['cancelled']():
                self.send('STOP');op['stop_sent']=True
            try:
                row = super()._take_until(min(deadline,time.monotonic()+.1),timeout)
                break
            except api.EngineTimeout:
                if time.monotonic()>=deadline: raise
        if op['kind']=='score' and row.startswith('DONE '):
            d=api._parse_done(row)
            if d['finish']=='cancel':
                if d['produced']!=0: raise api.EngineProtocolError('cancelled SCORE emitted tokens')
                if op.get('automatic'):self._cooperative=None
                raise ScoreCancelled('private score cancelled and drained')
        if op['kind']=='gen' and row.startswith('T '):
            try:
                token=int(row[2:].strip())
                if not 0<=token<=2147483647 or op['count']>=op['maximum']:
                    raise ValueError()
            except ValueError:
                raise api.EngineProtocolError('invalid/overflow streamed token')
            op['count']+=1
            if op['callback'] is not None and not op['stop_sent']:
                try:
                    op['callback'](token)
                except OSError:
                    self.send('STOP');op['stop_sent']=True
                except Exception as exc:
                    raise api.EngineProtocolError('stream callback failed: '+str(exc))
        # Existing SCORE/GEN parsers still validate every row/count/DONE. Cancel
        # drains a complete response before private reset and lock release.
        if op.get('automatic') and row.startswith(('DONE ','ERR ')):
            self._cooperative=None
        return row
