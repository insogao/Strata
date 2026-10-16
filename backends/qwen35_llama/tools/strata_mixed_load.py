"""Bounded client admission experiment. Exactly ONE serial workflow worker.

No native batching, overlapping inference, hidden retries, or quality claims.
Concurrent clients submit whole SCORE/continue/release transactions, so sessions
cannot interleave. Busy queue returns503; failed submissions remain in totals.
"""
from concurrent.futures import ThreadPoolExecutor
import queue
import statistics
import threading
import time


def bounded(rows, perform, *, clients, capacity):
    if type(clients) is not int or not 1<=clients<=8 or type(capacity) is not int or not 1<=capacity<=8:
        raise ValueError('clients/capacity must be1..8')
    rows=list(rows)
    if not 1<=len(rows)<=64:raise ValueError('bounded pilot must contain1..64 requests')
    jobs=queue.Queue(maxsize=capacity);results=[];failures=[];lock=threading.Lock()
    begun=time.perf_counter()
    def worker():
        while True:
            job=jobs.get()
            try:
                if job is None:return
                i,sample,arrival,done,box=job
                start=time.perf_counter()
                if failures:
                    box.append({'sample_id':sample['sample_id'],'invalid_reason':'aborted_after_unsafe_failure'})
                else:
                    try:box.append(perform(sample))
                    except BaseException as exc:
                        failures.append(exc)
                        box.append({'sample_id':sample['sample_id'],'invalid_reason':'unsafe_failure'})
                box[0].update(queue_wait_ms=1000*(start-arrival),service_ms=1000*(time.perf_counter()-start),
                              submission_index=i,client_end_to_end_ms=1000*(time.perf_counter()-arrival),http_status=200 if not box[0].get('invalid_reason') else 500)
                done.set()
            finally:jobs.task_done()
    thread=threading.Thread(target=worker);thread.start()
    def submit(pair):
        i,sample=pair;arrival=time.perf_counter();done=threading.Event();box=[]
        try:jobs.put_nowait((i,sample,arrival,done,box))
        except queue.Full:
            return {'sample_id':sample['sample_id'],'submission_index':i,'invalid_reason':'queue_full',
                    'http_status':503,'queue_wait_ms':0,'service_ms':0,'client_end_to_end_ms':1000*(time.perf_counter()-arrival)}
        # perform is bounded by the HTTP/native lease deadline. No response is
        # abandoned while it might still own an engine transaction.
        done.wait()
        return box[0]
    try:
        with ThreadPoolExecutor(max_workers=clients) as pool:results=list(pool.map(submit,enumerate(rows)))
    finally:
        jobs.put(None);jobs.join();thread.join()
    if failures:raise RuntimeError('mixed experiment stopped after unsafe workflow failure') from failures[0]
    return results,1000*(time.perf_counter()-begun)


def summarize(rows,wall_ms):
    valid=[r for r in rows if not r.get('invalid_reason')]
    return {'submitted':len(rows),'completed':len(valid),'failed':len(rows)-len(valid),
            '503':sum(r.get('http_status')==503 for r in rows),
            'wall_ms':wall_ms,'valid_completed_per_s':1000*len(valid)/wall_ms if wall_ms else None,
            'mean_queue_wait_ms':statistics.mean(r.get('queue_wait_ms',0) for r in rows),
            'max_queue_wait_ms':max(r.get('queue_wait_ms',0) for r in rows),
            'mean_client_e2e_ms':statistics.mean(r['client_end_to_end_ms'] for r in rows),
            'p95_client_e2e_ms':None,'tail_note':'bounded pilot<100; no reliable p95',
            'mean_actual_output_tokens':statistics.mean(r.get('generated_tokens',0) for r in rows),
            'completion_by_submission':[not r.get('invalid_reason') for r in rows],
            'quality_qualified_per_s':None}


def experiment(rows,client,guard,record,clients,capacity,maximum):
    if maximum not in (16,64,128):raise ValueError('explain maximum16/64/128 required')
    # Predicted research policy only. Never use synthetic reference labels as
    # oracle production routing. No gold => no quality-qualified speed claim.
    route=lambda predicted:predicted>=2
    output={}
    for arm in ('M0','M1','M2'):
        guard()
        def perform(sample):
            guard()
            r=client.one(sample,'A0' if arm=='M0' else 'B1',conditional=arm!='M0',route=route,max_new=maximum)
            r['arm']=arm;return r
        if arm=='M2':
            measured,wall=bounded(rows,perform,clients=clients,capacity=capacity)
        else:
            measured=[];start=time.perf_counter()
            for i,sample in enumerate(rows):
                r=perform(sample)
                r.update(queue_wait_ms=0,service_ms=r['e2e_ms'],client_end_to_end_ms=r['e2e_ms'],submission_index=i)
                measured.append(r)
            wall=1000*(time.perf_counter()-start)
        for r in measured:record(r)
        output[arm]=summarize(measured,wall)
    return {'arms':output,'routing':'predicted>=2; synthetic design labels not used for routing',
        'clients':clients,'queue_capacity':capacity,'explanation_limit':maximum,
        'status':'INCONCLUSIVE','gpu_parallelism':False,'batching':False,
        'schedule_gain':None,'reduced_work_gain':None,
        'reason':'Small development pilot and fixed arm order; explanation semantic equivalence unmeasured. Synthetic-reference evaluation is authorized; raw arithmetic is not a validated gain.',
        'M0_M1_scope':'Business prompt/output policy changes; reference-risk scoring alone does not establish explanation semantic equivalence.',
        'M1_M2_scope':'Same samples and predicted policy, serial GPU; failed queue admissions stay in denominator.',
        'oracle':'not run; predicted routing only, human labels are not a prerequisite',
        'next':'interleave repeated M1/M2 blocks with the same valid output contract; M3 not authorized'}


def main(argv=None):
    from strata_risk_eval import main as risk_main
    return risk_main(['run','--mixed']+list(argv if argv is not None else __import__('sys').argv[1:]))


if __name__=='__main__':raise SystemExit(main())
