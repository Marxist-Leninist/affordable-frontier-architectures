"""At most two pinned one-thread CPU jobs; hard global time cap, own jobs only."""
import concurrent.futures,hashlib,json,os,queue,signal,subprocess,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol_long.json').read_text())
started=time.monotonic();cores=queue.Queue()
for core in P['cpu_cores']:cores.put(core)
processes=[];process_lock=threading.Lock();STOP=threading.Event();STOP_REASON=''
def write(name,obj):
    path=ROOT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def request_stop(reason):
    global STOP_REASON
    STOP_REASON=STOP_REASON or reason;STOP.set()
    with process_lock:active=[p for p in processes if p.poll() is None]
    for p in active:
        try:p.terminate()
        except ProcessLookupError:pass

def run(job):
    core=None;tag=f"{job['mode']}_{job['kind']}_s{job['seed']}_lr{job['lr']}"
    try:
        while core is None:
            if STOP.is_set():raise RuntimeError(STOP_REASON or 'cancelled before core acquisition')
            try:core=cores.get(timeout=.2)
            except queue.Empty:continue
        if STOP.is_set():raise RuntimeError(STOP_REASON or 'cancelled after core acquisition')
        remaining=P['global_wall_cap_seconds']-(time.monotonic()-started)
        if remaining<=0:raise RuntimeError('global wall cap reached')
        cmd=[sys.executable,'train_long.py','--kind',job['kind'],'--seed',str(job['seed']),'--lr',str(job['lr']),'--steps',str(job['steps']),'--mode',job['mode'],'--core',str(core)]
        with (ROOT/'logs'/f'{tag}.log').open('w') as log:
            with process_lock:
                if STOP.is_set():raise RuntimeError(STOP_REASON or 'cancelled before launch')
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});processes.append(p)
            while p.poll() is None:
                if time.monotonic()-started>=P['global_wall_cap_seconds']:request_stop('global wall cap reached')
                if STOP.is_set():
                    p.terminate()
                    try:p.wait(timeout=60)
                    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
                    raise RuntimeError(STOP_REASON or 'cancelled active job')
                try:p.wait(timeout=.5)
                except subprocess.TimeoutExpired:pass
            rc=p.returncode
        if rc:raise RuntimeError(f'{tag} exited {rc}; inspect log')
        result=json.loads((ROOT/'runs'/tag/'result.json').read_text())
        if result['status']!='completed':raise RuntimeError(f'{tag}: '+result['status'])
        if job['mode']=='final':
            task_result=json.loads((ROOT/'runs'/tag/'task_result.json').read_text())
            if task_result['status']!='completed' or task_result['steps']!=P['task_steps']:
                raise RuntimeError(f'{tag}: capability adaptation incomplete')
        print(json.dumps({'completed':tag,'cpu_seconds':result['training_cpu_seconds'],'wall_seconds':result['training_wall_seconds']}),flush=True)
        return result
    except Exception as e:
        request_stop(str(e));raise
    finally:
        if core is not None:cores.put(core)
def stop(signum,frame):
    request_stop(f'received signal {signum}')
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def stage(jobs):
    if STOP.is_set():raise RuntimeError(STOP_REASON)
    ex=concurrent.futures.ThreadPoolExecutor(max_workers=2)
    futures=[ex.submit(run,job) for job in jobs]
    try:return [f.result() for f in futures]
    except BaseException:
        request_stop(STOP_REASON or 'stage failed or interrupted')
        for f in futures:f.cancel()
        raise
    finally:
        # Queued jobs are cancelled; running owned jobs have a 60s grace + 10s kill wait.
        ex.shutdown(wait=True,cancel_futures=True)
def main():
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'runs').mkdir(exist_ok=True)
    write('pipeline_status.json',{'status':'tuning','started_epoch':time.time(),'parent_pid':os.getpid(),'max_jobs':2,'global_cap_seconds':P['global_wall_cap_seconds']})
    jobs=[{'mode':'tune','kind':kind,'seed':101,'lr':lr,'steps':P['tune_steps']} for lr in P['learning_rates'] for kind in ['transformer','mamba']]
    tuned=stage(jobs)
    selected={kind:min([x for x in tuned if x['kind']==kind],key=lambda x:x['validation_curves'][-1]['validation']['loss_nats'])['lr'] for kind in ['transformer','mamba']}
    write('selection_long.json',{'selected_lrs':selected,'criterion':'final validation mean suffix CE, tuning seed101; no test metrics','tuning_runs':[{'kind':x['kind'],'lr':x['lr'],'validation':x['validation_curves'][-1]['validation']} for x in tuned]})
    write('pipeline_status.json',{'status':'final_training','selected':selected,'elapsed_seconds':time.monotonic()-started,'parent_pid':os.getpid()})
    jobs=[{'mode':'final','kind':kind,'seed':seed,'lr':selected[kind],'steps':P['final_steps']} for seed in P['seeds'] for kind in ['mamba','transformer']]
    finals=stage(jobs)
    write('pipeline_status.json',{'status':'training_complete','elapsed_seconds':time.monotonic()-started,'final_runs':len(finals),'selected':selected,'parent_pid':os.getpid()})
    print('ALL_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        request_stop(str(e));write('pipeline_status.json',{'status':'failed_or_capped','error':str(e),'elapsed_seconds':time.monotonic()-started});raise
