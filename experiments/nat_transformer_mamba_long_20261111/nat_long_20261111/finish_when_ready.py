"""Wait for complete training, then execute hash-verified postprocessing once."""
import fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON=ROOT.parent/'frontier_cpu_runtime_20261110/venv/bin/python'
STOP=False;CHILD=None;OWNS_LOCK=False
PHASES=[('heldout_evaluation','evaluate_long.py',5400,'evaluation_results.json'),('standalone_timing','benchmark_long.py',600,'standalone_timings.json'),('summary','summarize_long.py',300,'summary_long.json')]
def state(status,**extra):
 p=ROOT/'postprocess_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'status':status,'utc_epoch':time.time(),**extra},indent=2)+'\n');t.replace(p)
def stop(signum,frame):
 global STOP
 STOP=True
 if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
def verify_sources():
 hashes=json.loads((ROOT/'postprocess_source_hashes.json').read_text())
 for name,digest in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f'Postprocess source changed: {name}'
 for _,name,_,_ in PHASES:assert name in hashes
 return hashes
def run_owned(script,cap,log):
 global CHILD
 if STOP:raise RuntimeError('controller cancelled')
 CHILD=subprocess.Popen([str(PYTHON),'-u',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 deadline=time.monotonic()+cap
 try:
  while CHILD.poll() is None:
   if STOP or time.monotonic()>=deadline:
    CHILD.terminate()
    try:CHILD.wait(timeout=10)
    except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait(timeout=10)
    raise RuntimeError('owned postprocess child cancelled or timed out')
   try:CHILD.wait(timeout=.2)
   except subprocess.TimeoutExpired:pass
  if STOP:raise RuntimeError('controller cancelled')
  if CHILD.returncode:raise RuntimeError(f'{script} exited {CHILD.returncode}')
 finally:
  if CHILD is not None and CHILD.poll() is None:
   CHILD.terminate()
   try:CHILD.wait(timeout=10)
   except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait(timeout=10)
  CHILD=None
def main():
 global OWNS_LOCK
 lock=(ROOT/'.postprocess_controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 OWNS_LOCK=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 verify_sources()
 for _,_,_,name in PHASES:
  if (ROOT/name).exists():raise RuntimeError('Refusing duplicate postprocessing: '+name)
 state('waiting_for_training');deadline=time.monotonic()+21600
 while True:
  if STOP:raise RuntimeError('controller cancelled while waiting')
  r=json.loads((ROOT/'pipeline_status.json').read_text())
  if r['status']=='training_complete':break
  if r['status'] not in ['tuning','final_training']:raise RuntimeError('Training did not complete: '+json.dumps(r))
  if time.monotonic()>deadline:raise TimeoutError('Training readiness wait capped at six hours')
  time.sleep(1)
 # Recheck all outputs after readiness: another caller may have finished during wait.
 for _,_,_,name in PHASES:
  if (ROOT/name).exists():raise RuntimeError('Refusing post-wait duplicate: '+name)
 for phase,script,cap,output in PHASES:
  verify_sources()
  if STOP:raise RuntimeError('controller cancelled')
  if (ROOT/output).exists():raise RuntimeError('Refusing duplicate phase: '+output)
  if json.loads((ROOT/'pipeline_status.json').read_text())['status']!='training_complete':raise RuntimeError('Training completion receipt changed')
  state(phase);print(phase,flush=True)
  with (ROOT/(phase+'.log')).open('w') as log:run_owned(script,cap,log)
  if not (ROOT/output).is_file():raise RuntimeError('Phase missing output receipt: '+output)
 state('completed');print('POSTPROCESS_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OWNS_LOCK:state('failed',error=str(e))
  traceback.print_exc();sys.exit(1)
