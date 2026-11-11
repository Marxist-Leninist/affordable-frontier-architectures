"""Bounded mock-child tests: no subprocesses or model training launched."""
import concurrent.futures,json,queue,tempfile,threading,time
from pathlib import Path
import orchestrate_long as o
ROOT=Path(__file__).resolve().parent
results={}
original=o.subprocess.Popen
class Fake:
 def __init__(self,*args,**kw):self.returncode=None;self.done=threading.Event();self.terminated=False;self.killed=False
 def poll(self):return self.returncode
 def terminate(self):self.terminated=True;self.returncode=-15;self.done.set()
 def kill(self):self.killed=True;self.returncode=-9;self.done.set()
 def wait(self,timeout=None):
  if not self.done.wait(min(timeout or .1,.1)):raise o.subprocess.TimeoutExpired('mock',timeout)
  return self.returncode
jobs=[{'kind':'transformer','mode':'tune','seed':i,'lr':.001,'steps':1} for i in range(8)]
def reset(tmp):
 o.ROOT=Path(tmp);(o.ROOT/'logs').mkdir(exist_ok=True);o.STOP.clear();o.STOP_REASON='';o.processes.clear();o.started=time.monotonic();o.cores=queue.Queue()
 for c in [1,2]:o.cores.put(c)
def background(fn):
 state={}
 def go():
  try:state['result']=fn()
  except BaseException as e:state['error']=str(e)
 t=threading.Thread(target=go);t.start();return t,state
def until(fn):
 end=time.monotonic()+3
 while not fn():
  if time.monotonic()>end:raise AssertionError('mock test timeout')
  time.sleep(.005)
with tempfile.TemporaryDirectory() as tmp:
 reset(tmp);launched=[]
 def factory(*a,**kw):
  p=Fake();launched.append(p);return p
 o.subprocess.Popen=factory
 t,state=background(lambda:o.stage(jobs));until(lambda:len(launched)==2);o.request_stop('audit requested stop');t.join(3)
 assert not t.is_alive() and len(launched)==2 and all(p.terminated for p in launched)
 results['active_cancel_stops_two_children_and_prevents_six_queued_launches']=True
 reset(tmp);launched=[]
 def failfactory(*a,**kw):
  p=Fake();launched.append(p)
  if len(launched)==1:p.returncode=1;p.done.set()
  return p
 o.subprocess.Popen=failfactory
 t,state=background(lambda:o.stage(jobs));t.join(3)
 assert not t.is_alive() and len(launched)<=2 and o.STOP.is_set() and all(p.poll() is not None for p in launched)
 results['child_failure_cancels_queued_and_other_owned_children']=True
 reset(tmp);launched=[];entered=threading.Event();release=threading.Event()
 def blockedfactory(*a,**kw):
  entered.set();assert release.wait(3);p=Fake();launched.append(p);return p
 o.subprocess.Popen=blockedfactory
 t,state=background(lambda:o.stage(jobs));assert entered.wait(3)
 stopper,ss=background(lambda:o.request_stop('race audit stop'));until(o.STOP.is_set);release.set();stopper.join(3);t.join(3)
 assert not stopper.is_alive() and not t.is_alive() and len(launched)==1 and launched[0].terminated
 results['stop_during_popen_registration_catches_child_and_blocks_later_launches']=True
 reset(tmp);o.STOP.set();o.STOP_REASON='preexisting stop';launched=[];o.subprocess.Popen=factory
 t,state=background(lambda:o.stage(jobs));t.join(3);assert not t.is_alive() and not launched
 results['preexisting_stop_launches_nothing']=True
 o.subprocess.Popen=original
(ROOT/'orchestrator_cancel_audit.json').write_text(json.dumps({'passed':all(results.values()),'mock_tests':results,'scope':'Mock children only. No real subprocesses or training. Bounded cancellation, failure, pre-stop, and in-flight launch race checks.'},indent=2)+'\n')
print(json.dumps(results,indent=2))
