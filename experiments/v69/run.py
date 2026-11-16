import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,pathlib,hashlib,sys,math,time,resource,signal,io
P=pathlib.Path(__file__).resolve().parent;R=P.parent;V=R/'frontier_research_v56_prediction_recovery'
resource.setrlimit(resource.RLIMIT_CPU,(300,300));resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));signal.alarm(360)
def digest(b):return hashlib.sha256(b).hexdigest()
def audit(event,args):
 if event.startswith(('socket.','subprocess.','os.system','os.spawn','os.exec')):raise RuntimeError('No network/process')
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
  assert pathlib.Path(os.fsdecode(args[0])).resolve().is_relative_to(P),'outside write'
 if event in ('os.remove','os.rename','os.replace','os.rmdir','os.mkdir','os.chmod','os.truncate','os.utime','os.link','os.symlink'):
  for a in args[:2 if event in ('os.rename','os.replace','os.link','os.symlink') else 1]:
   assert isinstance(a,(str,bytes,os.PathLike)) and pathlib.Path(os.fsdecode(a)).resolve().is_relative_to(P),'outside mutation'
sys.addaudithook(audit)
sys.path[:0]=[str(V/'runtime'),str(V/'frozen')]
def forbidden(*a,**k):raise RuntimeError('No training/save')
def statehash(state):
 h=hashlib.sha256()
 for n,t in sorted(state.items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 import random,statistics
 import v42_models
 original=v42_models.delta_recurrent
 def fast(q,k,v,logalpha,beta,state=None):
  if q.shape[2]!=1: return original(q,k,v,logalpha,beta,state)
  st=q.new_zeros((q.shape[0],q.shape[1],q.shape[3],v.shape[3])) if state is None else state
  dec=logalpha[:,:,0].exp()[...,None,None]*st
  kk=k[:,:,0];qq=q[:,:,0];vv=v[:,:,0]
  residual=vv-(kk.unsqueeze(-2)@dec).squeeze(-2)
  st=dec+beta[:,:,0,None,None]*kk.unsqueeze(-1)*residual.unsqueeze(-2)
  return (qq.unsqueeze(-2)@st),st
 start=time.perf_counter();cpu=time.process_time();S=R/'frontier_research_v63_fresh_context'
 protocol=json.loads((S/'protocol.json').read_text());raw=(S/'private_windows.json').read_bytes();assert digest(raw)==protocol['private_windows_sha256'];windows=json.loads(raw)
 with (P/'attempt.json').open('x') as f:json.dump({'start':time.time(),'runner_hash':digest((P/'run.py').read_bytes())},f)
 for f,h in protocol['protected'].items():assert digest((R/f).read_bytes())==h
 import torch,models
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 torch.Tensor.backward=forbidden;torch.autograd.backward=forbidden;torch.autograd.grad=forbidden;torch.optim.AdamW.__init__=forbidden;torch.save=forbidden
 load=torch.load;torch.load=forbidden;ms={};hs={}
 for cp in protocol['checkpoints']:
  data=(V/'frozen'/cp['path']).read_bytes();assert digest(data)==cp['sha256'];state=load(io.BytesIO(data),weights_only=True,map_location='cpu')
  m=models.build(cp['seed'],cp['arm']).cpu().eval();m.load_state_dict(state,strict=True)
  for t in m.parameters():t.requires_grad_(False)
  key=(cp['arm'],cp['seed']);ms[key]=m;hs[key]=statehash(m.state_dict())
 xs={b:torch.tensor([w['window'][:512] for w in windows[:b]]) for b in [1,4]};prefixes={b:x[:,:480] for b,x in xs.items()};steps={b:[x[:,i:i+1] for i in range(480,512)] for b,x in xs.items()}
 tasks=[(k,b,variant) for k in ms for b in [1,4] for variant in (['original','fast'] if k[0]=='GGG' else ['original'])];rng=random.Random(69);out=[]
 with torch.inference_mode():
  for repeat in range(-2,5):
   order=tasks.copy();rng.shuffle(order)
   for key,b,variant in order:
    v42_models.delta_recurrent=original if variant=='original' else fast
    t0=time.perf_counter_ns();y,cache=ms[key](prefixes[b],cache=True);pre=time.perf_counter_ns()-t0
    pieces=[];t0=time.perf_counter_ns()
    for x in steps[b]:
     z,cache=ms[key](x,states=cache,cache=True);pieces.append(z)
    dec=time.perf_counter_ns()-t0
    joined=torch.cat(pieces,1);reference=ms[key](xs[b])[:,480:]
    assert torch.isfinite(joined).all() and torch.allclose(joined,reference,atol=1e-4,rtol=1e-4)
    if repeat>=0:out.append(dict(variant=variant,arm=key[0],seed=key[1],batch=b,repeat=repeat,prefix=480,steps=32,prefill_ns=pre,decode_ns=dec,max_abs_error=(joined-reference).abs().max().item()))
 for k,m in ms.items():assert statehash(m.state_dict())==hs[k]
 for f,h in protocol['protected'].items():assert digest((R/f).read_bytes())==h
 (P/'timings.json').write_text(json.dumps(out,indent=2));summary=[]
 for arm in ['AAAwide','GGG']:
  for b in [1,4]:
   for variant in (['original','fast'] if arm=='GGG' else ['original']):
    vals=[x for x in out if x['arm']==arm and x['batch']==b and x['variant']==variant]
    summary.append(dict(arm=arm,batch=b,variant=variant,samples=len(vals),prefill_median_ms=statistics.median(x['prefill_ns'] for x in vals)/1e6,decode32_median_ms=statistics.median(x['decode_ns'] for x in vals)/1e6,combined_median_ms=statistics.median(x['decode_ns']+x['prefill_ns'] for x in vals)/1e6,max_abs_error=max(x['max_abs_error'] for x in vals)))
 (P/'summary.json').write_text(json.dumps(summary,indent=2));(P/'receipt.json').write_text(json.dumps(dict(cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-start,cases=len(out),preserved=True),indent=2))
if __name__=='__main__':main()
