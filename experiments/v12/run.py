"""Frozen-budget WikiText-2 experiment. Run from this directory."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,platform,math,statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import LM,count,state_bytes
P=Path(__file__).parent
DATA=Path(os.environ.get('WIKITEXT_DATA',str(P.parent/'frontier_research_v8')))
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):(P/name).write_text(json.dumps(obj,indent=2)+'\n')
S=1000; B=8; L=128; RATES=[.001,.003,.01]
manifest=json.loads((P/'data_manifest.json').read_text());data={}
for split,row in manifest['splits'].items():
 path=DATA/(split+'.txt');assert sha(path)==row['text_sha256'];data[split]=torch.tensor(np.frombuffer(path.read_bytes(),dtype=np.uint8).copy(),dtype=torch.long)
# Identical paired windows for all initializations/models, independent from initialization RNG.
rng=np.random.default_rng(1212026);starts=rng.integers(0,len(data['train'])-L,size=(S,B),dtype=np.int64)
np.save(P/'train_starts.npy',starts)
protocol={'experiment':'v12 modern Transformer versus gated recurrence and statistical ngrams','steps':S,'batch':B,'context_bytes':L,'training_target_presentations_per_run':S*B*L,'learning_rates':RATES,'pilot_seed':101,'final_seeds':[1,2,3],'optimizer':'AdamW betas (.9,.95), eps1e-8, weight_decay .01; all parameters; gradient clip1; warmup50 cosine to .1x','selection':'final-step full validation bpb only; then fresh final seeds; no test tuning','models':{k:count(LM(k)) for k in ['transformer','gru']},'dtype':'float32','threads':1,'torch':torch.__version__,'platform':platform.platform(),'training_windows_sha256':sha(P/'train_starts.npy'),'source_sha256':{f:sha(P/f) for f in ['models.py','run.py','test_models.py']},'dataset':manifest,'evaluation':'All complete 128-byte windows, state reset per window; first source byte and final incomplete targets excluded; identical for all models','timing':'single-process CPU only, batch1 cached decode with real text inputs, no sampling; 5 warmups and 50 measured steps at contexts128/512/2048; p50/p95 are local timing statistics, not service SLO'}
write('protocol.json',protocol)
@torch.inference_mode()
def evaluate(m,split):
 m.eval();arr=data[split];n=(len(arr)-1)//L;total=0.;tokens=0
 for first in range(0,n,32):
  ix=torch.arange(first,min(first+32,n))[:,None]*L+torch.arange(L);x=arr[ix];y=arr[ix+1];loss=F.cross_entropy(m(x).flatten(0,1),y.flatten(),reduction='sum');total+=loss.item();tokens+=y.numel()
 return {'bpb':total/tokens/math.log(2),'targets':tokens,'unscored_trailing_targets':(len(arr)-1)%L}
def train(kind,seed,lr):
 torch.manual_seed(seed);m=LM(kind);opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(.9,.95),weight_decay=.01);trace=[];t=time.perf_counter()
 for step,start in enumerate(starts):
  m.train();ix=torch.from_numpy(start)[:,None]+torch.arange(L);x=data['train'][ix];y=data['train'][ix+1]
  factor=min(1.,(step+1)/50)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-50)/(S-50))))
  for g in opt.param_groups:g['lr']=lr*factor
  opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x).flatten(0,1),y.flatten());loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if (step+1)%200==0:trace.append({'step':step+1,'training_batch_bpb':loss.item()/math.log(2)});print(kind,seed,lr,step+1,trace[-1]['training_batch_bpb'],flush=True)
 trainsec=time.perf_counter()-t;m.eval();path=P/'checkpoints'/f'{kind}_seed{seed}_lr{lr}.pt';path.parent.mkdir(exist_ok=True)
 torch.save(m.state_dict(),path);n=LM(kind);n.load_state_dict(torch.load(path,weights_only=True));n.eval()
 with torch.inference_mode():assert torch.equal(m(x),n(x))
 row={'kind':kind,'seed':seed,'learning_rate':lr,'parameters':count(m),'training_seconds':trainsec,'trace':trace,'checkpoint':{'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size,'reload_exact':True},'validation':evaluate(m,'validation')};return m,row
@torch.inference_mode()
def latency(m):
 m.eval();rows=[]
 for length in [128,512,2048]:
  prefix=data['test'][10000:10000+length][None,:]
  # Prefill independently re-run; decode starts from corresponding fresh state.
  pre=[]
  for i in range(15):
   t=time.perf_counter();_,s=m(prefix,cache=True);dt=time.perf_counter()-t
   if i>=5:pre.append(dt*1000)
  initial_bytes=state_bytes(s);times=[]
  for i in range(55):
   x=data['test'][10000+length+i:10001+length+i][None,:];t=time.perf_counter();_,s=m(x,s,True);dt=time.perf_counter()-t
   if i>=5:times.append(dt*1000)
  rows.append({'prefix_bytes':length,'prefill_p50_ms':float(np.median(pre)),'prefill_p95_ms':float(np.quantile(pre,.95)),'cached_decode_p50_ms':float(np.median(times)),'cached_decode_p95_ms':float(np.quantile(times,.95)),'initial_cache_tensor_bytes':initial_bytes,'final_cache_tensor_bytes':state_bytes(s),'decode_samples':50,'prefill_samples':10})
 return rows
if __name__=='__main__':
 tuning=[];selected={}
 for kind in ['transformer','gru']:
  for lr in RATES:
   _,row=train(kind,101,lr);tuning.append(row);write('tuning_partial.json',tuning);print('TUNED',kind,lr,row['validation'],flush=True)
  selected[kind]=min([r for r in tuning if r['kind']==kind],key=lambda r:r['validation']['bpb'])['learning_rate']
 write('selection.json',{'selected':selected,'tuning':tuning});final=[]
 for seed in [1,2,3]:
  for kind in ['transformer','gru']:
   m,row=train(kind,seed,selected[kind]);row['test']=evaluate(m,'test');row['latency']=latency(m);final.append(row);write('final_partial.json',final);print('FINAL',kind,seed,row['test'],flush=True)
 write('results.json',{'protocol_sha256':sha(P/'protocol.json'),'selection_sha256':sha(P/'selection.json'),'tuning':tuning,'final':final,'summary':{k:{'mean_test_bpb':statistics.mean(r['test']['bpb'] for r in final if r['kind']==k),'sd_test_bpb':statistics.stdev(r['test']['bpb'] for r in final if r['kind']==k)} for k in selected}})
 print('DONE',flush=True)
