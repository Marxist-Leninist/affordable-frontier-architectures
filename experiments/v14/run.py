"""v14: validation-locked training-adequacy and quality/cost gate, CPU only."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import json,time,hashlib,platform,math,statistics,copy,pickle
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import LM,count,state_bytes
from ngram import Ngram
P=Path(__file__).parent
DATA=Path(os.environ.get('WIKITEXT_DATA',str(P.parent/'frontier_research_v8')))
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
STEPS=4000; BATCH=8; LENGTH=128; RATES=[.003,.01,.03]; GATES=[2.5,2.35,2.25]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,obj): (P/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def load_data():
 manifest=json.loads((P/'data_manifest.json').read_text()); raw={}
 for split,row in manifest['splits'].items():
  path=DATA/(split+'.txt'); assert sha(path)==row['text_sha256']; raw[split]=path.read_bytes()
 return manifest,raw,{k:torch.tensor(np.frombuffer(v,dtype=np.uint8).copy(),dtype=torch.long) for k,v in raw.items()}
@torch.inference_mode()
def evaluate(model,arr):
 model.eval(); n=(len(arr)-1)//LENGTH; loss=0.; per=[]
 for first in range(0,n,32):
  ix=torch.arange(first,min(first+32,n))[:,None]*LENGTH+torch.arange(LENGTH)
  losses=F.cross_entropy(model(arr[ix]).flatten(0,1),arr[ix+1].flatten(),reduction='none').reshape(-1,LENGTH)
  per.extend((losses.sum(1)/LENGTH/math.log(2)).tolist());loss+=losses.double().sum().item()
 return dict(bpb=loss/(n*LENGTH)/math.log(2),targets=n*LENGTH,unscored_trailing_targets=(len(arr)-1)%LENGTH,block_bpb=per)
def compact(score): return {k:v for k,v in score.items() if k!='block_bpb'}
def train(kind,seed,lr,starts,data,pilot=False):
 torch.manual_seed(seed);model=LM(kind); opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.95),weight_decay=.01)
 curve=[]; train_seconds=0.; snapshots={}
 for step,offset in enumerate(starts):
  model.train();tic=time.perf_counter();ix=torch.from_numpy(offset)[:,None]+torch.arange(LENGTH)
  factor=min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-100)/(STEPS-100))))
  for group in opt.param_groups: group['lr']=lr*factor
  opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(data['train'][ix]).flatten(0,1),data['train'][ix+1].flatten())
  if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
  loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step();train_seconds+=time.perf_counter()-tic
  if step+1 in (1000,2000,4000):
   val=compact(evaluate(model,data['validation'])); row=dict(step=step+1,training_seconds=train_seconds,validation=val);curve.append(row)
   if not pilot: snapshots[step+1]=copy.deepcopy(model.state_dict())
   print('CURVE',kind,seed,lr,row,flush=True)
 return model,dict(kind=kind,seed=seed,lr=lr,parameters=count(model),curve=curve,training_seconds=train_seconds),snapshots
@torch.inference_mode()
def benchmark(model,raw):
 """Direct end-to-end batch1 known-continuation probe, prefill and full softmax included."""
 pre=[]; dec=[]; end=[];cache=[];p95_context=[]
 # Locked prefixes spread across test; never select by measured latency or quality.
 offsets=np.linspace(0,len(raw)-161,8,dtype=int)
 for offset in offsets:
  sequence=torch.tensor(list(raw[offset:offset+160]),dtype=torch.long)[None,:]
  for rep in range(5):
   tic=time.perf_counter();logits,state=model(sequence[:,:128],cache=True);torch.softmax(logits[:,-1],-1);t1=time.perf_counter()
   for j in range(32):
    logits,state=model(sequence[:,128+j:129+j],state,True); torch.softmax(logits[:,-1],-1)
   t2=time.perf_counter()
   if rep>=2:pre.append((t1-tic)*1000);dec.append((t2-t1)*1000/32);end.append((t2-tic)*1000);cache.append(state_bytes(state))
 return dict(prefill_bytes=128,decode_steps=32,contexts=8,repetitions_after_warmup=3,prefill_p50_ms=float(np.median(pre)),prefill_p95_ms=float(np.quantile(pre,.95)),decode_step_p50_ms=float(np.median(dec)),decode_step_p95_ms=float(np.quantile(dec,.95)),end_to_end_p50_ms=float(np.median(end)),end_to_end_p95_ms=float(np.quantile(end,.95)),end_to_end_mean_ms=float(np.mean(end)),final_cache_tensor_bytes=max(cache),scope='Single-thread CPU batch1 teacher-forced replay, full softmax. p95 is over 24 local probes, not service p95. Last byte predicts next distribution; outputs not sampled.')
def ng_benchmark(model,raw):
 pre=[];dec=[];end=[]
 for offset in np.linspace(0,len(raw)-161,8,dtype=int):
  seq=raw[offset:offset+160]
  for rep in range(5):
   tic=time.perf_counter();context=seq[:128][-model.order:]; np.array([model.prob(context,y) for y in range(256)]);t1=time.perf_counter()
   for j in range(32):
    context=(context+seq[128+j:129+j])[-model.order:];np.array([model.prob(context,y) for y in range(256)])
   t2=time.perf_counter()
   if rep>=2:pre.append((t1-tic)*1000);dec.append((t2-t1)*1000/32);end.append((t2-tic)*1000)
 return dict(prefill_bytes=128,decode_steps=32,contexts=8,repetitions_after_warmup=3,prefill_p50_ms=float(np.median(pre)),prefill_p95_ms=float(np.quantile(pre,.95)),decode_step_p50_ms=float(np.median(dec)),decode_step_p95_ms=float(np.quantile(dec,.95)),end_to_end_p50_ms=float(np.median(end)),end_to_end_p95_ms=float(np.quantile(end,.95)),end_to_end_mean_ms=float(np.mean(end)),logical_suffix_bytes=model.order,scope='Same 24 batch1 teacher-forced sequences; full 256-way distributions. Finite-order prefill needs only suffix; not prefix likelihood scoring.')
def main():
 begin=time.perf_counter();manifest,raw,data=load_data();rng=np.random.default_rng(14112026);starts=rng.integers(0,len(raw['train'])-LENGTH,size=(STEPS,BATCH),dtype=np.int64);np.save(P/'train_starts.npy',starts)
 protocol=dict(experiment='v14 training-adequacy and matched-quality gate',steps=STEPS,batch=BATCH,context=LENGTH,training_targets=STEPS*BATCH*LENGTH,rates=RATES,pilot_seed=101,final_seeds=[1,2,3],gates_validation_bpb=GATES,checkpoints=[1000,2000,4000],selection='Best 4000-step full validation LR on pilot; final checkpoints first crossing each fixed validation gate; no test selection.',source_hashes={f:sha(P/f) for f in ['run.py','models.py','ngram.py']},ledger_sha256=sha(P/'train_starts.npy'),dataset=manifest,torch=torch.__version__,platform=platform.platform(),cpu=platform.processor(),threads=1,optimizer='AdamW .9/.95 eps1e-8 decay.01 all parameters clip1 warmup100 cosine .1x',baseline='v12 modern 2-block RoPE/RMSNorm/SwiGLU Transformer vs parameter-matched residual gated GRU; 4x target presentations; wider LR grid; stronger training is not proof of convergence',quality_match='Shared validation thresholds; test within .02 bpb descriptive equivalence screen, not statistical equivalence proof. Test already seen in prior work.')
 write('protocol.json',protocol); tuning=[];selected={}
 for kind in ['transformer','gru']:
  for lr in RATES:
   _,row,_=train(kind,101,lr,starts,data,True);tuning.append(row);write('tuning_partial.json',tuning)
  selected[kind]=min([r for r in tuning if r['kind']==kind],key=lambda r:r['curve'][-1]['validation']['bpb'])['lr']
 # Train non-neural control on EXACTLY the same ledger, then tune order only on validation.
 stat=Ngram();tic=time.perf_counter()
 for start in starts.flatten():stat.add(raw['train'][start:start+129])
 statfit=time.perf_counter()-tic; statval={order:stat.eval(raw['validation'],order) for order in [2,3,4]};order=min(statval,key=lambda n:statval[n]['bpb']);stat.order=order
 write('selection.json',dict(neural=selected,ngram_order=order,ngram_validation=statval,tuning=tuning))
 finals=[];(P/'checkpoints').mkdir(exist_ok=True)
 for seed in [1,2,3]:
  for kind in ['transformer','gru']:
   model,row,snapshots=train(kind,seed,selected[kind],starts,data)
   row['validation_gate_steps']={str(g):next((v['step'] for v in row['curve'] if v['validation']['bpb']<=g),None) for g in GATES}
   row['evaluated_checkpoints']=[]
   for step in sorted(set([STEPS]+[s for s in row['validation_gate_steps'].values() if s is not None])):
    model.load_state_dict(snapshots[step]);score=evaluate(model,data['test']);np.save(P/f'{kind}_seed{seed}_step{step}_test_blocks.npy',np.array(score.pop('block_bpb')))
    path=P/'checkpoints'/f'{kind}_seed{seed}_step{step}.pt';torch.save(model.state_dict(),path)
    row['evaluated_checkpoints'].append(dict(step=step,test=score,checkpoint_file=path.name,checkpoint_sha256=sha(path),checkpoint_bytes=path.stat().st_size))
   # All timing later: no competing fitting/scoring within experiment.
   finals.append(row);write('final_partial.json',finals)
 stat_result=dict(training_seconds=statfit,training_target_presentations=STEPS*BATCH*LENGTH,selected_order=order,validation=statval,test=stat.eval(raw['test'],order),serialized_bytes=len(pickle.dumps(stat,protocol=5)),context_counts=[len(t) for t in stat.counts])
 # Timing is an isolated final stage, not parallel with neural fitting.
 for row in finals:
  model=LM(row['kind']).eval()
  for cp in row['evaluated_checkpoints']:
   model.load_state_dict(torch.load(P/'checkpoints'/cp['checkpoint_file'],weights_only=True));cp['latency']=benchmark(model,raw['test'])
 stat_result['latency']=ng_benchmark(stat,raw['test'])
 write('results.json',dict(protocol_sha256=sha(P/'protocol.json'),selection_sha256=sha(P/'selection.json'),tuning=tuning,final=finals,ngram=stat_result,total_wall_seconds=time.perf_counter()-begin))
 print('DONE',flush=True)
if __name__=='__main__':main()
