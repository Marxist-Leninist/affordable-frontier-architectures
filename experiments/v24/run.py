"""v24 fixed alternating context continuation, saved v23 controls."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time,math,platform,resource
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import LM
from local_attention import localize
from prefill import prefill
from pruning import local_prefill
P=Path(__file__).parent;V=P.parent/'frontier_research_v19';DATA=P.parent/'frontier_research_v8';C=P.parent/'frontier_research_v23'
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(f,x):(P/f).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(kind,seed):
 m=LM('transformer' if kind=='local' else kind)
 if kind=='local':m=localize(m,128)
 f=V/'checkpoints'/f'{kind}_seed{seed}_adapt500.pt'
 assert sha(f)==json.loads((V/'checkpoint_manifest.json').read_text())[f.name]
 m.load_state_dict(torch.load(f,weights_only=True));return m

def data():
 mf=json.loads((P.parent/'frontier_research_v14/data_manifest.json').read_text());raw=(DATA/'train.txt').read_bytes();assert hashlib.sha256(raw).hexdigest()==mf['splits']['train']['text_sha256']
 fw=json.loads((P/'data_manifest.json').read_text());assert sha(P/'rows.json')==fw['response_sha256'];items=[]
 for row,m in zip(json.loads((P/'rows.json').read_text())['rows'],fw['documents']):
  d=row['row']['text'].encode()[:4097];assert hashlib.sha256(d).hexdigest()==m['capped_text_sha256']
  for s in range(0,len(d)-2048,2048):items.append(dict(doc=row['row_idx'],start=s,raw=d[s:s+2049]))
 return torch.tensor(np.frombuffer(raw,dtype=np.uint8).copy(),dtype=torch.long),items
@torch.inference_mode()
def score(m,kind,items):
 m.eval();out={512:[],2048:[]}
 for length in out:
  for b in range(0,len(items),4):
   raw=torch.tensor([list(i['raw']) for i in items[b:b+4]]);x=raw[:,:-1][:,-length:];y=raw[:,-128:]
   z,_=local_prefill(m,x,128,prune=True) if kind=='local' else prefill(m,x,None,128)
   loss=-z.double().log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)/math.log(2)
   assert torch.isfinite(loss).all();out[length].extend(loss.numpy())
 return {k:np.array(v) for k,v in out.items()}
def freeze():
 data_,items=data();starts=np.load(C/'train_starts.npy');np.save(P/'train_starts.npy',starts);assert sha(P/'train_starts.npy')==json.loads((C/'protocol.json').read_text())['ledger_sha256']
 files=['run.py','models.py','local_attention.py','prefill.py','pruning.py','prepare_data.py','test_v24.py','timing.py','summarize.py','check_data.py']
 inputs={f.name:sha(f) for f in (V/'checkpoints').glob('*adapt500.pt') if not f.name.startswith('transformer_short')}
 inputs.update({'v23/'+f.name:sha(f) for f in (C/'checkpoints').glob('*.pt')});write('input_checkpoint_manifest.json',inputs)
 write('wikitext_manifest.json',json.loads((P.parent/'frontier_research_v14/data_manifest.json').read_text()))
 write('items.json',[{k:v for k,v in i.items() if k!='raw'} for i in items])
 write('protocol.json',dict(experiment='v24 alternating mixed-context continuation',seeds=[1,2,3],arms=['mixed'],control_arms=['context2048','reset512'],order='zero-based even steps full2048, odd steps independent4x512; 125 each; same for every seed',training_rng_seed=240000,initial='same v19 transformer adapt500 state per seed; fresh AdamW each arm',steps=250,targets_per_update=2048,targets_per_arm_seed=512000,ledger_seed=23112026,ledger_sha256=sha(P/'train_starts.npy'),optimizer=dict(name='AdamW',lr=.001,betas=[.9,.95],eps=1e-8,weight_decay=.01,clip=1.,warmup=25,schedule='min(1,(step+1)/25)*(.1+.9*.5*(1+cos(pi*max(0,step-25)/225)))'),intervention='Same contiguous2048 x/y independently reshaped to4x512 in reset512 arm; same update count, targets, initial state, optimizer. Not equal FLOPs, time or convergence.',evaluation='Fresh pinned FineWeb rows800-899 capped4097 bytes; all nonoverlapping2048-input windows final128 targets. Nested512 diagnostic resets to last512 input of exactly same windows/targets; effective context385-512 versus1921-2048.',references='Frozen v19 local and GRU, no additional training; unequal training budget descriptive only',selection='No tuning, no early stopping selection, no validation/test-based choices. All completed arms and failed work retained.',tokenizer='UTF8 bytes0-255, no normalization/joins for evaluation',source_sha256={f:sha(P/f) for f in files},inputs_sha256=sha(P/'input_checkpoint_manifest.json'),data_sha256=sha(P/'data_manifest.json'),threads=1,torch=torch.__version__,platform=platform.platform(),cpu_budget_seconds=900,main_hard_cpu_limit_seconds=820,planned_statistics='Mean seed effects, paired document-cluster bootstrap5000 seed24; seeds share corpus; descriptive not formal equivalence',quality_screen_bpb=.02,timing='After quality, fixed first8 rows with >=2080 bytes;3seeds,3trainedarms,512/2048 nested prefill plus32 same observed updates;seed240024 shuffles arm/seed/context jobs;one full warmup per measured doc;batch1;80CPU-second limit;descriptive only'))
 print('Frozen',len(items),'windows',flush=True)
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(820,825));cpu=time.process_time();begin=time.perf_counter();protocol=json.loads((P/'protocol.json').read_text())
 for f,h in protocol['source_sha256'].items():assert sha(P/f)==h
 assert sha(P/'train_starts.npy')==protocol['ledger_sha256'];data_,items=data();starts=np.load(P/'train_starts.npy');results=[];arrays={};trains=[];ck={};score_costs=[]
 def record(m,kind,label,seed):
  sc=time.process_time();sw=time.perf_counter();scores=score(m,kind,items);score_costs.append(dict(label=label,seed=seed,cpu_seconds=time.process_time()-sc,wall_seconds=time.perf_counter()-sw))
  for length,a in scores.items():
   arrays[f'{label}_seed{seed}_ctx{length}']=a;row=dict(label=label,seed=seed,context=length,bpb=float(a.mean()));results.append(row);print('RESULT',row,flush=True)
  np.savez_compressed(P/'losses.npz',**arrays);write('results.json',dict(rows=results,training=trains,scoring=score_costs,cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-begin,complete=False))
 for seed in (1,2,3):
  record(load('transformer',seed),'transformer','initial',seed)
  for arm in ('mixed',):
   torch.manual_seed(240000+seed)
   m=load('transformer',seed);m.train();opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01);tc=time.process_time();curve=[]
   for step,start in enumerate(starts):
    if time.process_time()-cpu>770:raise RuntimeError('Budget safety stop; retain prior completed arms only')
    ix=torch.arange(2048)[None,:]+int(start);x=data_[ix];y=data_[ix+1]
    if step%2==1:x=x.reshape(4,512);y=y.reshape(4,512)
    factor=min(1.,(step+1)/25)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-25)/225)))
    for g in opt.param_groups:g['lr']=.001*factor
    opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x).flatten(0,1),y.flatten());assert torch.isfinite(loss)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    if (step+1)%50==0:curve.append(dict(step=step+1,nats=float(loss.detach()),grad_norm=float(norm)));print('TRAIN',seed,arm,curve[-1],flush=True)
   trains.append(dict(seed=seed,arm=arm,steps=250,targets=512000,cpu_seconds=time.process_time()-tc,curve=curve));f=f'{arm}_seed{seed}_step250.pt';torch.save(m.state_dict(),P/'checkpoints'/f);ck[f]=sha(P/'checkpoints'/f);write('checkpoint_manifest.json',ck);record(m,'transformer',arm,seed)
  for control in ('context2048','reset512'):
   f=C/'checkpoints'/f'{control}_seed{seed}_step250.pt';assert sha(f)==json.loads((C/'checkpoint_manifest.json').read_text())[f.name]
   m=load('transformer',seed);m.load_state_dict(torch.load(f,weights_only=True));record(m,'transformer',control,seed)
 write('results.json',dict(rows=results,training=trains,scoring=score_costs,cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-begin,complete=True));print('DONE',time.process_time()-cpu,flush=True)
if __name__=='__main__':
 import sys
 freeze() if '--freeze' in sys.argv else main()
