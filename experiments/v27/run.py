"""v27 bounded memory quality/storage experiment; frozen before any scoring."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time,math,resource,platform
from pathlib import Path
import numpy as np
import torch
from models import LM,state_bytes,count
from local_attention import localize
from prefill import prefill
from pruning import local_prefill
from compact_cache import CompactCache
from bounded_cache import BoundedCache,reachable_bytes
P=Path(__file__).parent;ARMS=('local','gru');SCOPES=(512,1024,2048,'full');ALPHAS={'local':.3,'gru':.2}
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(f,x):(P/f).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def checkpoint(a,s):return P.parent/'frontier_research_v19/checkpoints'/f'{a}_seed{s}_adapt500.pt'
def load(a,s):
 f=checkpoint(a,s);assert sha(f)==json.loads((f.parent.parent/'checkpoint_manifest.json').read_text())[f.name]
 m=LM('gru' if a=='gru' else 'transformer')
 if a=='local':m=localize(m,128)
 m.load_state_dict(torch.load(f,weights_only=True));return m.eval()
def data():
 mf=json.loads((P/'data_manifest.json').read_text());assert sha(P/'rows.json')==mf['response_sha256'];items=[]
 for r,d in zip(json.loads((P/'rows.json').read_text())['rows'],mf['documents']):
  raw=r['row']['text'].encode()[:8193];assert hashlib.sha256(raw).hexdigest()==d['capped_text_sha256']
  for start in range(0,len(raw)-4096,4096):items.append(dict(doc=r['row_idx'],start=start,raw=raw[start:start+4097]))
 return items
@torch.inference_mode()
def score(m,a,items):
 out=[]
 for b in range(0,len(items),4):
  raw=torch.tensor([list(i['raw']) for i in items[b:b+4]]);x=raw[:,:-1];y=raw[:,-128:]
  z,_=local_prefill(m,x,128,prune=True) if a=='local' else prefill(m,x,None,128)
  out.extend((-z.double().log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)/math.log(2)).numpy())
 return np.array(out)
def make_cache(scope):return CompactCache() if scope=='full' else BoundedCache(scope)
def copy_scores(items,scope):
 probs=[];active=[]
 for i in items:
  c=make_cache(scope);pp=[];aa=[]
  for t,b in enumerate(i['raw'][:-1]):
   c.feed(b)
   if t>=3968:
    q,n,k=c.distribution();pp.append(q[i['raw'][t+1]]);aa.append(n>0)
  probs.append(pp);active.append(aa)
 return np.array(probs),np.array(active)
def freeze():
 items=data();assert items
 write('items.json',[dict(doc=i['doc'],start=i['start'],sha256=hashlib.sha256(i['raw']).hexdigest()) for i in items])
 write('input_checkpoint_manifest.json',{str(checkpoint(a,s).relative_to(P.parent)):sha(checkpoint(a,s)) for a in ARMS for s in (1,2,3)})
 write('protocol.json',dict(experiment='v27 physical bounded copy memory',data='FineWeb-Edu sample10BT pinned rows1100–1199 cap8193, nonoverlapping4096-input windows final128 targets, no training/tuning',arms=ARMS,seeds=[1,2,3],scopes=SCOPES,alpha=ALPHAS,design='Hold frozen neural output and target ledger identical across memory scopes. Bounded cache stores only complete context/follower pairs in dynamic last S observed bytes; physical eviction. Full uses compact exact unbounded index. Fixed orders4/8/16; one uniform pseudocount; fall back to neural if no match.',statistics='All targets primary; seeds averaged first; 5000 paired document cluster bootstrap seed27; scope-minus-full and hybrid-minus-neural; conditional convenience cohort not population inference.',quality_gate='Prespecified practical loss increase <=0.01 bits/byte versus full hybrid; require upper paired95% CI<=0.01 to claim noninferiority here. No frontier/task equivalence.',timing='First8 rows>=4128 bytes,3seeds,neural/full/512/1024/2048 paths; random jobs seed270027; warmup plus2 measured repeats,4096 prefill+32 observed updates,complete256 distributions. Bounded index may initialize only final S prefix bytes because older pairs are necessarily evicted; neural gets same4096 inputs. Same optimized local path in all comparisons.',cpu_limit_main=500,cpu_limit_timing=200,source_sha256={f.name:sha(f) for f in P.glob('*.py')},data_manifest_sha256=sha(P/'data_manifest.json'),input_checkpoint_manifest_sha256=sha(P/'input_checkpoint_manifest.json'),torch=torch.__version__,platform=platform.platform()))
 print('Frozen',len(items),'windows',len({i['doc'] for i in items}),'docs',flush=True)
def verify():
 pr=json.loads((P/'protocol.json').read_text())
 for f,h in pr['source_sha256'].items():assert sha(P/f)==h
 for f in ('data_manifest','input_checkpoint_manifest'):assert sha(P/(f+'.json'))==pr[f+'_sha256']
 for f,h in json.loads((P/'input_checkpoint_manifest.json').read_text()).items():assert sha(P.parent/f)==h
@torch.inference_mode()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(500,505));start=time.process_time();verify();items=data();arrays={};copies={s:copy_scores(items,s) for s in SCOPES}
 for s,(p,act) in copies.items():arrays[f'copy_{s}']=-np.log2(p);arrays[f'active_{s}']=act
 for seed in (1,2,3):
  for a in ARMS:
   loss=score(load(a,seed),a,items);arrays[f'{a}_seed{seed}_neural']=loss
   for scope,(q,active) in copies.items():
    w=ALPHAS[a]*active;arrays[f'{a}_seed{seed}_{scope}']=-np.log2((1-w)*np.exp2(-loss)+w*q)
   print(a,seed,float(loss.mean()),flush=True)
 np.savez_compressed(P/'losses.npz',**arrays);write('results.json',dict(complete=True,cpu_seconds=time.process_time()-start,bpb={k:float(v.mean()) for k,v in arrays.items() if not k.startswith('active')}))
if __name__=='__main__':
 import sys
 freeze() if '--freeze' in sys.argv else main()
