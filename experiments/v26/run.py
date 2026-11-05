"""Frozen distant-prefix content intervention; no training or test selection."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time,math,platform,resource
from pathlib import Path
import numpy as np
import torch
from models import LM
from local_attention import localize
from prefill import prefill
from pruning import local_prefill
from compact_cache import CompactCache
from copy_cache import CopyCache
ALPHAS={"context2048":.6,"local":.3,"gru":.2}
P=Path(__file__).parent
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
ARMS=('context2048','local','gru');CONDS=('original2048','donor2048','suffix512')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(f,x):(P/f).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def checkpoint(arm,seed):
 v=19 if arm in ('local','gru') else 24 if arm=='mixed' else 23
 f=f'{arm}_seed{seed}_adapt500.pt' if v==19 else f'{arm}_seed{seed}_step250.pt'
 return P.parent/f'frontier_research_v{v}'/'checkpoints'/f
def load(arm,seed):
 f=checkpoint(arm,seed);assert sha(f)==json.loads((f.parent.parent/'checkpoint_manifest.json').read_text())[f.name]
 m=LM('gru' if arm=='gru' else 'transformer')
 if arm=='local':m=localize(m,128)
 m.load_state_dict(torch.load(f,weights_only=True));return m.eval()
def data():
 mf=json.loads((P/'data_manifest.json').read_text());assert sha(P/'rows.json')==mf['response_sha256'];docs=[]
 for r,d in zip(json.loads((P/'rows.json').read_text())['rows'],mf['documents']):
  raw=r['row']['text'].encode()[:4097];assert hashlib.sha256(raw).hexdigest()==d['capped_text_sha256']
  if len(raw)>=2049:docs.append((r['row_idx'],raw,d['full_text_sha256']))
 items=[]
 for j,(doc,raw,h) in enumerate(docs):
  donor=next(docs[(j+k)%len(docs)] for k in range(1,len(docs)) if docs[(j+k)%len(docs)][2]!=h)
  for start in range(0,len(raw)-2048,2048):
   original=raw[start:start+2049];changed=donor[1][:1408]+original[1408:]
   items.append(dict(doc=doc,start=start,donor_doc=donor[0],original_sha256=hashlib.sha256(original).hexdigest(),donor_window_sha256=hashlib.sha256(changed).hexdigest(),raw=original,changed=changed))
 return items
@torch.inference_mode()
def score(m,arm,items,condition):
 out=[]
 for b in range(0,len(items),4):
  raw=torch.tensor([list(i['changed' if condition=='donor2048' else 'raw']) for i in items[b:b+4]])
  x=raw[:,:-1];y=raw[:,-128:]
  if condition=='suffix512':x=x[:,-512:]
  z,_=local_prefill(m,x,128,prune=True) if arm=='local' else prefill(m,x,None,128)
  losses=-z.double().log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)/math.log(2)
  assert torch.isfinite(losses).all();out.extend(losses.numpy())
 return np.array(out)
def subgroup(history):
 # history length t, observed followers j<t. Suffix starts t-512.
 t=len(history); key=history[-8:]; cutoff=max(0,t-512)
 far=any(history[j-8:j]==key for j in range(8,min(cutoff,1408)))
 near=any(history[j-8:j]==key for j in range(cutoff+8,t))
 return far and not near

def copy_scores(items,condition):
 probs=[];active=[]
 for item in items:
  raw=item['changed' if condition=='donor2048' else 'raw']; c=CopyCache(); pp=[];aa=[]
  for t,b in enumerate(raw[:-1]):
   c.feed(b)
   if t>=1920:
    counts,n,k,far=c.query(512 if condition=='suffix512' else None)
    pp.append((counts[raw[t+1]]+1/256)/(n+1));aa.append(n>0)
  probs.append(pp);active.append(aa)
 return np.array(probs),np.array(active)

def freeze():
 items=data();write('items.json',[{k:v for k,v in i.items() if k not in ('raw','changed')} for i in items])
 masks=np.array([[subgroup(i['raw'][:t]) for t in range(1921,2049)] for i in items]);np.save(P/'subgroup.npy',masks)
 inputs={str(checkpoint(a,s).relative_to(P.parent)):sha(checkpoint(a,s)) for a in ARMS for s in (1,2,3)};write('input_checkpoint_manifest.json',inputs)
 write('protocol.json',dict(experiment='v26 history-only distant-repeat subgroup',seeds=[1,2,3],arms=list(ARMS),conditions=list(CONDS),training='none; frozen v23 context2048 and v19 local/GRU; unmatched inherited histories',data='Pinned FineWeb rows1000–1099 cap4097; all nonoverlapping2048-input windows final128 targets; UTF8 bytes0–255',subgroup='Before scores, for history raw[:t], t1921..2048, last8 bytes have >=1 matching context with observed follower j satisfying8<=j<min(t-512,1408); no matching complete pair inside suffix: t-512+8<=j<t. Current target raw[t] never consulted. Entire far9-byte pair lies in replaced prefix and precedes suffix; retained-middle matches may remain. Crossing pairs do not qualify as far or inside.',intervention='v25 donor first1408 bytes from next eligible different full-document-hash row cyclically; preserve final641 including all128 targets',copy='Complete normalized LM longest available exact context orders4/8/16; one uniform pseudocount. This differs from fixed8 subgroup. Copy suffix requires complete pair in last512 observed bytes. Hybrid uses fixed v20 weights context2048 .6,local .3,GRU .2, falling back to neural when no copy match; no tuning.',comparisons='All-target primary and fixed subgroup secondary. donor-original, original-suffix512. Hybrid additional suffixCopyOnly holds original neural logits fixed while shortening only copy memory. Neural suffix reset effective385–512 unlike copy sliding512.',statistics='Average seeds then paired recipient document-cluster5000 bootstrap seed26; subgroup ratio of sums, omit zero-denominator replicates and report them; convenience cohort and donor dependence limitations.',timing='first8 documents>=2080 bytes,all3seeds and3arms,original2048 neural/hybrid paths,one full warmup then measured prefill+32 observed updates,complete256 softmax,randomized seed260026; compact cache exact parity tested',source_sha256={f.name:sha(f) for f in P.glob('*.py')},input_manifest_sha256=sha(P/'input_checkpoint_manifest.json'),data_manifest_sha256=sha(P/'data_manifest.json'),subgroup_sha256=sha(P/'subgroup.npy'),cpu_budget_seconds=900,main_cpu_limit=650,timing_cpu_limit=120,torch=torch.__version__,platform=platform.platform()))
 print('Frozen',len(items),'windows; subgroup',int(masks.sum()),'/',masks.size,flush=True)
def verify():
 pr=json.loads((P/'protocol.json').read_text())
 for f,h in pr['source_sha256'].items():assert sha(P/f)==h
 assert sha(P/'subgroup.npy')==pr['subgroup_sha256']
 assert sha(P/'input_checkpoint_manifest.json')==pr['input_manifest_sha256'];assert sha(P/'data_manifest.json')==pr['data_manifest_sha256']
 for f,h in json.loads((P/'input_checkpoint_manifest.json').read_text()).items():assert sha(P.parent/f)==h
@torch.inference_mode()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(650,655));start=time.process_time();verify();items=data();arrays={};rows=[]
 copies={c:copy_scores(items,c) for c in CONDS}
 for c,(p,active) in copies.items():arrays['copy_'+c]=-np.log2(p)
 for s in (1,2,3):
  for arm in ARMS:
   m=load(arm,s);neural={c:score(m,arm,items,c) for c in CONDS}
   for cond,a in neural.items():
    arrays[f'{arm}_seed{s}_{cond}']=a
    p,active=copies[cond];w=ALPHAS[arm]*active
    arrays[f'{arm}Hybrid_seed{s}_{cond}']=-np.log2((1-w)*np.exp2(-a)+w*p)
   p,active=copies['suffix512'];w=ALPHAS[arm]*active
   arrays[f'{arm}Hybrid_seed{s}_suffixCopyOnly']=-np.log2((1-w)*np.exp2(-neural['original2048'])+w*p)
   print(arm,s,{c:float(a.mean()) for c,a in neural.items()},flush=True)
   np.savez_compressed(P/'losses.npz',**arrays)
   write('results.json',dict(complete=False,cpu_seconds=time.process_time()-start))
 for k,a in arrays.items():rows.append(dict(key=k,bpb=float(a.mean())))
 write('results.json',dict(rows=rows,complete=True,cpu_seconds=time.process_time()-start))
if __name__=='__main__':
 import sys
 freeze() if '--freeze' in sys.argv else main()
