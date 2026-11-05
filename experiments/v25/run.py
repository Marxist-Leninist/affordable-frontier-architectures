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
P=Path(__file__).parent
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
ARMS=('context2048','mixed','reset512','local','gru');CONDS=('original2048','donor2048','suffix512')
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
def freeze():
 items=data();write('items.json',[{k:v for k,v in i.items() if k not in ('raw','changed')} for i in items])
 inputs={str(checkpoint(a,s).relative_to(P.parent)):sha(checkpoint(a,s)) for a in ARMS for s in (1,2,3)};write('input_checkpoint_manifest.json',inputs)
 write('protocol.json',dict(experiment='v25 learned distant-content specificity',seeds=[1,2,3],arms=list(ARMS),conditions=list(CONDS),training='none; frozen v23 context2048/reset512, v24 mixed, v19 local/GRU',data='Pinned FineWeb-Edu rows900-999 first4097 UTF8 bytes/document, no joins/content selection; all nonoverlapping2048-input windows final128 targets',intervention='Replace first1408 bytes of each2049-byte window by first1408 bytes of deterministic next eligible different full-document-hash document, cycling ascending row order. Preserve final641 bytes including all128 targets and513 real preceding bytes before earliest target. Original/donor equal input lengths/RoPE positions.',controls='suffix512 feeds final512 inputs, effective385-512 contexts; local finite receptive field; GRU near-constant distant influence but not exact invariance',statistics='Mean seeds then paired document-cluster5000 bootstrap seed25; donor-original and original-suffix, all arms. Conditional convenience sample; no population guarantee.',limits='Artificial unrelated-prefix boundary; effects can include disruption. Original better than donor does not establish beneficial context if suffix remains better. Unequal inherited training; no distant reasoning/frontier capability claim.',timing='Original paths only,first8 documents >=2080 bytes,3seeds,all5arms,512/2048 contexts,one full warmup then one measured prefill+32 observed updates/full256way softmax;batch1 randomized order seed250025;descriptive byte units',source_sha256={f.name:sha(f) for f in P.glob('*.py')},input_manifest_sha256=sha(P/'input_checkpoint_manifest.json'),data_manifest_sha256=sha(P/'data_manifest.json'),cpu_budget_seconds=900,main_cpu_limit=700,timing_cpu_limit=100,torch=torch.__version__,platform=platform.platform()))
 print('Frozen',len(items),'windows')
def verify():
 pr=json.loads((P/'protocol.json').read_text())
 for f,h in pr['source_sha256'].items():assert sha(P/f)==h
 assert sha(P/'input_checkpoint_manifest.json')==pr['input_manifest_sha256'];assert sha(P/'data_manifest.json')==pr['data_manifest_sha256']
 for f,h in json.loads((P/'input_checkpoint_manifest.json').read_text()).items():assert sha(P.parent/f)==h
@torch.inference_mode()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(700,705));start=time.process_time();verify();items=data();arrays={};rows=[]
 for s in (1,2,3):
  for arm in ARMS:
   m=load(arm,s)
   for cond in CONDS:
    a=score(m,arm,items,cond);arrays[f'{arm}_seed{s}_{cond}']=a;row=dict(arm=arm,seed=s,condition=cond,bpb=float(a.mean()));rows.append(row);print(row,flush=True)
   np.savez_compressed(P/'losses.npz',**arrays);write('results.json',dict(rows=rows,complete=False,cpu_seconds=time.process_time()-start))
 write('results.json',dict(rows=rows,complete=True,cpu_seconds=time.process_time()-start))
if __name__=='__main__':
 import sys
 freeze() if '--freeze' in sys.argv else main()
