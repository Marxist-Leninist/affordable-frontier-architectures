"""Frozen receptive-field prefix pruning, predeclared exactness gate, no tuning."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time,math,statistics
from pathlib import Path
import torch
from torch.nn import functional as F
from models import LM
from local_attention import localize
from prefill import prefill
from pruning import local_prefill
P=Path(__file__).parent;V=Path(os.environ.get('V14_DIR',P.parent/'frontier_research_v14'));torch.set_num_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(f,x):(P/f).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load():
 mf=json.loads((P/'data_manifest.json').read_text());assert sha(P/'rows.json')==mf['response_sha256'];rows=json.loads((P/'rows.json').read_text())['rows'];docs=[r['row']['text'].encode()[:4097] for r in rows]
 for d,r in zip(docs,mf['documents']):assert hashlib.sha256(d).hexdigest()==r['capped_text_sha256']
 return docs

def models(seed):
 manifest=json.loads((V/'checkpoint_manifest.json').read_text());out={}
 for kind in ['transformer','gru']:
  m=LM(kind).eval();p=V/'checkpoints'/f'{kind}_seed{seed}_step4000.pt';assert sha(p)==manifest[p.name];m.load_state_dict(torch.load(p,weights_only=True));out[kind]=m
 out['local']=localize(out['transformer']);return out

@torch.inference_mode()
def main():
 start=time.perf_counter();docs=load();write('protocol.json',dict(dataset_manifest_sha256=sha(P/'data_manifest.json'),selection='Fresh FineWeb rows300-399, first4097 bytes, nonoverlapping2048 input+next target windows, last128 identical targets. No tuning.',hypothesis='Two layers span128: final distribution receptive field255 inputs. Last128 distributions require382 inputs. Preserve absolute RoPE counters after initial reset-position control failed declared gates. Prefix pruning exact up to floating-point effects, does not add long-range capability.',gate='local max logits abs error <=1e-4 and each window bpb abs error <=2e-6',baseline='Same frozen v14 checkpoints; local baseline v17 chunk128 + final-head-only. Both paths use chunk128 for local. Full Transformer and GRU suffix-pruning controls are diagnostic and may not be quality matched.',checkpoint_manifest_sha256=sha(V/'checkpoint_manifest.json')))
 items=[(i+300,s,d[s:s+2049]) for i,d in enumerate(docs) for s in range(0,len(d)-2048,2048)];results=[];details=[]
 for seed in [1,2,3]:
  for kind,m in models(seed).items():
   errors=[];losses=[[],[]];logitmax=0
   for begin in range(0,len(items),4):
    batch=items[begin:begin+4];raw=torch.tensor([list(x[2]) for x in batch]);x=raw[:,:-1];y=raw[:,-128:];zs=[]
    for j,xx in enumerate([x,x[:,-382:]]):
     z,_=local_prefill(m,x,128,prune=bool(j)) if kind=='local' else prefill(m,xx,None,128);zs.append(z);l=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').reshape(len(batch),128).double().mean(1)/math.log(2);losses[j].extend(l.tolist())
    logitmax=max(logitmax,float((zs[0]-zs[1]).abs().max()))
   for item,a,b in zip(items,*losses):details.append(dict(seed=seed,model=kind,row_index=item[0],start=item[1],full_bpb=a,pruned_bpb=b,delta=b-a))
   errors=[b-a for a,b in zip(*losses)];r=dict(seed=seed,model=kind,windows=len(items),targets=len(items)*128,full_bpb=statistics.mean(losses[0]),pruned_bpb=statistics.mean(losses[1]),delta=statistics.mean(errors),max_abs_window_bpb_error=max(map(abs,errors)),max_abs_logit_error=logitmax,strict_parity=max(map(abs,errors))<=2e-6 and logitmax<=1e-4);results.append(r);print(r,flush=True)
 write('scores.json',details);write('results.json',dict(rows=results,wall_seconds=time.perf_counter()-start,documents=len(set(x[0] for x in items)),windows=len(items),torch=torch.__version__))
if __name__=='__main__':main()
