"""v16: fixed-span inference intervention on a fresh FineWeb slice; no tuning."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,time,hashlib,math,statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import LM
from ngram import Ngram
from local_attention import localize,tensor_bytes,allocated_tensor_bytes
P=Path(__file__).parent;V=Path(os.environ.get('V14_DIR',P.parent/'frontier_research_v14'));D=Path(os.environ.get('WIKITEXT_DATA',P.parent/'frontier_research_v8'))
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(f,x):(P/f).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def windows(docs,n):return [(i,s,d[s:s+n+1]) for i,d in enumerate(docs) for s in range(0,len(d)-n,n)]
def load():
 mf=json.loads((P/'data_manifest.json').read_text());assert sha(P/'rows.json')==mf['response_sha256'];rows=json.loads((P/'rows.json').read_text())['rows'];docs=[r['row']['text'].encode()[:4097] for r in rows]
 assert [r['row_idx'] for r in rows]==list(range(100,200))
 for doc,row in zip(docs,mf['documents']):assert hashlib.sha256(doc).hexdigest()==row['capped_text_sha256']
 vp=json.loads((V/'protocol.json').read_text());train=(D/'train.txt').read_bytes();assert hashlib.sha256(train).hexdigest()==vp['dataset']['splits']['train']['text_sha256'];assert sha(V/'train_starts.npy')==vp['ledger_sha256'];assert sha(P/'models.py')==sha(V/'models.py');assert sha(P/'ngram.py')==sha(V/'ngram.py')
 return docs,train,mf
@torch.inference_mode()
def score(model,items):
 out=[]
 for start in range(0,len(items),16):
  batch=items[start:start+16];a=torch.tensor([list(r[2]) for r in batch],dtype=torch.long);z=model(a[:,:-1])[:,-128:];y=a[:,1:][:,-128:]
  loss=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').reshape(len(batch),128)
  out.extend(dict(doc=i+100,start=s,bpb=float(l.double().mean()/math.log(2))) for (i,s,_),l in zip(batch,loss))
 return out
def statscore(model,items):
 out=[]
 for i,s,raw in items:
  n=len(raw)-1;total=0.
  for pos in range(n-128,n):total-=math.log2(model.prob(raw[max(0,pos-3):pos+1],raw[pos+1]))
  out.append(dict(doc=i+100,start=s,bpb=total/128))
 return out
def aggregate(rows):return dict(bpb=statistics.mean(r['bpb'] for r in rows),windows=len(rows),targets=len(rows)*128,documents=len(set(r['doc'] for r in rows)))
def paired(a,b):
 assert [(r['doc'],r['start']) for r in a]==[(r['doc'],r['start']) for r in b]
 delta=[y['bpb']-x['bpb'] for x,y in zip(a,b)]
 return dict(b_minus_a_bpb=statistics.mean(delta),fraction_windows_b_better=statistics.mean(d<0 for d in delta),abs_aggregate_gap_le_002=abs(statistics.mean(delta))<=.02)
def main():
 t=time.perf_counter();docs,train,mf=load();long=windows(docs,512);short=[(i,s,r[-129:]) for i,s,r in long];primary=windows(docs,128)
 protocol=dict(experiment='v16 weight-preserving128-span attention on fresh HF slice',data_manifest_sha256=sha(P/'data_manifest.json'),source_hashes={f:sha(P/f) for f in ['run.py','local_attention.py','models.py','ngram.py','prepare_data.py']},selection='No new fitting or hyperparameter search. W128 is fixed from original training context before scoring fresh viewer rows100-199. Final4000-step v14 seeds1/2/3 fixed.',paired='Same128 tail targets of complete512-target document windows; full512 Transformer, local128-span512 Transformer, reset128 Transformer, full512 GRU, reset128 GRU, full512 ngram.',quality_gate='Two-sided abs aggregate local512 minus reset128 Transformer loss<=.02 for each paired seed; descriptive screen only, not statistical equivalence. Also report all baseline gaps.',cache='Keep127 prior K/V positions per attention layer plus absolute-position int64 tensor; clone storage; no distant KV retained. Layerwise128 span can transmit information beyond128 raw bytes through depth.',prefix128_invariance='All primary128-window local logits should match baseline within floating-point tolerance; this is not proof of512 quality.',v14_manifest_sha256=sha(V/'checkpoint_manifest.json'),v14_results_sha256=sha(V/'results.json'),paired_windows=len(long),primary_windows=len(primary),torch=torch.__version__)
 write('protocol.json',protocol);manifest=json.loads((V/'checkpoint_manifest.json').read_text());results=[];details={}
 for kind in ['transformer','gru']:
  for seed in [1,2,3]:
   path=V/'checkpoints'/f'{kind}_seed{seed}_step4000.pt';assert sha(path)==manifest[path.name];base=LM(kind).eval();base.load_state_dict(torch.load(path,weights_only=True));a=score(base,short);b=score(base,long);pr=score(base,primary);row=dict(kind=kind,seed=seed,checkpoint_sha256=sha(path),reset128=aggregate(a),full512=aggregate(b),primary128=aggregate(pr),full_vs_reset=paired(a,b));details[f'{kind}_{seed}']=dict(reset=a,full=b,primary=pr)
   if kind=='transformer':
    local=localize(base).eval();assert all(torch.equal(v,local.state_dict()[k]) for k,v in base.state_dict().items());c=score(local,long);lp=score(local,primary);row.update(local512=aggregate(c),local_vs_reset=paired(a,c),local_vs_full=paired(b,c),primary128_max_window_bpb_delta=max(abs(x['bpb']-y['bpb']) for x,y in zip(pr,lp)));details[f'{kind}_{seed}']['local']=c
   results.append(row);write('partial.json',results);print(row,flush=True)
 stat=Ngram();fit=time.perf_counter()
 for start in np.load(V/'train_starts.npy').flatten():stat.add(train[start:start+129])
 fit=time.perf_counter()-fit;stat.order=json.loads((V/'selection.json').read_text())['ngram_order'];sr=statscore(stat,long);details['ngram']=dict(full=sr)
 write('window_scores.json',details);write('results.json',dict(protocol_sha256=sha(P/'protocol.json'),neural=results,ngram=dict(full512=aggregate(sr),rebuild_seconds=fit),total_seconds=time.perf_counter()-t));print('DONE',flush=True)
if __name__=='__main__':main()
