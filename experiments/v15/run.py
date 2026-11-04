"""Frozen v14 transfer to FineWeb-Edu; no fitting/selection on new corpus."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,math,statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import LM,state_bytes
from ngram import Ngram
P=Path(__file__).parent;V=Path(os.environ.get('V14_DIR',str(P.parent/'frontier_research_v14')));D=Path(os.environ.get('WIKITEXT_DATA',str(P.parent/'frontier_research_v8')))
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x):(P/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def windows(docs,length):
 return [(i,s,raw[s:s+length+1]) for i,raw in enumerate(docs) for s in range(0,len(raw)-length,length)]
def load_inputs():
 mf=json.loads((P/'data_manifest.json').read_text());assert sha(P/'rows.json')==mf['response_sha256'];rows=json.loads((P/'rows.json').read_text())['rows'];docs=[r['row']['text'].encode()[:4097] for r in rows]
 for d,rec in zip(docs,mf['documents']):assert hashlib.sha256(d).hexdigest()==rec['capped_text_sha256']
 vresults=json.loads((V/'results.json').read_text());vprotocol=json.loads((V/'protocol.json').read_text());manifest=json.loads((V/'checkpoint_manifest.json').read_text())
 assert sha(P/'models.py')==sha(V/'models.py');assert sha(P/'ngram.py')==sha(V/'ngram.py')
 for file,h in manifest.items():assert sha(V/'checkpoints'/file)==h
 raw=(D/'train.txt').read_bytes();assert hashlib.sha256(raw).hexdigest()==vprotocol['dataset']['splits']['train']['text_sha256'];assert sha(V/'train_starts.npy')==vprotocol['ledger_sha256']
 return mf,docs,vresults,raw
@torch.inference_mode()
def neural_scores(m,items,last=128):
 m.eval();out=[]
 for start in range(0,len(items),16):
  batch=items[start:start+16];a=torch.tensor([list(x[2]) for x in batch],dtype=torch.long);logits=m(a[:,:-1])[:,-last:];y=a[:,1:][:,-last:]
  losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape(len(batch),last)
  for item,loss in zip(batch,losses):out.append(dict(doc=item[0],start=item[1],bpb=float(loss.double().mean()/math.log(2)),targets=last))
 return out
def stat_scores(m,items,last=128):
 out=[]
 for doc,start,raw in items:
  n=len(raw)-1;loss=0.
  for pos in range(n-last,n):loss-=math.log2(m.prob(raw[max(0,pos-m.order+1):pos+1],raw[pos+1]))
  out.append(dict(doc=doc,start=start,bpb=loss/last,targets=last))
 return out
def aggregate(rows):return dict(bpb=statistics.mean(r['bpb'] for r in rows),targets=sum(r['targets'] for r in rows),windows=len(rows),documents=len(set(r['doc'] for r in rows)))
def pair_summary(short,long):
 assert [(x['doc'],x['start']) for x in short]==[(x['doc'],x['start']) for x in long]
 delta=[b['bpb']-a['bpb'] for a,b in zip(short,long)];bydoc={}
 for a,d in zip(short,delta):bydoc.setdefault(a['doc'],[]).append(d)
 vals=[statistics.mean(v) for v in bydoc.values()]
 # Fixed document bootstrap is descriptive convenience-sample uncertainty, not population inference.
 rng=np.random.default_rng(15112026);draws=np.array([statistics.mean(rng.choice(vals,len(vals),replace=True)) for _ in range(1000)])
 return dict(long_minus_short_bpb=statistics.mean(delta),fraction_windows_improved=statistics.mean(d<0 for d in delta),equal_document_mean_delta_bpb=statistics.mean(vals),descriptive_document_bootstrap_interval=[float(x) for x in np.quantile(draws,[.025,.975])],documents=len(vals),windows=len(delta))
@torch.inference_mode()
def timing(model,docs,length):
 seqs=[d for d in docs if len(d)>=length+32][:8];pre=[];dec=[];total=[];cache=[]
 for raw in seqs:
  x=torch.tensor(list(raw[:length+32]),dtype=torch.long)[None,:]
  for rep in range(5):
   t=time.perf_counter()
   if isinstance(model,Ngram):c=raw[:length][-model.order:];np.array([model.prob(c,y) for y in range(256)])
   else:z,s=model(x[:,:length],cache=True);torch.softmax(z[:,-1],-1)
   t1=time.perf_counter()
   for j in range(32):
    if isinstance(model,Ngram):c=(c+raw[length+j:length+j+1])[-model.order:];np.array([model.prob(c,y) for y in range(256)])
    else:z,s=model(x[:,length+j:length+j+1],s,True);torch.softmax(z[:,-1],-1)
   t2=time.perf_counter()
   if rep>=2:pre.append(1000*(t1-t));dec.append(1000*(t2-t1)/32);total.append(1000*(t2-t));cache.append(model.order if isinstance(model,Ngram) else state_bytes(s))
 return dict(prefix_bytes=length,contexts=len(seqs),retained_probes=len(total),prefill_p50_ms=float(np.median(pre)),sequence_average_decode_p50_ms=float(np.median(dec)),end_to_end_p50_ms=float(np.median(total)),end_to_end_p95_ms=float(np.quantile(total,.95)),final_state_bytes=max(cache),state_scope='logical suffix only' if isinstance(model,Ngram) else 'tensor bytes only',distributions_per_probe=33)
def main():
 t=time.perf_counter();mf,docs,vr,train=load_inputs();primary=windows(docs,128);long=windows(docs,512);short=[(i,s,r[-129:]) for i,s,r in long]
 protocol=dict(experiment='v15 frozen-model second-corpus transfer and paired context sensitivity',dataset=mf,models='v14 final4000-step checkpoints seeds1/2/3; v14 exact ngram table; no adaptation, mixtures, LR/seed/checkpoint choice on new corpus',primary='All complete per-document128-target windows from first4097 bytes; no cross-document state',context='Same last128 target bytes of each complete512-target document window; compare fresh128-input suffix versus full512-input prefix. IDs use parent512-window start.',quality_screen='Prespecified pairwise abs aggregate bpb gap<=.02 descriptive only; not statistical equivalence. No winner chosen for deployment.',source_hashes={f:sha(P/f) for f in ['run.py','models.py','ngram.py','prepare_data.py']},v14_results_sha256=sha(V/'results.json'),v14_checkpoint_manifest_sha256=sha(V/'checkpoint_manifest.json'),ledger_sha256=sha(V/'train_starts.npy'),primary_windows=len(primary),paired_windows=len(long),seeds=[1,2,3],timing='First8 qualifying documents for each prefix length,5 repeats,2warmups,33 distributions; float32 single-thread CPU; no service dollar mapping',torch=torch.__version__)
 write('protocol.json',protocol)
 # Cheap exact-overlap diagnostic; no removal or test-driven filtering.
 overlap=sum(r in train for _,_,r in primary);write('overlap.json',dict(exact_129_byte_primary_windows_found_in_wikitext_training=overlap,total=len(primary),scope='Exact whole-window substring only; not a near-duplicate or training-contamination guarantee'))
 stat=Ngram();fit=time.perf_counter()
 for start in np.load(V/'train_starts.npy').flatten():stat.add(train[start:start+129])
 rebuild=time.perf_counter()-fit;stat.order=json.loads((V/'selection.json').read_text())['ngram_order']
 finals=[];details={}
 for kind in ['transformer','gru']:
  for seed in [1,2,3]:
   prev=next(r for r in vr['final'] if r['kind']==kind and r['seed']==seed);cp=next(c for c in prev['evaluated_checkpoints'] if c['step']==4000);path=V/'checkpoints'/cp['checkpoint_file'];assert sha(path)==cp['checkpoint_sha256']
   model=LM(kind).eval();model.load_state_dict(torch.load(path,weights_only=True));a=neural_scores(model,primary);b=neural_scores(model,short);c=neural_scores(model,long)
   key=f'{kind}_{seed}';details[key]=dict(primary=a,short=b,long=c);finals.append(dict(kind=kind,seed=seed,checkpoint_sha256=cp['checkpoint_sha256'],primary=aggregate(a),context_short=aggregate(b),context_long=aggregate(c),paired=pair_summary(b,c)));write('partial.json',finals);print(key,finals[-1],flush=True)
 a=stat_scores(stat,primary);b=stat_scores(stat,short);c=stat_scores(stat,long);details['ngram']=dict(primary=a,short=b,long=c);sr=dict(primary=aggregate(a),context_short=aggregate(b),context_long=aggregate(c),paired=pair_summary(b,c),rebuild_seconds=rebuild)
 write('window_scores.json',details)
 # Timing after all scores, no fitting overlap within this process.
 for row in finals:
  path=V/'checkpoints'/f"{row['kind']}_seed{row['seed']}_step4000.pt";model=LM(row['kind']).eval();model.load_state_dict(torch.load(path,weights_only=True));row['timing']=[timing(model,docs,n) for n in [128,512]]
 sr['timing']=[timing(stat,docs,n) for n in [128,512]]
 write('results.json',dict(protocol_sha256=sha(P/'protocol.json'),neural=finals,ngram=sr,total_seconds=time.perf_counter()-t));print('DONE',flush=True)
if __name__=='__main__':main()
