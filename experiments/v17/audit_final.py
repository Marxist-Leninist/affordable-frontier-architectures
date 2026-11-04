"""Independent artifact, ordinary-forward, real-tail and continuation verification."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,time,hashlib,math,statistics
from pathlib import Path
import torch,numpy as np
from models import LM
from local_attention import localize,allocated_tensor_bytes
from prefill import prefill
P=Path(__file__).parent;V=Path(os.environ.get('V14_DIR',P.parent/'frontier_research_v14'))
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loss(z,y):return float(-z.double().log_softmax(-1).gather(-1,y[...,None]).mean()/math.log(2))
def main():
 begin=time.perf_counter();r=json.loads((P/'results.json').read_text());protocol=json.loads((P/'protocol.json').read_text());ws=json.loads((P/'window_scores.json').read_text());rows=json.loads((P/'rows.json').read_text())['rows'];docs=[x['row']['text'].encode()[:4097] for x in rows];manifest=json.loads((V/'checkpoint_manifest.json').read_text())
 assert r['protocol_sha256']==sha(P/'protocol.json');assert sha(V/'checkpoint_manifest.json')==protocol['v14_checkpoint_manifest_sha256']
 for f,h in protocol['source_hashes'].items():assert sha(P/f)==h
 observations=[]
 for row in r['rows']:
  seed,L=row['seed'],row['input_bytes'];detail=ws[f'seed{seed}_length{L}'];expected=[(i+200,s) for i,d in enumerate(docs) for s in range(0,len(d)-L,L)]
  assert row['windows']==len(expected) and row['targets']==len(expected)*128
  for kind,values in detail.items():
   assert [(x['row_index'],x['start']) for x in values]==expected
   assert abs(statistics.mean(x['bpb'] for x in values)-row['mean_bpb'][kind])<1e-12
  diffs=[b['bpb']-a['bpb'] for a,b in zip(detail['local_monolithic'],detail['local_chunked'])]
  assert abs(max(abs(x) for x in diffs)-row['max_abs_window_bpb_error'])<1e-12
  assert row['strict_quality_parity']==(max(abs(x) for x in diffs)<=2e-6 and row['max_abs_local_logits_error']<=1e-4)
  models={}
  for kind in ['transformer','gru']:
   path=V/'checkpoints'/f'{kind}_seed{seed}_step4000.pt';assert sha(path)==manifest[path.name];m=LM(kind).eval();m.load_state_dict(torch.load(path,weights_only=True));models['full_transformer' if kind=='transformer' else 'gru']=m
  models['local_monolithic']=localize(models['full_transformer']).eval();models['local_chunked']=models['local_monolithic']
  for kind,m in models.items():
   errors=[];logiterrors=[];continuation=[];storage=[]
   for idx in sorted(set([0,len(expected)//2,len(expected)-1])):
    doc,s=expected[idx];raw=docs[doc-200][s:s+L+1];x=torch.tensor(list(raw[:-1]))[None,:];y=torch.tensor(list(raw[-128:]))[None,:]
    with torch.inference_mode():
     ordinary,refstate=m(x,cache=True);z,state=prefill(m,x,128 if kind=='local_chunked' else None,128)
     logiterrors.append(float((z-ordinary[:,-128:]).abs().max()));errors.append(abs(loss(ordinary[:,-128:],y)-detail[kind][idx]['bpb']))
     nxt=torch.tensor(list(raw[-1:]))[None,:];a,_=m(nxt,refstate,True);b,_=m(nxt,state,True);continuation.append(float((a-b).abs().max()))
     storage.append(allocated_tensor_bytes(state))
   assert max(errors)<2e-6,(kind,errors);assert max(logiterrors)<1e-4,(kind,logiterrors);assert max(continuation)<1e-4,(kind,continuation)
   if kind.startswith('local'):assert set(storage)=={130064}
   observations.append(dict(seed=seed,input_bytes=L,model=kind,sampled_windows=len(errors),max_fresh_loss_error=max(errors),max_ordinary_prefill_logit_error=max(logiterrors),max_next_byte_continuation_error=max(continuation),allocated_state_bytes=storage))
 assert r['all_quality_gates_pass']==all(x['strict_quality_parity'] for x in r['rows'])
 out=dict(status='pass',checks=observations,scope='Three actual windows for every model/seed/length row,ordinary full-head forward as independent comparator,plus all stored means/target IDs/gates. No latency rerun;not a full fresh neural rescore.',seconds=time.perf_counter()-begin)
 (P/'audit_final.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
