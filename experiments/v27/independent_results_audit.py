"""Post-freeze independent reconstruction. Does not edit scientific sources."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json,hashlib,math
from pathlib import Path
import numpy as np
import torch
from run import P,data,load,score,verify,ARMS,SCOPES,ALPHAS
from bounded_cache import BoundedCache
from independent_cache_audit import oracle
from pruning import local_prefill
from prefill import prefill

def sha(b):return hashlib.sha256(b).hexdigest()
@torch.inference_mode()
def main():
 verify();items=data();saved=np.load(P/'losses.npz');result=json.loads((P/'results.json').read_text());assert result['complete'];ledger=json.loads((P/'items.json').read_text())
 assert ledger==[dict(doc=i['doc'],start=i['start'],sha256=sha(i['raw'])) for i in items]
 for k,v in saved.items():
  assert v.shape==(len(items),128) and np.isfinite(v).all()
  if not k.startswith('active'):assert abs(v.mean()-result['bpb'][k])<1e-12
 fullchecks=0;initchecks=0
 for scope in SCOPES:
  probs=[];active=[]
  for item in items:
   raw=item['raw'];pp=[];aa=[]
   for t in range(3969,4097):
    q,n,k=oracle(raw[:t],None if scope=='full' else scope);pp.append(q[raw[t]]);aa.append(bool(n));fullchecks+=1
   probs.append(pp);active.append(aa)
   if scope!='full':
    c=BoundedCache(scope);d=BoundedCache(scope)
    for b in raw[:4096]:c.feed(b)
    for b in raw[4096-scope:4096]:d.feed(b)
    for b in raw[4065:4097]:
     np.testing.assert_array_equal(c.distribution()[0],d.distribution()[0]);assert c.distribution()[1:]==d.distribution()[1:];initchecks+=1;c.feed(b);d.feed(b)
  probs=np.asarray(probs);active=np.asarray(active)
  np.testing.assert_array_equal(saved[f'active_{scope}'],active)
  np.testing.assert_array_equal(saved[f'copy_{scope}'],-np.log2(probs))
  for arm in ARMS:
   for seed in (1,2,3):
    w=ALPHAS[arm]*active;expected=-np.log2((1-w)*np.exp2(-saved[f'{arm}_seed{seed}_neural'])+w*probs)
    np.testing.assert_array_equal(saved[f'{arm}_seed{seed}_{scope}'],expected)
 # Native whole-head path, optimized scoring and causal future-invariance.
 checks=[]
 for arm in ARMS:
  for seed in (1,2,3):
   m=load(arm,seed);raw=torch.tensor(list(items[0]['raw']))[None,:];x=raw[:,:-1];native=m(x)[:,-128:];y=raw[:,-128:]
   loss=(-native.double().log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)/math.log(2)).numpy()[0]
   optimized,_=local_prefill(m,x,128,prune=True) if arm=='local' else prefill(m,x,None,128)
   logit_error=float((native-optimized).abs().max());loss_error=float(abs(loss-saved[f'{arm}_seed{seed}_neural'][0]).max())
   assert logit_error<1e-4 and loss_error<1e-4
   changed=x.clone();changed[:,3969:]=(changed[:,3969:]+71)%256
   future_error=float((m(changed)[:,3968]-native[:,0]).abs().max());assert future_error==0
   checks.append(dict(arm=arm,seed=seed,max_logit_error=logit_error,max_saved_loss_error=loss_error,future_change_error=future_error))
 # Exact target-span search; this is not semantic or shorter overlap detection.
 prior=[]
 for path in sorted(P.parent.glob('frontier_research_v*/rows.json')):
  if path.parent==P:continue
  try:
   rows=json.loads(path.read_text())['rows'];prior.extend((str(path.parent.name),r['row_idx'],r['row']['text'].encode()) for r in rows)
  except (KeyError,TypeError):pass
 train=(P.parent/'frontier_research_v8/train.txt').read_bytes();current=json.loads((P/'rows.json').read_text())['rows'];overlap=[]
 for item in items:
  span=item['raw'][-128:]
  if span in train:overlap.append(dict(doc=item['doc'],source='WikiText training'))
  for source,doc,raw in prior:
   if span in raw:overlap.append(dict(doc=item['doc'],source=source,other_doc=doc))
  for other in current:
   if other['row_idx']!=item['doc'] and span in other['row']['text'].encode():overlap.append(dict(doc=item['doc'],source='current other document',other_doc=other['row_idx']))
 out=dict(status='pass',windows=len(items),documents=len({i['doc'] for i in items}),targets=len(items)*128,oracle_scored_distributions=fullchecks,suffix_initialization_and_continuation_checks=initchecks,aggregate_arrays=len(result['bpb']),native_neural_checks=checks,prior_document_records=len(prior),exact_target_span_overlap=overlap)
 (P/'independent_results_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
