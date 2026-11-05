"""Independent v26 data, history-only oracle, and full distribution audit. No timing/training."""
import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text())
pr=read(P/'protocol.json');mf=read(P/'data_manifest.json');rows=read(P/'rows.json')['rows'];saved=read(P/'items.json')
assert [r['row_idx'] for r in rows]==list(range(1000,1100))
assert sha((P/'rows.json').read_bytes())==mf['response_sha256']
for r,d in zip(rows,mf['documents']):
 raw=r['row']['text'].encode();assert sha(raw)==d['full_text_sha256'];assert sha(raw[:4097])==d['capped_text_sha256']
for f,h in pr['source_sha256'].items():assert sha((P/f).read_bytes())==h
for f,key in [('input_checkpoint_manifest.json','input_manifest_sha256'),('data_manifest.json','data_manifest_sha256'),('subgroup.npy','subgroup_sha256')]:assert sha((P/f).read_bytes())==pr[key]
for f,h in read(P/'input_checkpoint_manifest.json').items():assert sha((P.parent/f).read_bytes())==h
full={r['row_idx']:r['row']['text'].encode() for r in rows};eligible=[(d,b[:4097]) for d,b in full.items() if len(b)>=2049];items=[];targets=[];raws=[];changedraws=[]
for j,(d,b) in enumerate(eligible):
 donor=next(eligible[(j+k)%len(eligible)] for k in range(1,len(eligible)) if sha(full[eligible[(j+k)%len(eligible)][0]])!=sha(full[d]))
 for start in range(0,len(b)-2048,2048):
  raw=b[start:start+2049];changed=donor[1][:1408]+raw[1408:];raws.append(raw);changedraws.append(changed)
  assert len(raw)==len(changed)==2049 and raw[1408:]==changed[1408:] and raw[:1408]!=changed[:1408]
  assert len(raw[1408:1921])==513 and len(raw[1921:])==128
  targets.append((d,start,raw[1921:]))
  items.append(dict(doc=d,start=start,donor_doc=donor[0],original_sha256=sha(raw),donor_window_sha256=sha(changed)))
assert items==saved
# Independent match starts, not producer's follower-index loop. Limit search to observed bytes.
def starts(hist,key):
 pos=hist.find(key);out=[]
 while pos!=-1:
  if pos+len(key)<len(hist):out.append(pos)
  pos=hist.find(key,pos+1)
 return out
mask=[];retained_middle=0
for raw in raws:
 row=[]
 for t in range(1921,2049):
  hist=raw[:t];ms=starts(hist,hist[-8:]);far=any(s+8<1408 for s in ms);near=any(s>=t-512 for s in ms);selected=far and not near;row.append(selected)
  if selected and any(1408<=s+8<t-512 for s in ms):retained_middle+=1
 mask.append(row)
mask=np.array(mask);assert np.array_equal(mask,np.load(P/'subgroup.npy'));assert mask.any()
# Full independently normalized copy distributions for each scored condition. All-target loss check.
z=np.load(P/'losses.npz');maxerr=0.;sumerr=0.
from copy_cache import CopyCache
from compact_cache import CompactCache
for name,corpus in [('original2048',raws),('donor2048',changedraws),('suffix512',raws)]:
 for wi,raw in enumerate(corpus):
  cache=CopyCache();compact=CompactCache()
  for t,b in enumerate(raw[:-1],1):
   cache.feed(b);compact.feed(b)
   if t<1921:continue
   hist=raw[:t];cut=t-512 if name=='suffix512' else 0;counts=Counter()
   for k in (16,8,4):
    ms=[s for s in starts(hist,hist[-k:]) if s>=cut]
    if ms:
     counts=Counter(hist[s+k] for s in ms);break
   n=sum(counts.values());q=np.array([(counts[b]+1/256)/(n+1) for b in range(256)])
   qprod,np_,_,_=cache.distribution(512 if name=='suffix512' else None);assert np_==n
   assert np.max(np.abs(q-qprod))<1e-15
   if name!='suffix512':assert np.max(np.abs(q-compact.distribution()[0]))<1e-15
   sumerr=max(sumerr,abs(q.sum()-1));loss=-np.log2(q[raw[t]])
   maxerr=max(maxerr,abs(loss-z['copy_'+name][wi,t-1921]))
assert maxerr<1e-12 and sumerr<1e-14
train=(P.parent/'frontier_research_v8/train.txt').read_bytes()
train_matches=[dict(doc=d,start=s) for d,s,t in targets if t in train];prior=[]
for n in range(15,26):
 q=P.parent/f'frontier_research_v{n}'/'rows.json'
 if q.exists():
  old=read(q);rr=old['rows'] if isinstance(old,dict) and 'rows' in old else []
  for r in rr:
   if isinstance(r,dict) and isinstance(r.get('row'),dict) and 'text' in r['row']:prior.append((n,r['row_idx'],r['row']['text'].encode()))
prior_matches=[dict(doc=d,start=s,prior_version=n,prior_row=row) for d,s,t in targets for n,row,b in prior if t in b]
within=[dict(doc=d,start=s,other_doc=e) for d,s,t in targets for e,b in full.items() if e!=d and t in b]
out=dict(frozen_hashes_verified=True,rows=100,eligible_documents=len(eligible),windows=len(items),targets_per_seed_condition=128*len(items),capped_bytes=sum(len(b[:4097]) for b in full.values()),donor_different_full_document_hash=True,all_prefixes_actually_changed=True,deterministic_next_eligible_donor=True,identical_final641_bytes=True,preserved_real_prior_bytes_before_earliest_target=513,independent_history_only_mask_verified=True,selected_targets=int(mask.sum()),selected_windows=int(mask.any(1).sum()),selected_documents=len({items[k]['doc'] for k in range(len(items)) if mask[k].any()}),selected_targets_with_retained_middle_followers=retained_middle,copy_full_distributions_verified=True,compact_full_distribution_parity=True,copy_max_loss_error=maxerr,copy_max_sum_error=sumerr,exact_128_target_spans_in_full_wikitext_training=train_matches,exact_128_target_spans_in_previous_evaluation_documents=prior_matches,exact_128_target_spans_in_other_current_documents=within,prior_document_records_checked=len(prior),duplicate_128_target_spans=len(targets)-len(set(t for _,_,t in targets)),limits='Exact128-byte span checks do not exclude shorter overlap,near duplicates,semantic overlap or common upstream provenance. Dataset snapshot is hashed,not independently refetched.')
(P/'independent_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
