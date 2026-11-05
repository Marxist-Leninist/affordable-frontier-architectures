"""Independent data and frozen-source audit; stdlib only, no training/timing."""
import json, hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text())
pr=read(P/'protocol.json'); mf=read(P/'data_manifest.json'); rows=read(P/'rows.json')['rows']; saved=read(P/'items.json')
assert [r['row_idx'] for r in rows]==list(range(900,1000))
assert sha((P/'rows.json').read_bytes())==mf['response_sha256']
for r,d in zip(rows,mf['documents']):
 raw=r['row']['text'].encode();assert sha(raw)==d['full_text_sha256'];assert sha(raw[:4097])==d['capped_text_sha256']
for f,h in pr['source_sha256'].items():assert sha((P/f).read_bytes())==h
for f,key in [('input_checkpoint_manifest.json','input_manifest_sha256'),('data_manifest.json','data_manifest_sha256')]:assert sha((P/f).read_bytes())==pr[key]
for f,h in read(P/'input_checkpoint_manifest.json').items():assert sha((P.parent/f).read_bytes())==h
full={r['row_idx']:r['row']['text'].encode() for r in rows}; eligible=[(d,b[:4097]) for d,b in full.items() if len(b)>=2049]; items=[]; targets=[]
for j,(d,b) in enumerate(eligible):
 donor=next(eligible[(j+k)%len(eligible)] for k in range(1,len(eligible)) if sha(full[eligible[(j+k)%len(eligible)][0]])!=sha(full[d]))
 for start in range(0,len(b)-2048,2048):
  raw=b[start:start+2049]; changed=donor[1][:1408]+raw[1408:]
  assert len(raw)==len(changed)==2049 and raw[1408:]==changed[1408:]
  assert len(raw[1408:1921])==513 and len(raw[1921:])==128
  assert raw[:1408]!=changed[:1408]
  target=raw[1921:]; targets.append((d,start,target))
  items.append(dict(doc=d,start=start,donor_doc=donor[0],original_sha256=sha(raw),donor_window_sha256=sha(changed)))
assert items==saved
train=(P.parent/'frontier_research_v8/train.txt').read_bytes()
train_matches=[dict(doc=d,start=s) for d,s,t in targets if t in train]
prior=[]
for n in range(15,25):
 q=P.parent/f'frontier_research_v{n}'/'rows.json'
 if q.exists():
  old=read(q); rr=old['rows'] if isinstance(old,dict) and 'rows' in old else []
  for r in rr:
   if isinstance(r,dict) and isinstance(r.get('row'),dict) and 'text' in r['row']:prior.append((n,r['row_idx'],r['row']['text'].encode()))
prior_matches=[dict(doc=d,start=s,prior_version=n,prior_row=row) for d,s,t in targets for n,row,b in prior if t in b]
within=[dict(doc=d,start=s,other_doc=e) for d,s,t in targets for e,b in full.items() if e!=d and t in b]
out=dict(frozen_hashes_verified=True,rows=100,eligible_documents=len(eligible),windows=len(items),targets_per_seed_condition=128*len(items),capped_bytes=sum(len(b[:4097]) for b in full.values()),donor_different_full_document_hash=True,all_prefixes_actually_changed=True,deterministic_next_eligible_donor=True,identical_final641_bytes=True,preserved_real_prior_bytes_before_earliest_target=513,exact_128_target_spans_in_full_wikitext_training=train_matches,exact_128_target_spans_in_previous_evaluation_documents=prior_matches,exact_128_target_spans_in_other_current_documents=within,prior_document_records_checked=len(prior),duplicate_128_target_spans=len(targets)-len(set(t for _,_,t in targets)),limits='Exact 128-byte target-span and full-document identity checks do not rule out shorter overlap, near duplicates, semantic overlap, or common upstream provenance. Prior dataset check covers available v15-v24 rows snapshots only.')
(P/'independent_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
