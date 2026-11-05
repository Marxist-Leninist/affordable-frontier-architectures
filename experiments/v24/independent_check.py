"""Read-only artifact reconstruction; no training or timed inference."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json, hashlib, time
from pathlib import Path
from collections import Counter
import numpy as np
P=Path(__file__).parent; base=P.parent; start=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(P/'protocol.json');out={'source_hashes':{f:sha(P/f)==h for f,h in protocol['source_sha256'].items()}}
# Timing can be amended independently before its first run, without changing frozen training protocol.
out['protocol_manifest_hashes']={key:sha(P/f)==protocol[key] for key,f in [('inputs_sha256','input_checkpoint_manifest.json'),('data_sha256','data_manifest.json')]}
ledger=np.load(P/'train_starts.npy');train=(base/'frontier_research_v8/train.txt').read_bytes()
out['ledger_reconstructed']=bool(np.array_equal(ledger,np.random.default_rng(23112026).integers(0,len(train)-2048,size=250,dtype=np.int64)))
out['ledger_matches_v23']=sha(P/'train_starts.npy')==sha(base/'frontier_research_v23/train_starts.npy')==protocol['ledger_sha256']
out['train_hash_matches']=hashlib.sha256(train).hexdigest()==read(P/'wikitext_manifest.json')['splits']['train']['text_sha256']
inputs=read(P/'input_checkpoint_manifest.json');out['input_checkpoint_hashes']={f:sha(base/('frontier_research_v23' if f.startswith('v23/') else 'frontier_research_v19')/'checkpoints'/f.split('/')[-1])==h for f,h in inputs.items()}
rows=read(P/'rows.json')['rows'];caps=[r['row']['text'].encode()[:4097] for r in rows];items=[];raws=[]
for r,d in zip(rows,caps):
 for s in range(0,len(d)-2048,2048):items.append({'doc':r['row_idx'],'start':s});raws.append(d[s:s+2049])
targets=[d[-128:] for d in raws];out['items_match']=items==read(P/'items.json');out['row_indices_match']= [r['row_idx'] for r in rows]==list(range(800,900))
out['cohort']={'sampled_documents':len(rows),'eligible_documents':len({i['doc'] for i in items}),'windows':len(items),'capped_bytes':sum(map(len,caps)),'targets_per_condition_seed':len(items)*128}
out['contamination']={'capped_document_duplicates':len(caps)-len(set(caps)),'window_duplicates':len(raws)-len(set(raws)),'target_span_duplicates':len(targets)-len(set(targets)),'complete128byte_target_spans_in_training':[j for j,d in enumerate(targets) if d in train],'limits':'Exact spans only; no exhaustive shorter-passage or near-duplicate decontamination.'}
r=read(P/'results.json');out['complete']=r['complete']
if r['complete']:
 z=np.load(P/'losses.npz');summary=read(P/'summary.json');out['loss_arrays']={k:{'shape':list(z[k].shape),'finite':bool(np.isfinite(z[k]).all())} for k in z.files};out['row_max_error']=max(abs(row['bpb']-z[f"{row['label']}_seed{row['seed']}_ctx{row['context']}"].mean()) for row in r['rows'])
 out['training_records_valid']=len(r['training'])==3 and all(t['steps']==250 and t['targets']==512000 for t in r['training'])
 out['new_checkpoint_hashes']={f:sha(P/'checkpoints'/f)==h for f,h in read(P/'checkpoint_manifest.json').items()}
 ids=np.array([i['doc'] for i in items]);docs=np.unique(ids);draws=np.random.default_rng(24).integers(0,len(docs),(5000,len(docs)));errors=[]
 for c in summary['comparisons']:
  a=np.stack([z[f"{c['a']}_seed{s}_ctx{c['a_context']}"] for s in (1,2,3)]);b=np.stack([z[f"{c['b']}_seed{s}_ctx{c['b_context']}"] for s in (1,2,3)]);v=(a-b).mean(axis=(0,2));sums=np.array([v[ids==d].sum() for d in docs]);counts=np.array([(ids==d).sum() for d in docs]);boots=sums[draws].sum(1)/counts[draws].sum(1)
  errors.extend([abs(v.mean()-c['a_minus_b_bpb']),*abs(np.quantile(boots,[.025,.975])-c['document_cluster_bootstrap95'])])
 out['comparison_max_error']=float(max(errors));out['main_cpu_seconds']=r['cpu_seconds'];out['training_cpu_seconds']=sum(t['cpu_seconds'] for t in r['training']);out['scoring_cpu_seconds']=sum(t['cpu_seconds'] for t in r['scoring'])
if (P/'timing.json').exists():
 t=read(P/'timing.json');groups={}
 for row in t['rows']:
  key=f"{row['arm']}_ctx{row['context']}";g=groups.setdefault(key,{'rows':0,'prefill_bytes':0,'decode_bytes':0,'prefill_wall_seconds':0,'decode_wall_seconds':0,'prefill_cpu_seconds':0,'decode_cpu_seconds':0});g['rows']+=1;g['prefill_bytes']+=row['prefill_bytes'];g['decode_bytes']+=row['decode_observed_bytes']
  for k in ['prefill_wall_seconds','decode_wall_seconds','prefill_cpu_seconds','decode_cpu_seconds']:g[k]+=row[k]
 for g in groups.values():g['prefill_bytes_per_wall_second']=g['prefill_bytes']/g['prefill_wall_seconds'];g['decode_bytes_per_wall_second']=g['decode_bytes']/g['decode_wall_seconds'];g['input_plus_observed_updates_wall_seconds']=g['prefill_wall_seconds']+g['decode_wall_seconds']
 out['timing_groups']=groups;out['timing_cpu_seconds']=t['cpu_seconds'];out['timing_rows']=len(t['rows']);out['timing_memory_values']={str(c):{k:sorted({r[k] for r in t['rows'] if r['context']==c}) for k in ['parameters','fp32_weight_bytes','prefill_state_bytes','final_state_bytes']} for c in (512,2048)}
out['audit_cpu_after_imports_seconds']=time.process_time()-start
(P/'independent_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
