"""Summarize fixed quality tests and all prespecified timing lengths; no winner selection."""
import json,statistics,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).parent

def main():
 r=json.loads((P/'results.json').read_text());t=json.loads((P/'timings.json').read_text());s={'quality':{},'timing':{},'local_paired_timing':{}}
 for length in [512,2048]:
  a=[x for x in r['rows'] if x['input_bytes']==length];s['quality'][str(length)]={k:dict(mean_bpb=statistics.mean(x['mean_bpb'][k] for x in a),sample_seed_sd=statistics.stdev(x['mean_bpb'][k] for x in a)) for k in a[0]['mean_bpb']}
 s['quality_gate']=dict(all_pass=r['all_quality_gates_pass'],max_window_bpb_error=max(x['max_abs_window_bpb_error'] for x in r['rows']),max_logits_error=max(x['max_abs_local_logits_error'] for x in r['rows']),max_state_error=max(x['max_abs_local_state_error'] for x in r['rows']))
 for model in ['full_transformer','local_monolithic','local_chunked','gru','ngram']:
  s['timing'][model]={}
  for length in [128,512,2048]:
   rows=[x for x in t['samples'] if x['model']==model and x['prefix_bytes']==length];seeds=sorted(set(x['seed'] for x in rows));byseed=[]
   for seed in seeds:
    a=[x for x in rows if x['seed']==seed];byseed.append(dict(seed=seed,**{k:float(np.median([x[k] for x in a])) for k in ['prefill_ms','decode_mean_ms','end_ms']},end_p95_ms=float(np.quantile([x['end_ms'] for x in a],.95))))
   s['timing'][model][str(length)]=dict(median_seed_p50={k:statistics.median(x[k] for x in byseed) for k in ['prefill_ms','decode_mean_ms','end_ms']},median_seed_p95_ms=statistics.median(x['end_p95_ms'] for x in byseed),by_seed=byseed,allocated_state_bytes=rows[0]['allocated_state_bytes'],probes=len(rows))
 for length in [128,512,2048]:
  a={ (x['seed'],x['row_index'],x['repetition']):x for x in t['samples'] if x['model']=='local_monolithic' and x['prefix_bytes']==length};b={ (x['seed'],x['row_index'],x['repetition']):x for x in t['samples'] if x['model']=='local_chunked' and x['prefix_bytes']==length};assert a.keys()==b.keys()
  s['local_paired_timing'][str(length)]={key:dict(median_monolithic_over_chunked=float(np.median([a[k][key]/b[k][key] for k in a])),fraction_chunked_faster=statistics.mean(b[k][key]<a[k][key] for k in a)) for key in ['prefill_ms','end_ms']}
 s['mask_shape_counts']=dict(monolithic2048=2048*2048,chunked_max=128*(128+127),scope='Largest explicit boolean attention-mask element count, not full peak memory or actual score-tensor allocation.')
 s['economic_gate']='Unmeasured. Matched byte-model implementation quality is not frontier quality or commercial-token throughput.'
 (P/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
 ws=json.loads((P/'window_scores.json').read_text());d={}
 for group,conditions in ws.items():
  d[group]={}
  for condition,rows in conditions.items():
   bydoc={}
   for row in rows:bydoc.setdefault(row['row_index'],[]).append(row['bpb'])
   d[group][condition]=[dict(row_index=i,windows=len(v),mean_bpb=statistics.mean(v)) for i,v in sorted(bydoc.items())]
 (P/'document_scores.json').write_text(json.dumps(d,separators=(',',':'))+'\n')
 (P/'intermediate_manifest.json').write_text(json.dumps({f:dict(sha256=hashlib.sha256((P/f).read_bytes()).hexdigest(),bytes=(P/f).stat().st_size,redistributed=False) for f in ['rows.json','window_scores.json']},indent=2)+'\n');print(json.dumps(s,indent=2))
if __name__=='__main__':main()
