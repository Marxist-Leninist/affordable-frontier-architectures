"""No model selection: summarize frozen quality and timing measurements."""
import json,statistics,hashlib
from pathlib import Path
P=Path(__file__).parent

def main():
 r=json.loads((P/'results.json').read_text());t=json.loads((P/'timings.json').read_text());s={'quality':{},'timing':{}}
 for kind in ['transformer','gru']:
  rows=[x for x in r['neural'] if x['kind']==kind];fields=['reset128','full512','primary128']+(['local512'] if kind=='transformer' else [])
  s['quality'][kind]={f:dict(mean_bpb=statistics.mean(x[f]['bpb'] for x in rows),sample_seed_sd_bpb=statistics.stdev(x[f]['bpb'] for x in rows)) for f in fields}
 s['quality']['ngram']=r['ngram']['full512'];ts=[x for x in r['neural'] if x['kind']=='transformer'];s['local_reset_gate']=dict(all_seeds_within_two_sided_002=all(x['local_vs_reset']['abs_aggregate_gap_le_002'] for x in ts),seed_gaps=[x['local_vs_reset']['b_minus_a_bpb'] for x in ts],primary128_max_window_delta=max(x['primary128_max_window_bpb_delta'] for x in ts),interpretation='Two-sided equality screen fails for2 seeds because candidate is better by slightly more than.02;not a harmful regression or a statistical noninferiority test.')
 for model in ['transformer','local_transformer','gru','ngram']:
  rows=[x for x in t['rows'] if x['model']==model];s['timing'][model]={}
  for length in [128,512]:
   a=[next(z for z in x['timings'] if z['prefix_bytes']==length) for x in rows];s['timing'][model][str(length)]={key:statistics.median(z[key] for z in a) if a[0][key] is not None else None for key in ['prefill_p50_ms','sequence_mean_decode_p50_ms','end_to_end_p50_ms','end_to_end_p95_ms','final_state_tensor_bytes','allocated_state_storage_bytes']}
 s['cache_reduction_544']=1-s['timing']['local_transformer']['512']['allocated_state_storage_bytes']/s['timing']['transformer']['512']['allocated_state_storage_bytes'];s['economic_gate']='Unmeasured:CPU byte probes do not establish commercial-token cost or frontier quality.'
 (P/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
 w=json.loads((P/'window_scores.json').read_text());d={}
 for model,conditions in w.items():
  d[model]={}
  for condition,rows in conditions.items():
   bydoc={}
   for row in rows:bydoc.setdefault(row['doc'],[]).append(row['bpb'])
   d[model][condition]=[dict(row_index=i,windows=len(v),mean_bpb=statistics.mean(v)) for i,v in sorted(bydoc.items())]
 (P/'document_scores.json').write_text(json.dumps(d,separators=(',',':'))+'\n')
 (P/'intermediate_manifest.json').write_text(json.dumps({f:dict(sha256=hashlib.sha256((P/f).read_bytes()).hexdigest(),bytes=(P/f).stat().st_size,redistributed=False) for f in ['rows.json','window_scores.json']},indent=2)+'\n');print(json.dumps(s,indent=2))
if __name__=='__main__':main()
