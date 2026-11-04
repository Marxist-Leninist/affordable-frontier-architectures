"""Descriptive summary of locked transfer experiment, no model selection."""
import json,statistics,hashlib
from pathlib import Path
P=Path(__file__).parent

def main():
 r=json.loads((P/'results.json').read_text());tim=json.loads((P/'retiming.json').read_text());s={'families':{},'pairwise_quality_screen':{},'economic_gate':'Unmeasured; byte CPU probes cannot qualify commercial-token cost thresholds.'}
 for kind in ['transformer','gru']:
  rows=[x for x in r['neural'] if x['kind']==kind];vals=[x['primary']['bpb'] for x in rows]
  s['families'][kind]=dict(mean_primary_bpb=statistics.mean(vals),sample_seed_sd_bpb=statistics.stdev(vals),mean_paired_short_bpb=statistics.mean(x['context_short']['bpb'] for x in rows),mean_paired_long_bpb=statistics.mean(x['context_long']['bpb'] for x in rows),mean_long_minus_short_bpb=statistics.mean(x['paired']['long_minus_short_bpb'] for x in rows))
 s['families']['ngram']=dict(primary_bpb=r['ngram']['primary']['bpb'],short_bpb=r['ngram']['context_short']['bpb'],long_bpb=r['ngram']['context_long']['bpb'])
 for a,b in [('transformer','gru'),('transformer','ngram'),('gru','ngram')]:
  gaps=[]
  for seed in [1,2,3]:
   av=next(x['primary']['bpb'] for x in r['neural'] if x['kind']==a and x['seed']==seed);bv=r['ngram']['primary']['bpb'] if b=='ngram' else next(x['primary']['bpb'] for x in r['neural'] if x['kind']==b and x['seed']==seed);gaps.append(bv-av)
  s['pairwise_quality_screen'][a+'_vs_'+b]=dict(b_minus_a_bpb=gaps,all_within_002_bpb=all(abs(x)<=.02 for x in gaps))
 s['timing_summary']={}
 for kind in ['transformer','gru','ngram']:
  s['timing_summary'][kind]={}
  for length in [128,512]:
   ts=[next(t for t in row['timing'] if t['prefix_bytes']==length) for row in tim['rows'] if row['kind']==kind]
   s['timing_summary'][kind][str(length)]={key:statistics.median(t[key] for t in ts) for key in ['prefill_p50_ms','sequence_average_decode_p50_ms','end_to_end_p50_ms','end_to_end_p95_ms','final_state_bytes']}
 ws=json.loads((P/'window_scores.json').read_text());doc_scores={}
 for model,conditions in ws.items():
  doc_scores[model]={}
  for condition,rows in conditions.items():
   bydoc={}
   for row in rows:bydoc.setdefault(row['doc'],[]).append(row['bpb'])
   doc_scores[model][condition]=[dict(document=i,windows=len(v),mean_bpb=statistics.mean(v)) for i,v in sorted(bydoc.items())]
 (P/'document_scores.json').write_text(json.dumps(doc_scores,separators=(',',':'))+'\n')
 (P/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
 (P/'intermediate_manifest.json').write_text(json.dumps({f:dict(sha256=hashlib.sha256((P/f).read_bytes()).hexdigest(),bytes=(P/f).stat().st_size,redistributed=False) for f in ['rows.json','window_scores.json']},indent=2)+'\n')
 print(json.dumps(s,indent=2))
if __name__=='__main__':main()
