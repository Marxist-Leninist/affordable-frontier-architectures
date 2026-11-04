"""Deterministic summary from completed artifacts, no model execution."""
import json,statistics,hashlib
from pathlib import Path
P=Path(__file__).parent
r=json.loads((P/'results.json').read_text());n=json.loads((P/'ngram_results.json').read_text());rows=r['final'];summary={}
for k in ['transformer','gru']:
 a=[x for x in rows if x['kind']==k]
 summary[k]={'parameters':a[0]['parameters'],'seed_test_bpb':[x['test']['bpb'] for x in a],'mean_test_bpb':statistics.mean(x['test']['bpb'] for x in a),'sd_test_bpb':statistics.stdev(x['test']['bpb'] for x in a),'mean_training_seconds':statistics.mean(x['training_seconds'] for x in a),'all_pilot_final_training_seconds':sum(x['training_seconds'] for x in rows+r['tuning'] if x['kind']==k),'total_training_target_presentations_including_pilots':6*1024000,'latency_median_of_seed_medians':[]}
 for length in [128,512,2048]:
  l=[next(t for t in x['latency'] if t['prefix_bytes']==length) for x in a]
  summary[k]['latency_median_of_seed_medians'].append({'prefix_bytes':length,'prefill_ms':statistics.median(t['prefill_p50_ms'] for t in l),'cached_decode_ms':statistics.median(t['cached_decode_p50_ms'] for t in l),'cache_tensor_bytes':l[0]['initial_cache_tensor_bytes']})
summary['ngram']={'test_bpb':n['test']['bpb'],'context_bytes':n['selected_order'],'all_tables_serialized_bytes':n['serialized_all_order_tables_bytes'],'fit_seconds':n['training_seconds_shared_order4_table'],'distribution_p50_ms':n['next_byte_distribution_p50_ms']}
summary['gru_bpb_reduction_vs_transformer_fraction']=1-summary['gru']['mean_test_bpb']/summary['transformer']['mean_test_bpb']
summary['ngram_bpb_reduction_vs_gru_fraction']=1-n['test']['bpb']/summary['gru']['mean_test_bpb']
summary['interpretation']='Cross-entropy reductions, not accuracy or intelligence percentages; same sampled target exposure, not equal model memory/FLOPs or full quality parity.'
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
