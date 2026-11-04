"""Postprocess locked v14 results; never selects models on test scores."""
import json,statistics,hashlib
from pathlib import Path
P=Path(__file__).parent

def main():
 r=json.loads((P/'results.json').read_text());s={'families':{},'quality_gates':[],'note':'The .02-bpb screen is descriptive, not a statistical equivalence test. No failed screen is called matched quality.'}
 for kind in ['transformer','gru']:
  rows=[row for row in r['final'] if row['kind']==kind]; cp=[next(c for c in row['evaluated_checkpoints'] if c['step']==4000) for row in rows]
  bpb=[c['test']['bpb'] for c in cp]
  s['families'][kind]=dict(mean_test_bpb=statistics.mean(bpb),seed_sample_sd_bpb=statistics.stdev(bpb),individual_test_bpb=bpb,total_training_seconds_including_pilots=sum(row['training_seconds'] for row in r['final']+r['tuning'] if row['kind']==kind),mean_final_training_seconds=statistics.mean(row['training_seconds'] for row in rows),median_seed_prefill_p50_ms=statistics.median(c['latency']['prefill_p50_ms'] for c in cp),median_seed_decode_step_p50_ms=statistics.median(c['latency']['decode_step_p50_ms'] for c in cp),median_seed_end_to_end_p50_ms=statistics.median(c['latency']['end_to_end_p50_ms'] for c in cp),checkpoint_bytes=cp[0]['checkpoint_bytes'],final_cache_tensor_bytes=cp[0]['latency']['final_cache_tensor_bytes'])
 for gate in [2.5,2.35,2.25]:
  pairs=[]
  for seed in [1,2,3]:
   chosen={}
   for kind in ['transformer','gru']:
    row=next(row for row in r['final'] if row['kind']==kind and row['seed']==seed);step=row['validation_gate_steps'][str(gate)]
    if step is not None:
     cp=next(c for c in row['evaluated_checkpoints'] if c['step']==step); curve=next(c for c in row['curve'] if c['step']==step)
     chosen[kind]=dict(step=step,validation_bpb=curve['validation']['bpb'],test_bpb=cp['test']['bpb'],training_seconds=curve['training_seconds'],end_to_end_p50_ms=cp['latency']['end_to_end_p50_ms'])
   pair=dict(seed=seed,selected=chosen,both_validation_gates_reached=len(chosen)==2)
   if len(chosen)==2:
    gap=chosen['gru']['test_bpb']-chosen['transformer']['test_bpb'];pair.update(test_gap_gru_minus_transformer_bpb=gap,within_descriptive_002_bpb_screen=abs(gap)<=.02)
   pairs.append(pair)
  s['quality_gates'].append(dict(validation_threshold=gate,pairs=pairs,all_seeds_reached_and_within_screen=all(p.get('within_descriptive_002_bpb_screen',False) for p in pairs)))
 s['ngram']=r['ngram'];s['wall_seconds']=r['total_wall_seconds']
 (P/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
 manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((P/'checkpoints').glob('*.pt'))}
 (P/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(s,indent=2))
if __name__=='__main__':main()
