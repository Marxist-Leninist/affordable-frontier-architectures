import json, statistics, hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
new=json.loads((P/'restart_results.json').read_text()); assert new['complete']
old=json.loads((P/'historical_seed1.json').read_text())
rows=old['rows']+new['rows'];metrics=list(rows[0]['after']);summary={}
for kind in ['attention','signed']:
 summary[kind]={}
 for metric in metrics:
  pairs=[]
  for seed in [1,2,3]:
   r={x['mode']:x['after'][metric]['bpb'] for x in rows if x['kind']==kind and x['seed']==seed}
   pairs.append({'seed':seed,**r,'retention_improvement':r['reset']-r['retain']})
  summary[kind][metric]={'pairs':pairs,'mean_reset':statistics.mean(x['reset'] for x in pairs),'mean_retain':statistics.mean(x['retain'] for x in pairs),'mean_improvement':statistics.mean(x['retention_improvement'] for x in pairs)}
assert all(x['historic_weight_maxerror']==0 for x in new['rows'])
(P/'summary_v6.json').write_text(json.dumps(summary,indent=2))
(P/'compute_v6.json').write_text(json.dumps({'new_training_target_bytes':4*819200+8*409600,'disposable_training_probe_target_bytes':235520,'total_target_presentations_excluding_inference_evaluation':6789120,'reconstruction_seconds':sum(x['reconstruction_seconds'] for x in new['rows'][::2]),'continuation_step_seconds':sum(x['seconds'] for x in new['rows']),'paid_resources_created':False,'incremental_charges_authorized':0,'note':'Historical seed1 evidence reused, not rerun; timings exclude evaluation and other overhead. No monetary hardware rate observed.'},indent=2))
print(json.dumps({k:{m:round(v['mean_improvement'],6) for m,v in ms.items()} for k,ms in summary.items()},indent=2))
