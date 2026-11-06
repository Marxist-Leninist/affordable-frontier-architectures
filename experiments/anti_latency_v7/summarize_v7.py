from pathlib import Path
import json,statistics,hashlib
P=Path(__file__).resolve().parent
r=json.loads((P/'results.json').read_text());assert r['complete'] and len(r['rows'])==18
rows=r['rows'];summary={}
for kind in ['attention','signed']:
 summary[kind]={}
 for metric in rows[0]['after']:
  paired=[]
  for seed in [1,2,3]:
   a={x['mode']:x['after'][metric]['bpb'] for x in rows if x['kind']==kind and x['seed']==seed};paired.append({'seed':seed,**a,'mixed_minus_long':a['mixed']-a['long'],'mixed_minus_short':a['mixed']-a['short']})
  summary[kind][metric]={'pairs':paired,**{m:statistics.mean(x[m] for x in paired) for m in ['short','long','mixed','mixed_minus_long','mixed_minus_short']}}
(P/'summary.json').write_text(json.dumps(summary,indent=2))
invalid=json.loads((P/'invalid_pilot_receipt.json').read_text())
compute={'corrected_reconstruction_targets':6*819200,'corrected_continuation_targets':18*409600,'corrected_total_quality_training_targets':6*819200+18*409600,'corrected_elapsed_wall_seconds':r['elapsed_wall_seconds'],'reconstruction_seconds':sum(x['reconstruction_seconds'] for x in rows[::3]),'continuation_step_seconds':sum(sum(x['step_seconds']) for x in rows),'disposable_profile_training_targets':36*6*2048,'invalid_pilot_confirmed_targets':invalid['confirmed_target_presentations'],'invalid_pilot_partial_next_reconstruction':'not instrumented before interrupt; not included in confirmed lower bound','paid_resources_created':False,'invoice_cost':'not observed; no zero-cost claim'}
(P/'compute.json').write_text(json.dumps(compute,indent=2))
print(json.dumps({k:{m:{a:round(r[a],6) for a in ['short','long','mixed']} for m,r in v.items()} for k,v in summary.items()},indent=2))
