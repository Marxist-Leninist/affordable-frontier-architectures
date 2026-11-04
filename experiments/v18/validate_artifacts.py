"""Standard-library consistency audit of retained numerical artifacts."""
import json,statistics,math,hashlib
from pathlib import Path
P=Path(__file__).parent
def read(f):return json.loads((P/f).read_text())
def percentile(xs,q):
 x=sorted(xs);p=(len(x)-1)*q;i=int(p);return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(p-i)
r=read('results.json');d=read('scores.json');checks=[]
for a in r['rows']:
 rows=[x for x in d if x['seed']==a['seed'] and x['model']==a['model']];assert len(rows)==98
 assert math.isclose(statistics.mean(x['full_bpb'] for x in rows),a['full_bpb'],abs_tol=1e-12)
 assert math.isclose(statistics.mean(x['pruned_bpb'] for x in rows),a['pruned_bpb'],abs_tol=1e-12)
 assert math.isclose(max(abs(x['delta']) for x in rows),a['max_abs_window_bpb_error'],abs_tol=1e-12)
 if a['model']=='local':assert a['strict_parity']
t=read('timings.json');assert len(t['samples'])==576;summary=[]
for a in t['summary']:
 rows=[x for x in t['samples'] if x['length']==a['length']];pairs={}
 for x in rows:
  assert all(math.isfinite(x[k]) and x[k]>0 for k in ['prefill_ms','end_ms','decode_ms'])
  pair=pairs.setdefault((x['seed'],x['row_index'],x['rep']),{});assert x['pruned'] not in pair;pair[x['pruned']]=x
 assert len(pairs)==96 and all(set(x)=={False,True} for x in pairs.values())
 ratios=[x[False]['end_ms']/x[True]['end_ms'] for x in pairs.values()];assert statistics.median(ratios)==a['median_paired_end_speedup'];assert sum(x>1 for x in ratios)==a['faster']
 assert all(x[False]['state_bytes']==x[True]['state_bytes']==130064 for x in pairs.values())
 summary.append(dict(**a,full_end_p95=percentile([x[False]['end_ms'] for x in pairs.values()],.95),pruned_end_p95=percentile([x[True]['end_ms'] for x in pairs.values()],.95)))
result=dict(score_groups=9,score_rows=len(d),timing_samples=576,paired_timings=288,state_bytes=130064,summary=summary,validation='passed; arithmetic consistency is not independent inference reproduction')
(P/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
