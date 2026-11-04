"""Independent standard-library-only timing artifact audit; no model execution."""
import json, statistics, random, math, hashlib
from pathlib import Path
P=Path(__file__).parent
t=json.loads((P/'timings.json').read_text());v=json.loads((P/'validation.json').read_text())
rows=t['samples'];cohort=t['cohort'];assert len(rows)==576 and len(cohort)==8 and len(set(cohort))==8
identities=[(r['seed'],r['length'],r['row_index'],r['rep'],r['pruned']) for r in rows]
assert len(set(identities))==len(identities)
expected=[];rng=random.Random(18112026)
for seed in (1,2,3):
 for n in (128,512,2048):
  for i in cohort:
   for rep in range(6):
    order=[False,True];rng.shuffle(order)
    if rep>=2:expected.extend((seed,n,i,rep,p) for p in order)
assert identities==expected
for r in rows:
 assert all(math.isfinite(r[k]) and r[k]>0 for k in ('end_ms','prefill_ms','decode_ms'))
 assert abs(r['end_ms']-r['prefill_ms']-r['decode_ms'])<1e-9
 assert r['state_bytes']==130064

def p95(xs):
 xs=sorted(xs);k=(len(xs)-1)*.95;i=int(k)
 return xs[i]+(k-i)*(xs[min(i+1,len(xs)-1)]-xs[i])
summary=[]
for n in (128,512,2048):
 pairs={}
 for r in rows:
  if r['length']==n:pairs.setdefault((r['seed'],r['row_index'],r['rep']),{})[r['pruned']]=r
 assert len(pairs)==96 and all(set(p)=={False,True} for p in pairs.values())
 ratios=[p[False]['end_ms']/p[True]['end_ms'] for p in pairs.values()]
 out=dict(length=n,pairs=len(pairs),median_paired_end_speedup=statistics.median(ratios),faster=sum(x>1 for x in ratios),median_paired_prefill_speedup=statistics.median(p[False]['prefill_ms']/p[True]['prefill_ms'] for p in pairs.values()))
 for prune,name in ((False,'full'),(True,'pruned')):
  xs=[p[prune]['end_ms'] for p in pairs.values()]
  out[name+'_end_p50']=statistics.median(xs);out[name+'_end_p95']=p95(xs)
 for target in (next(x for x in t['summary'] if x['length']==n),next(x for x in v['summary'] if x['length']==n)):
  for k,value in target.items():assert abs(value-out[k])<1e-10,(k,value,out[k])
 summary.append(out)
result=dict(samples=len(rows),pairs=288,exact_cohort_and_randomized_order_verified=True,summary=summary,sha256={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ('benchmark.py','timings.json','validation.json')})
(P/'audit_timing.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
