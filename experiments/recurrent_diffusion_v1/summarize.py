"""Summarize every predeclared endpoint without selecting favorable seeds."""
import json,hashlib,statistics
from pathlib import Path
P=Path(__file__).parent;seeds=[1,2,3];kinds=['bptt','diffusion_weighted','diffusion_ce','transformer'];rows=[]
for seed in seeds:
 for kind in kinds:
  rows.append(json.loads((P/f'{kind}_{seed}_equal_data.json').read_text()))
  if kind!='bptt':
   paths=list(P.glob(f'{kind}_{seed}_*cpu.json'));assert len(paths)==1;rows.append(json.loads(paths[0].read_text()))
summary=[]
for kind in kinds:
 rr=[r for r in rows if r['kind']==kind and r['endpoint']=='equal_data'];scores=[r['test']['bpb'] for r in rr]
 summary.append(dict(kind=kind,mean_test_bpb=statistics.mean(scores),sample_sd=statistics.stdev(scores),seed_test_bpb=scores,parameters=rr[0]['parameters'],mean_training_cpu_seconds=statistics.mean(r['training_cpu_seconds'] for r in rr),train_exposures_per_seed=614400))
extra=[]
for r in rows:
 if r['endpoint']!='equal_data':
  b=next(z for z in rows if z['kind']=='bptt' and z['seed']==r['seed']);extra.append(dict(kind=r['kind'],seed=r['seed'],endpoint=r['endpoint'],steps=r['steps'],test_bpb=r['test']['bpb'],training_cpu_seconds=r['training_cpu_seconds'],baseline_cpu_budget=b['training_cpu_seconds'],cpu_budget_fraction=r['training_cpu_seconds']/b['training_cpu_seconds']))
out=dict(equal_data=summary,extra_training=extra,limits='Three tiny seeds, one byte-level corpus and an explicitly finite-mixture decoder. Results do not measure exact diffusion likelihood or reproduce original Huginn.',all_endpoints=rows)
(P/'results.json').write_text(json.dumps(out,indent=2));(P/'summary.json').write_text(json.dumps({k:v for k,v in out.items() if k!='all_endpoints'},indent=2))
manifest={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted((P/'checkpoints').glob('*.pt'))}
(P/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(out['equal_data'],indent=2))
