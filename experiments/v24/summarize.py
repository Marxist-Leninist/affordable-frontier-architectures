"""Prespecified descriptive document-cluster summaries, all three seeds averaged."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).parent
r=json.loads((P/'results.json').read_text());assert r['complete'];z=np.load(P/'losses.npz');items=json.loads((P/'items.json').read_text());docs=sorted({i['doc'] for i in items});rng=np.random.default_rng(24);resamples=rng.integers(0,len(docs),(5000,len(docs)))
labels=['initial','context2048','reset512','mixed'];means={}
for label in labels:
 means[label]={str(c):float(np.mean([z[f'{label}_seed{s}_ctx{c}'] for s in (1,2,3)])) for c in (512,2048)}
def diff(a,ac,b,bc):
 v=np.mean([z[f'{a}_seed{s}_ctx{ac}'].mean(1)-z[f'{b}_seed{s}_ctx{bc}'].mean(1) for s in (1,2,3)],axis=0);idx=np.array([i['doc'] for i in items]);summ=np.array([v[idx==d].sum() for d in docs]);counts=np.array([(idx==d).sum() for d in docs]);boot=summ[resamples].sum(1)/counts[resamples].sum(1)
 return dict(a=a,a_context=ac,b=b,b_context=bc,a_minus_b_bpb=float(v.mean()),document_cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),documents_a_lower=int(sum(v[idx==d].mean()<0 for d in docs)),seed_effects=[float(z[f'{a}_seed{s}_ctx{ac}'].mean()-z[f'{b}_seed{s}_ctx{bc}'].mean()) for s in (1,2,3)])
comparisons=[diff('mixed',c,a,c) for a in ('initial','context2048','reset512') for c in (512,2048)]+[diff(a,2048,a,512) for a in labels]
out=dict(means=means,comparisons=comparisons,windows=len(items),documents=len(docs),targets_per_condition_seed=len(items)*128,training_cpu_seconds=sum(t['cpu_seconds'] for t in r['training']),main_cpu_seconds=r['cpu_seconds'],limits='Bootstrap conditional on deterministic convenience cohort; seeds share data; references have unequal added training; no convergence, equal-compute or frontier claim.')
(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
