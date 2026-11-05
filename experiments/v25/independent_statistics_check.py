"""Recompute paired recipient-document bootstrap from saved per-target losses."""
import json
from pathlib import Path
import numpy as np
p=Path(__file__).parent;j=json.loads((p/'summary.json').read_text());items=json.loads((p/'items.json').read_text());z=np.load(p/'losses.npz');docs=sorted({i['doc'] for i in items});indices=[[k for k,i in enumerate(items) if i['doc']==d] for d in docs];draw=np.random.default_rng(25).integers(0,len(docs),(5000,len(docs)));counts=np.array([len(v) for v in indices]);results=[]
for r in j['comparisons']:
 a,b=r['comparison'].split(' minus ');delta=np.stack([z[f"{r['arm']}_seed{s}_{a}"]-z[f"{r['arm']}_seed{s}_{b}"] for s in (1,2,3)]).mean((0,2));sums=np.array([delta[idx].sum() for idx in indices]);boot=sums[draw].sum(1)/counts[draw].sum(1);ci=np.quantile(boot,[.025,.975]);error=float(np.max(np.abs(ci-r['document_cluster_bootstrap95'])));assert error<1e-12;results.append(dict(arm=r['arm'],comparison=r['comparison'],max_ci_error=error))
(p/'independent_statistics_checks.json').write_text(json.dumps(results,indent=2)+'\n');print('All10 paired document-bootstrap intervals reproduce within1e-12.')
