"""Independent ratio-of-sums paired recipient cluster bootstrap verification."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).parent;j=json.loads((P/'summary.json').read_text());items=json.loads((P/'items.json').read_text());z=np.load(P/'losses.npz');mask=np.load(P/'subgroup.npy');docs=sorted({i['doc'] for i in items});idx=[np.array([k for k,i in enumerate(items) if i['doc']==d]) for d in docs];draw=np.random.default_rng(26).integers(0,len(docs),(5000,len(docs)));out=[]
def arr(a,c):return z['copy_'+c] if a=='copy' else np.stack([z[f'{a}_seed{s}_{c}'] for s in (1,2,3)]).mean(0)
for cohort,m in [('all',np.ones_like(mask)),('distant_repeat',mask)]:
 n=np.array([m[i].sum() for i in idx]);den=n[draw].sum(1);valid=den>0
 for a,cc in j[cohort]['means'].items():
  for c,v in cc.items():assert abs(float(arr(a,c)[m].mean())-v)<1e-12
 for r in j[cohort]['effects']:
  a,b=r['comparison'].split(' minus ');delta=arr(r['arm'],a)-arr(r['arm'],b);sums=np.array([delta[i][m[i]].sum() for i in idx]);boot=sums[draw].sum(1)[valid]/den[valid];ci=np.quantile(boot,[.025,.975]);err=float(np.max(np.abs(ci-r['bootstrap95'])));assert err<1e-12;assert abs(delta[m].mean()-r['effect_bpb'])<1e-12;assert int((~valid).sum())==r['zero_denominator_bootstraps'];out.append(dict(cohort=cohort,arm=r['arm'],comparison=r['comparison'],max_ci_error=err))
(P/'independent_statistics_checks.json').write_text(json.dumps(out,indent=2)+'\n');print('Verified all means and',len(out),'paired bootstrap intervals.')
counts=[dict(doc=d,targets=int(mask[i].sum()),windows=int(mask[i].any(1).sum())) for d,i in zip(docs,idx)];counts.sort(key=lambda x:(-x['targets'],x['doc']));positive=[x['targets'] for x in counts if x['targets']];concentration=dict(document_counts=counts,max_document_target_share=max(positive)/int(mask.sum()),top5_document_target_share=sum(positive[:5])/int(mask.sum()),nonempty_document_target_count_median=float(np.median(positive)))
(P/'independent_subgroup_concentration.json').write_text(json.dumps(concentration,indent=2)+'\n')
