from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;r=json.loads((P/'results.json').read_text());assert r['complete'];rows=r['rows'];n=len(rows[0]['document_bpb']);draw=np.random.default_rng(9009).integers(0,n,(5000,n));means=[];contrasts=[]
def comparison(label,a,b,primary=False):
 d=a-b;doc=d.mean(0) if d.ndim==2 else d;boot=doc[draw].mean(1);contrasts.append(dict(label=label,primary=primary,mean_delta_bpb=float(doc.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),seed_deltas=d.mean(1).tolist() if d.ndim==2 else None,documents_favor_original=int((doc<0).sum()),documents=n,document_deltas=doc.tolist()))
for kind in ['attention','signed']:
 for arm in ['long','mixed']:
  arrays={}
  for condition in ['original512','donor512','suffix256']:
   a=np.array([next(x for x in rows if x['kind']==kind and x['arm']==arm and x['condition']==condition and x['seed']==seed)['document_bpb'] for seed in [1,2,3]]);arrays[condition]=a;means.append(dict(kind=kind,arm=arm,condition=condition,bpb=float(a.mean()),seed_bpb=a.mean(1).tolist()))
  comparison(kind+':'+arm+':original-minus-donor',arrays['original512'],arrays['donor512'],True);comparison(kind+':'+arm+':original-minus-suffix',arrays['original512'],arrays['suffix256'])
c=json.loads((P/'copy_results.json').read_text())['rows'];arrays={x['condition']:np.array(x['document_bpb']) for x in c}
comparison('copy:original-minus-donor',arrays['original512'],arrays['donor512']);comparison('copy:original-minus-suffix',arrays['original512'],arrays['suffix256'])
kl=[]
for kind in ['attention','signed']:
 for arm in ['long','mixed']:
  a=[x for x in r['kl'] if x['kind']==kind and x['arm']==arm];kl.append(dict(kind=kind,arm=arm,mean_kl_bits=float(np.mean([x['mean_kl_bits'] for x in a])),largest_token_kl_bits=max(x['max_token_kl_bits'] for x in a)))
(P/'summary.json').write_text(json.dumps(dict(means=means,contrasts=contrasts,kl=kl,bootstrap_draws=5000,bootstrap_seed=9009,unit='disjoint recipient/donor document pair; average model seeds first',multiplicity_adjustment=False),indent=2))
for x in contrasts:print(x['label'],round(x['mean_delta_bpb'],6),[round(z,6) for z in x['ci95']])
