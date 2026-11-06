from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent
r=json.loads((P/'results.json').read_text());assert r['complete'];rows=r['rows'];m=json.loads((P/'data_manifest.json').read_text());n=m['selected_documents'];rng=np.random.default_rng(8008);draw=rng.integers(0,n,(5000,n));summary=[];arrays={}
for kind in ['attention','signed']:
 for arm in ['short','long','mixed']:
  for context in [128,256,512]:
   a=np.array([next(x for x in rows if x['kind']==kind and x['arm']==arm and x['context']==context and x['seed']==seed)['document_bpb'] for seed in [1,2,3]]);assert a.shape==(3,n);arrays[kind,arm,context]=a
   summary.append({'kind':kind,'arm':arm,'context':context,'mean_bpb':float(a.mean()),'seed_bpb':a.mean(1).tolist()})
contrasts=[]
def contrast(label,a,b,primary=False):
 d=a-b;doc=d.mean(0);boot=doc[draw].mean(1);contrasts.append({'label':label,'primary':primary,'delta_bpb':float(doc.mean()),'ci95_document_bootstrap':np.quantile(boot,[.025,.975]).tolist(),'seed_deltas':d.mean(1).tolist(),'documents_favor_first':int((doc<0).sum()),'documents':n,'document_deltas':doc.tolist()})
for kind in ['attention','signed']:
 for context in [128,256,512]:
  for ref in ['long','short']:contrast(f'{kind}:mixed-minus-{ref}@{context}',arrays[kind,'mixed',context],arrays[kind,ref,context],ref=='long' and context in [128,512])
for arm in ['short','long','mixed']:
 for context in [128,256,512]:contrast(f'signed-minus-attention:{arm}@{context}',arrays['signed',arm,context],arrays['attention',arm,context])
(P/'summary.json').write_text(json.dumps({'means':summary,'contrasts':contrasts,'bootstrap_draws':5000,'bootstrap_seed':8008,'sampling_unit':'document; average seeds before resampling; all documents128 targets','multiplicity_adjustment':False},indent=2))
for x in contrasts:
 if x['primary']:print(x['label'],round(x['delta_bpb'],6),[round(t,6) for t in x['ci95_document_bootstrap']],x['documents_favor_first'])
