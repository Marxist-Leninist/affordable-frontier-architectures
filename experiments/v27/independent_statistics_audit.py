"""Independent paired-bootstrap and timing ledger reconstruction."""
import json
import numpy as np
from run import P,data,verify,load
from local_attention import allocated_tensor_bytes
from pruning import local_prefill
from prefill import prefill
import torch
verify();items=data();z=np.load(P/'losses.npz');summary=json.loads((P/'summary.json').read_text());ids=np.array([i['doc'] for i in items]);docs=np.unique(ids);samples=np.random.default_rng(27).integers(len(docs),size=(5000,len(docs)));contrasts=0
for name,row in summary['contrasts'].items():
 arm,scope,_,ref=name.split('_');delta=np.stack([z[f'{arm}_seed{s}_{scope}']-z[f'{arm}_seed{s}_{ref}'] for s in (1,2,3)]).mean(0)
 sums=np.array([delta[ids==d].sum() for d in docs]);counts=np.array([(ids==d).sum()*128 for d in docs]);bootstrap=sums[samples].sum(1)/counts[samples].sum(1);ci=np.percentile(bootstrap,[2.5,97.5]);np.testing.assert_allclose(ci,row['ci95'],rtol=0,atol=1e-12);assert abs(delta.mean()-row['mean'])<1e-12
 np.testing.assert_allclose(sums,row['document_sums'],rtol=0,atol=1e-10);np.testing.assert_array_equal(counts,row['document_target_counts'])
 if ref=='full':assert row['passes_fixed_gate']==bool(ci[1]<=.01)
 contrasts+=1
out={'contrasts_verified':contrasts,'all_bounded_scopes_fail_gate':all(not r['passes_fixed_gate'] for n,r in summary['contrasts'].items() if n.endswith('_minus_full'))}
if (P/'timing.json').exists():
 rows=json.loads((P/'timing.json').read_text())['rows'];keys={(r['seed'],r['arm'],str(r['scope']),r['doc'],r['rep']) for r in rows};assert len(keys)==len(rows)==480
 docs=[r['row_idx'] for r in json.loads((P/'rows.json').read_text())['rows'] if len(r['row']['text'].encode())>=4128][:8];assert keys=={(s,a,str(v),d,r) for s in (1,2,3) for a in ('local','gru') for v in ('neural',512,1024,2048,'full') for d in docs for r in (0,1)}
 for a in ('local','gru'):
  for s in (1,2,3):
   m=load(a,s);weights=sum(x.numel()*x.element_size() for x in m.state_dict().values());x=torch.tensor(list(items[0]['raw'][:4096]))[None,:]
   with torch.inference_mode():_,state=local_prefill(m,x,1,prune=True) if a=='local' else prefill(m,x)
   nbytes=allocated_tensor_bytes(state)
   for r in rows:
    if (r['arm'],r['seed'])!=(a,s):continue
    assert r['weight_bytes']==weights and r['prefill_state_bytes']==r['final_state_bytes']==nbytes
    assert all(r[k]>0 for k in ('prefill_wall_seconds','decode_wall_seconds'))
    assert all(r[k]>=0 for k in ('prefill_cpu_seconds','decode_cpu_seconds'))
    assert (r['final_copy_index_bytes']==0)==(r['scope']=='neural')
 out['timing_rows_verified']=len(rows)
 out['zero_cpu_prefill_rows']=sum(r['prefill_cpu_seconds']==0 for r in rows)
(P/'independent_statistics_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
