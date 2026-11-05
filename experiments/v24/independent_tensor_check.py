import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json, hashlib, time, resource
from pathlib import Path
import torch
P=Path(__file__).parent;torch.set_num_threads(1);start=time.process_time();out={}
for seed in (1,2,3):
 for arm in ('context2048','reset512','mixed'):
  f=(P if arm=='mixed' else P.parent/'frontier_research_v23')/'checkpoints'/f'{arm}_seed{seed}_step250.pt';d=torch.load(f,weights_only=True)
  out[f'{arm}_{seed}']={'dtypes':sorted({str(t.dtype) for t in d.values()}),'tensor_elements':sum(t.numel() for t in d.values()),'tensor_bytes':sum(t.numel()*t.element_size() for t in d.values()),'finite':all(bool(torch.isfinite(t).all()) for t in d.values())}
a=json.loads((P/'timing_amendment.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={'weights':out,'timing_amendment_hash_checks':{k:sha(P/f)==a[k] for k,f in [('original_protocol_sha256','protocol.json'),('original_timing_sha256','timing.py'),('executed_timing_sha256','timing_distribution.py')]},'expected_KV_bytes':{str(c):{'prefill':2*2*c*64*4,'after32':2*2*(c+32)*64*4} for c in (512,2048)},'cpu_after_imports_seconds':time.process_time()-start}
s=json.loads((P/'timing_summary.json').read_text());t=json.loads((P/'timing.json').read_text());errors=[]
for g in s:
 rows=[r for r in t['rows'] if r['arm']==g['arm'] and r['context']==g['context']]
 for phase,prefix in [('prefill','prefill'),('decode','observed_decode')]:
  for typ in ('wall','cpu'):
   rate=sum(r['prefill_bytes' if phase=='prefill' else 'decode_observed_bytes'] for r in rows)/sum(r[f'{phase}_{typ}_seconds'] for r in rows)
   errors.append(abs(rate-g[f'{prefix}_bytes_per_{typ}_second']))
checks['timing_summary_max_error']=max(errors)
(P/'independent_tensor_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
