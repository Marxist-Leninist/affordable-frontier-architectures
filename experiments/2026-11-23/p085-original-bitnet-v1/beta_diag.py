"""Post-test weight reconstruction only, never executes/retunes language model."""
import hashlib,json,sys
from pathlib import Path
import torch
torch.set_num_threads(1)
p=Path(sys.argv[1]);f=json.loads((p/'FIT-FROZEN.json').read_text());Z={}
for r in f:
 if r['arm']!='BitNet-v1':continue
 s=r['seed'];path=p/f'BitNet-v1-{s}.pt'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==r['checkpoint_sha']
 o=torch.load(path,map_location='cpu',weights_only=True)['net'];metrics=[]
 for k,v in o.items():
  if not k.startswith('layers.') or not k.endswith('.weight') or v.ndim!=2:continue
  v=v.double()
  b=v.abs().mean();z=torch.where(v>v.mean(),1.,-1.).double()
  ls=(v*z).mean().clamp_min(0.)
  q0=(v-b*z).square().sum();q1=(v-ls*z).square().sum()
  assert q1<=q0+1e-11 and ls<=b+1e-11
  metrics.append(dict(matrix=k,weight_count=v.numel(),beta_original=float(b),beta_LS=float(ls),SS_original=float(q0),SS_LS=float(q1)))
 Z[s]=metrics
print(json.dumps(dict(seeds=Z,scope='fixed centered signs; weight-space SSE only; no inference/training result'),allow_nan=False,separators=(',',':')))
