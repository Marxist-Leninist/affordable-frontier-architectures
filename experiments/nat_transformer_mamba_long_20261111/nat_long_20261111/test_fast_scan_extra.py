import copy,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'nat_compare_20261111'))
from models import NAT,MambaDirection,scan_parallel,scan_sequential
from fast_scan import scan_fast

torch.set_num_threads(1);torch.manual_seed(15)
results={}
a=(torch.randn(2,17,3,4)*.8).requires_grad_();b=torch.randn_like(a,requires_grad=True)
g=torch.randn_like(a).transpose(2,3).contiguous().transpose(2,3)
y=scan_fast(a,b);r=scan_sequential(a,b)
gy=torch.autograd.grad(y,(a,b),g);gr=torch.autograd.grad(r,(a,b),g)
assert torch.equal(y,r)
assert all(torch.equal(x,z) for x,z in zip(gy,gr))
results['signed_coefficients_noncontiguous_upstream_gradient_exact']=True
base=NAT('mamba',d=48,layers=2)
x=torch.randint(0,256,(2,8));target=torch.randint(0,256,(2,4))
records=[]
for fn in (scan_parallel,scan_sequential,scan_fast):
 m=copy.deepcopy(base)
 for module in m.modules():
  if isinstance(module,MambaDirection):module.scan=fn
 opt=torch.optim.AdamW(m.parameters(),lr=.001)
 y=m(x,out_len=4);loss=torch.nn.functional.cross_entropy(y.reshape(-1,256),target.flatten());loss.backward()
 grads={n:p.grad.clone() for n,p in m.named_parameters()};opt.step()
 records.append((y.detach(),grads,{n:p.detach().clone() for n,p in m.named_parameters()}))
for label,ref in zip(('parallel','sequential'),records[:2]):
 fast=records[2];diffs={}
 torch.testing.assert_close(ref[0],fast[0],rtol=2e-5,atol=2e-6)
 diffs['output_max_abs']=float((ref[0]-fast[0]).abs().max())
 for idx,name in [(1,'gradients'),(2,'adamw_parameters')]:
  for key in ref[idx]:torch.testing.assert_close(ref[idx][key],fast[idx][key],rtol=2e-5,atol=2e-6)
  diffs[name+'_max_abs']=max(float((ref[idx][key]-fast[idx][key]).abs().max()) for key in ref[idx])
 results['full_nat_vs_'+label]=diffs
Path(__file__).with_name('fast_scan_extra_results.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
