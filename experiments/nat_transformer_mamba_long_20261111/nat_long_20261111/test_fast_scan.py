import copy
import json
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nat_compare_20261111'))
from models import scan_sequential, scan_parallel, MambaDirection
import fast_scan
from fast_scan import scan_fast

torch.set_num_threads(1)
torch.manual_seed(319)
results = {'threads': torch.get_num_threads(), 'torch': torch.__version__, 'checks': [], 'benchmarks': []}
t0=time.perf_counter()
scan_fast(torch.ones(1,2,1,1),torch.ones(1,2,1,1))
results['first_use_seconds']=time.perf_counter()-t0
results['compile_seconds']=fast_scan.COMPILE_SECONDS

def compare(x,y):
    return float((x-y).abs().max())

for dtype in (torch.float32, torch.float64):
    for shape in [(1,1,1,1),(2,7,3,5),(3,33,8,4),(2,128,16,8)]:
        for noncontiguous in (False,True):
            a=torch.rand(shape,dtype=dtype)*.98
            b=torch.randn(shape,dtype=dtype)
            if noncontiguous:
                a=a.transpose(2,3).contiguous().transpose(2,3)
                b=b.transpose(2,3).contiguous().transpose(2,3)
            a.requires_grad_();b.requires_grad_()
            g=torch.randn(shape,dtype=dtype)
            ref=scan_sequential(a,b);out=scan_fast(a,b)
            rg=torch.autograd.grad(ref,(a,b),g)
            og=torch.autograd.grad(out,(a,b),g)
            for x,y in [(ref,out),*zip(rg,og)]:
                torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
                assert torch.isfinite(y).all()
            results['checks'].append({'dtype':str(dtype),'shape':shape,'noncontiguous':noncontiguous,'forward_max_abs':compare(ref,out),'grad_a_max_abs':compare(rg[0],og[0]),'grad_b_max_abs':compare(rg[1],og[1])})
a=(torch.rand(1,4,2,2,dtype=torch.double)*.8).requires_grad_()
b=torch.randn_like(a,requires_grad=True)
assert torch.autograd.gradcheck(scan_fast,(a,b))
results['gradcheck']=True
for dtype in [torch.float16,torch.bfloat16,torch.int64]:
    try: scan_fast(torch.ones(1,1,1,1,dtype=dtype),torch.ones(1,1,1,1,dtype=dtype))
    except TypeError: pass
    else: raise AssertionError('unsupported dtype accepted')
for a,b in [(torch.ones(1,0,1,1),torch.ones(1,0,1,1)),(torch.ones(2,2),torch.ones(2,2)),(torch.ones(1,2,1,1),torch.ones(1,3,1,1))]:
    try: scan_fast(a,b)
    except ValueError: pass
    else: raise AssertionError('invalid shape accepted')
try: scan_fast(torch.ones(1,1,1,1),torch.ones(1,1,1,1,dtype=torch.double))
except TypeError: pass
else: raise AssertionError('mixed dtype accepted')
results['validation_tests']=True
for d,state,T,B in [(48,8,48,16),(128,16,128,16)]:
    base=MambaDirection(d,state)
    ref=copy.deepcopy(base);ref.scan=scan_sequential
    fast=copy.deepcopy(base);fast.scan=scan_fast
    x=torch.randn(B,T,d)
    target=torch.randn_like(x)
    model_checks={}
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.001) for m in (ref,fast)]
    outs=[];grads=[];inputs=[]
    for m,opt in zip((ref,fast),optimizers):
        xi=x.clone().requires_grad_(); y=m(xi); loss=(y-target).square().mean();loss.backward()
        outs.append(y.detach());grads.append({n:p.grad.clone() for n,p in m.named_parameters()});inputs.append(xi.grad.clone())
        opt.step()
    torch.testing.assert_close(outs[0],outs[1],rtol=2e-5,atol=2e-6)
    for n in grads[0]: torch.testing.assert_close(grads[0][n],grads[1][n],rtol=2e-5,atol=2e-7)
    for p,q in zip(ref.parameters(),fast.parameters()): torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-7)
    torch.testing.assert_close(inputs[0],inputs[1],rtol=2e-5,atol=2e-7)
    model_checks={'output_max_abs':compare(*outs),'gradient_max_abs':max(compare(grads[0][n],grads[1][n]) for n in grads[0]),'adamw_parameter_max_abs':max(compare(p,q) for p,q in zip(ref.parameters(),fast.parameters())),'input_gradient_max_abs':compare(*inputs)}
    entry={'d':d,'state':state,'T':T,'B':B,'model_checks':model_checks,'training_step_ms':{}}
    for name,fn in [('parallel',scan_parallel),('sequential',scan_sequential),('fast',scan_fast)]:
        m=copy.deepcopy(base);m.scan=fn
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        samples=[]
        for i in range(13):
            t0=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss=(m(x)-target).square().mean();loss.backward();opt.step()
            assert torch.isfinite(loss)
            elapsed=(time.perf_counter()-t0)*1000
            if i>=3:samples.append(elapsed)
        entry['training_step_ms'][name]={'median': sorted(samples)[len(samples)//2], 'mean':sum(samples)/len(samples),'samples':samples}
    entry['speedup_vs_parallel']=entry['training_step_ms']['parallel']['median']/entry['training_step_ms']['fast']['median']
    results['benchmarks'].append(entry)
    print(json.dumps(entry),flush=True)
Path(__file__).with_name('fast_scan_results.json').write_text(json.dumps(results,indent=2))
print('PASS',json.dumps({'first_use_seconds':results['first_use_seconds'],'compile_seconds':results['compile_seconds']}),flush=True)
