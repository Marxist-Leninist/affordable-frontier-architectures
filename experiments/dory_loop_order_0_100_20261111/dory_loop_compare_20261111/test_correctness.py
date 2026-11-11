"""Tiny numerical checks; DEFER until current NAT timing completes."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from model import LoopedNAT,count
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.manual_seed(5);a=LoopedNAT('split',d=8,hidden=16)
torch.manual_seed(5);b=LoopedNAT('whole',d=8,hidden=16)
assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
assert a.schedule()==list(range(10))*2+list(range(10,20))*2
assert b.schedule()==list(range(20))*2
p=torch.tensor([[3,2,1]])
for model in (a,b):
    calls=[0]*36;handles=[]
    for i,block in enumerate(list(model.input_blocks)+list(model.blocks)+list(model.output_blocks)):
        def hook(_m,_i,_o,idx=i):calls[idx]+=1
        handles.append(block.register_forward_hook(hook))
    z=model(p,out_len=2);assert z.shape==(1,2,256) and torch.isfinite(z).all()
    torch.nn.functional.cross_entropy(z.flatten(0,1),torch.tensor([1,2])).backward()
    assert calls==[1]*8+[2]*20+[1]*8
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
    for h in handles:h.remove()
a.loops=b.loops=1
assert torch.equal(a(p,2),b(p,2)), 'one-loop schedules must coincide'
a.loops=b.loops=2
assert not torch.equal(a(p,2),b(p,2)), 'two-loop schedules should differ'
full=LoopedNAT('split');assert count(full)==count(LoopedNAT('whole'))==764304
outer=sum(count(m) for m in list(full.input_blocks)+list(full.output_blocks))
assert outer+count(full.embed)+count(full.output)+count(full.norm)==353424
# Zero bypasses only the middle; outer attention still mixes the prefix.
a.loops=b.loops=0
assert a.schedule()==b.schedule()==[]
assert not torch.equal(a(p,2),a(p+1,2)), 'outer attention must preserve prefix conditioning'
assert torch.equal(a(p,2),b(p,2))
a.zero_grad(set_to_none=True)
calls=[0]*36;handles=[]
for i,block in enumerate(list(a.input_blocks)+list(a.blocks)+list(a.output_blocks)):
    def hook(_m,_i,_o,idx=i):calls[idx]+=1
    handles.append(block.register_forward_hook(hook))
z=a(p,2);torch.nn.functional.cross_entropy(z.flatten(0,1),torch.tensor([1,2])).backward()
assert calls==[1]*8+[0]*20+[1]*8
assert all(t.grad is None for layer in a.blocks for t in layer.parameters())
assert all(t.grad is not None and torch.isfinite(t.grad).all() for layer in list(a.input_blocks)+list(a.output_blocks) for t in layer.parameters())
for h in handles:h.remove()
for loops in range(101):
    a.loops=b.loops=loops
    assert len(a.schedule())==len(b.schedule())==20*loops
    assert len(a.input_blocks)+len(a.schedule())+len(a.output_blocks)==16+20*loops
print('PASS: exact shared initialization,8/20/8 topology,finite gradients,zero-middle prefix conditioning,764304 total/353424 zero-active parameters,all101 schedules')
