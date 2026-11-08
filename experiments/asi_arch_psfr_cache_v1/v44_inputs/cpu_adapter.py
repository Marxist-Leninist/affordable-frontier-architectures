"""CPU dependency adapters, NOT an official FLA implementation.
Original ASI source remains unchanged; only @torch.compile is removed at load.
No masked/packed sequences supported. No claim of fused numerical equivalence.
"""
import sys, types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
class RMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-5,**kw):
        super().__init__();self.weight=nn.Parameter(torch.ones(hidden_size));self.eps=eps
    def forward(self,x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.eps)*self.weight
class FusedRMSNormGated(RMSNorm):
    def forward(self,x,g):return super().forward(x)*F.silu(g)
class ShortConvolution(nn.Conv1d):
    def __init__(self,hidden_size,kernel_size,activation=None,bias=False,**kw):
        super().__init__(hidden_size,hidden_size,kernel_size,groups=hidden_size,bias=bias);self.activation=activation;self.k=kernel_size
    def forward(self,x,cache=None,output_final_state=False,cu_seqlens=None):
        if cu_seqlens is not None:raise NotImplementedError('Packed sequences intentionally unsupported')
        x=x.transpose(1,2);old=x.new_zeros(x.shape[0],x.shape[1],self.k-1) if cache is None else cache[...,-(self.k-1):]
        combined=torch.cat((old,x),-1);y=super().forward(combined).transpose(1,2)
        if self.activation=='silu':y=F.silu(y)
        return y,combined[...,-self.k:].clone() if output_final_state else None

def unsupported(*a,**kw):raise NotImplementedError('Masks/packed sequences intentionally unsupported')
def load(name):
    for s in ('fla','fla.layers','fla.layers.utils','fla.modules','fla.modules.l2norm'):sys.modules[s]=types.ModuleType(s)
    for n in ('get_unpad_data','index_first_axis','pad_input'):setattr(sys.modules['fla.layers.utils'],n,unsupported)
    for n in ('RMSNorm','FusedRMSNormGated','ShortConvolution'):setattr(sys.modules['fla.modules'],n,globals()[n])
    sys.modules['fla.modules.l2norm'].l2norm=lambda x:x*torch.rsqrt(x.square().sum(-1,keepdim=True)+1e-6)
    path=Path(__file__).parent/'original'/f'delta_net_{name}.py';s=path.read_text();s='\n'.join(l for l in s.splitlines() if not l.startswith('@torch.compile'))
    m=types.ModuleType('asi_'+name);exec(compile(s,str(path),'exec'),m.__dict__);return m
class Cache:
    def __init__(self):self.layers={}
    def __len__(self):return len(self.layers)
    def __getitem__(self,i):return self.layers[i]
    def update(self,layer_idx,**kw):self.layers[layer_idx]=kw
