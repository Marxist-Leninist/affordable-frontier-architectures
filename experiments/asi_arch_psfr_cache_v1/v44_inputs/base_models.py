"""Small modern causal Transformer and gated recurrent language models."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self,d):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x): return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*self.weight

class Attention(nn.Module):
    def __init__(self,d=64,heads=4):
        super().__init__(); self.h=heads; self.d=d; self.qkv=nn.Linear(d,3*d,bias=False); self.out=nn.Linear(d,d,bias=False)
    def rope(self,x,offset):
        t=x.shape[-2]; k=x.shape[-1]; freqs=10000.**(-torch.arange(0,k,2,device=x.device,dtype=x.dtype)/k)
        a=torch.arange(offset,offset+t,device=x.device,dtype=x.dtype)[:,None]*freqs
        u,v=x[...,::2],x[...,1::2]
        return torch.stack((u*a.cos()-v*a.sin(),u*a.sin()+v*a.cos()),dim=-1).flatten(-2)
    def forward(self,x,state=None,cache=False):
        b,t,d=x.shape; q,k,v=self.qkv(x).chunk(3,-1)
        q,k,v=[a.reshape(b,t,self.h,d//self.h).transpose(1,2) for a in (q,k,v)]
        offset=0 if state is None else state[0].shape[-2]
        q,k=self.rope(q,offset),self.rope(k,offset)
        if state is not None: k=torch.cat((state[0],k),-2); v=torch.cat((state[1],v),-2)
        if offset:
            mask=torch.arange(k.shape[-2],device=x.device)[None,:]<=torch.arange(offset,offset+t,device=x.device)[:,None]
            z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        else: z=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(z.transpose(1,2).reshape(b,t,d)),(k,v) if cache else None

class Block(nn.Module):
    def __init__(self,kind,d=64):
        super().__init__(); self.kind=kind; inner=172 if kind=='transformer' else 128
        self.n1=RMSNorm(d);self.n2=RMSNorm(d)
        self.mix=Attention(d) if kind=='transformer' else nn.GRU(d,d,batch_first=True)
        self.gate=nn.Linear(d,inner,bias=False);self.up=nn.Linear(d,inner,bias=False);self.down=nn.Linear(inner,d,bias=False)
    def forward(self,x,state=None,cache=False):
        if self.kind=='transformer': z,s=self.mix(self.n1(x),state,cache)
        else: z,s=self.mix(self.n1(x),state)
        x=x+z; z=self.n2(x); return x+self.down(F.silu(self.gate(z))*self.up(z)),s if cache else None

class LM(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.emb=nn.Embedding(256,64);self.blocks=nn.ModuleList([Block(kind) for _ in range(2)]);self.norm=RMSNorm(64);self.head=nn.Linear(64,256,bias=False)
    def forward(self,x,states=None,cache=False):
        x=self.emb(x); new=[]
        for i,b in enumerate(self.blocks): x,s=b(x,None if states is None else states[i],cache);new.append(s)
        logits=self.head(self.norm(x));return (logits,new) if cache else logits

def count(m):return sum(p.numel() for p in m.parameters())
def state_bytes(s):
    if s is None:return 0
    if isinstance(s,torch.Tensor):return s.numel()*s.element_size()
    return sum(state_bytes(v) for v in s)
