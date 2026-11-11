"""Tiny one-pass NAT models; Mamba-1 equations, not a fused CUDA implementation.

The selective state-space equations and initialization follow state-spaces/mamba
(Tri Dao and Albert Gu, Apache-2.0). This independent CPU reference substitutes
an associative PyTorch scan. Both directions see only prefix + blank placeholders.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

MASK, PAD, VOCAB = 256, 257, 258

def scan_parallel(a, b):
    """Inclusive associative scan of affine maps: h[t] = a[t]*h[t-1]+b[t]."""
    step = 1
    while step < a.shape[1]:
        b = torch.cat((b[:, :step], b[:, step:] + a[:, step:] * b[:, :-step]), 1)
        a = torch.cat((a[:, :step], a[:, step:] * a[:, :-step]), 1)
        step *= 2
    return b

def scan_sequential(a, b):
    h = torch.zeros_like(b[:, 0]); out=[]
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, 1)

class RMS(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True)+1e-6) * self.weight

class MambaDirection(nn.Module):
    def __init__(self, d=48, state=8):
        super().__init__(); self.inner=2*d; self.state=state; self.rank=math.ceil(d/16)
        self.in_proj=nn.Linear(d,4*d,bias=False)
        self.conv=nn.Conv1d(2*d,2*d,4,groups=2*d,padding=3)
        self.x_proj=nn.Linear(2*d,self.rank+2*state,bias=False)
        self.dt_proj=nn.Linear(self.rank,2*d,bias=True)
        nn.init.uniform_(self.dt_proj.weight,-self.rank**-.5,self.rank**-.5)
        dt=torch.exp(torch.rand(2*d)*(math.log(.1)-math.log(.001))+math.log(.001))
        with torch.no_grad(): self.dt_proj.bias.copy_(dt+torch.log(-torch.expm1(-dt)))
        self.A_log=nn.Parameter(torch.arange(1,state+1).float().log().repeat(2*d,1))
        self.D=nn.Parameter(torch.ones(2*d))
        self.out_proj=nn.Linear(2*d,d,bias=False)
        self.scan=scan_parallel
    def forward(self,x):
        u,z=self.in_proj(x).chunk(2,-1)
        u=F.silu(self.conv(u.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        dt,b,c=self.x_proj(u).split([self.rank,self.state,self.state],-1)
        dt=F.softplus(self.dt_proj(dt))
        a=torch.exp(-dt.unsqueeze(-1)*self.A_log.exp())
        bu=dt.unsqueeze(-1)*b.unsqueeze(-2)*u.unsqueeze(-1)
        h=self.scan(a,bu)
        y=(h*c.unsqueeze(-2)).sum(-1)+u*self.D
        return self.out_proj(y*F.silu(z))

class MambaBlock(nn.Module):
    def __init__(self,d):
        super().__init__(); self.norm=RMS(d); self.forward_dir=MambaDirection(d); self.reverse_dir=MambaDirection(d)
    def forward(self,x):
        z=self.norm(x)
        return x+(self.forward_dir(z)+self.reverse_dir(z.flip(1)).flip(1))/math.sqrt(2)

class AttentionBlock(nn.Module):
    def __init__(self,d,hidden=266):
        super().__init__(); self.d=d; self.heads=4
        self.n1=RMS(d); self.n2=RMS(d)
        self.qkv=nn.Linear(d,3*d,bias=False); self.out=nn.Linear(d,d,bias=False)
        self.up=nn.Linear(d,hidden,bias=False); self.down=nn.Linear(hidden,d,bias=False)
    def rope(self,x):
        t,hd=x.shape[-2:]; freq=10000**(-torch.arange(0,hd,2,device=x.device,dtype=x.dtype)/hd)
        angle=torch.arange(t,device=x.device,dtype=x.dtype)[:,None]*freq
        a,b=x[...,::2],x[...,1::2]
        return torch.stack((a*angle.cos()-b*angle.sin(),a*angle.sin()+b*angle.cos()),-1).flatten(-2)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.n1(x)).view(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        q,k=self.rope(q),self.rope(k)
        att=F.scaled_dot_product_attention(q,k,v,is_causal=False)
        x=x+self.out(att.transpose(1,2).reshape(b,t,d))
        return x+self.down(F.relu(self.up(self.n2(x))).square())

class NAT(nn.Module):
    def __init__(self,kind,d=48,hidden=266,layers=2):
        super().__init__(); self.kind=kind; self.d=d
        self.embed=nn.Embedding(VOCAB,d)
        nn.init.normal_(self.embed.weight,std=.02)
        self.blocks=nn.ModuleList([AttentionBlock(d,hidden) if kind=='transformer' else MambaBlock(d) for _ in range(layers)])
        self.norm=RMS(d); self.output=nn.Linear(d,256,bias=False)
    def forward(self,prefix,out_len=16):
        blank=torch.full((prefix.shape[0],out_len),MASK,dtype=torch.long,device=prefix.device)
        ids=torch.cat((prefix,blank),1)
        pos=torch.arange(ids.shape[1],device=ids.device,dtype=self.embed.weight.dtype)[:,None]
        freq=torch.exp(torch.arange(0,self.d,2,device=ids.device,dtype=self.embed.weight.dtype)*(-math.log(10000)/self.d))
        pe=torch.stack(((pos*freq).sin(),(pos*freq).cos()),-1).flatten(-2)
        x=self.embed(ids)*math.sqrt(self.d)+.1*pe
        for layer in self.blocks: x=layer(x)
        return self.output(self.norm(x[:,-out_len:]))

def count(model): return sum(p.numel() for p in model.parameters())
