"""Controlled NAT recurrent-order motif; NOT an NVIDIA Dory reproduction."""
import math
import torch
from torch import nn
from torch.nn import functional as F
MASK,PAD,VOCAB=256,257,258

class RMS(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True)+1e-6) * self.weight

class AttentionBlock(nn.Module):
    def __init__(self,d,hidden=654):
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

class LoopedNAT(nn.Module):
    def __init__(self,kind,d=48,hidden=117,loops=2):
        super().__init__(); self.kind=kind; self.d=d
        if kind not in ('split','whole'): raise ValueError('unknown topology')
        self.loops=loops
        if not isinstance(loops,int) or loops < 0: raise ValueError('loops must be a nonnegative integer')
        self.embed=nn.Embedding(VOCAB,d)
        nn.init.normal_(self.embed.weight,std=.02)
        self.output=nn.Linear(d,256,bias=False)
        self.input_blocks=nn.ModuleList([AttentionBlock(d,hidden) for _ in range(8)])
        self.blocks=nn.ModuleList([AttentionBlock(d,hidden) for _ in range(20)])
        self.output_blocks=nn.ModuleList([AttentionBlock(d,hidden) for _ in range(8)])
        self.norm=RMS(d)
    def forward(self,prefix,out_len=32):
        blank=torch.full((prefix.shape[0],out_len),MASK,dtype=torch.long,device=prefix.device)
        ids=torch.cat((prefix,blank),1)
        pos=torch.arange(ids.shape[1],device=ids.device,dtype=self.embed.weight.dtype)[:,None]
        freq=torch.exp(torch.arange(0,self.d,2,device=ids.device,dtype=self.embed.weight.dtype)*(-math.log(10000)/self.d))
        pe=torch.stack(((pos*freq).sin(),(pos*freq).cos()),-1).flatten(-2)
        x=self.embed(ids)*math.sqrt(self.d)+.1*pe
        for layer in self.input_blocks: x=layer(x)
        for idx in self.schedule(): x=self.blocks[idx](x)
        for layer in self.output_blocks: x=layer(x)
        return self.output(self.norm(x[:,-out_len:]))

    def schedule(self):
        a=list(range(10)); b=list(range(10,20))
        return a*self.loops+b*self.loops if self.kind=='split' else (a+b)*self.loops

def count(model): return sum(p.numel() for p in model.parameters())
