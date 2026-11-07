"""Independent causal recurrent-depth denoising pilot. No upstream code imported."""
import math
import torch
from torch import nn
from torch.nn import functional as F

D=48; H=4; FF=128; V=256; CONTEXT=64

class Attention(nn.Module):
    def __init__(self):
        super().__init__(); self.q=nn.Linear(D,D,bias=False); self.kv=nn.Linear(D,2*D,bias=False); self.out=nn.Linear(D,D,bias=False)
    def forward(self,q,kv):
        b,t,d=q.shape; k,v=self.kv(kv).chunk(2,-1); q=self.q(q)
        q,k,v=[a.reshape(b,t,H,D//H).transpose(1,2) for a in (q,k,v)]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(a.transpose(1,2).reshape(b,t,d))

class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.n1=nn.LayerNorm(D);self.n2=nn.LayerNorm(D);self.attn=Attention();self.ff=nn.Sequential(nn.Linear(D,FF),nn.GELU(),nn.Linear(FF,D))
    def forward(self,x,context=None):
        n=self.n1(x); x=x+self.attn(n,n if context is None else context);return x+self.ff(self.n2(x))

class Recurrent(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(V,D);self.pos=nn.Embedding(CONTEXT,D);self.prelude=Block();self.core=Block();self.inp=nn.Linear(D,D,bias=False)
        self.time=nn.Sequential(nn.Linear(16,D),nn.SiLU(),nn.Linear(D,D));self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,V)
    def codebook(self):return F.normalize(self.emb.weight,dim=-1)
    def context(self,x):return self.prelude(self.emb(x)+self.pos(torch.arange(x.shape[1])))/math.sqrt(D)
    def denoise(self,c,z,s):
        # s shape B,1,1. EDM preconditioning with sigma_data=0.5.
        freq=2.**torch.arange(8,dtype=z.dtype);t=.25*s.log().flatten(1)[:,:1]*freq[None,:]
        t=self.time(torch.cat((t.sin(),t.cos()),-1))[:,None,:]
        ci=torch.rsqrt(s.square()+.25);co=.5*s*ci;cs=.25/(s.square()+.25)
        h=self.core(self.inp(z*ci)+t,c);d=cs*z+co*h
        return self.head(self.norm(d))
    def predict(self,x,noise,k=4):
        c=self.context(x);z=noise*80.;sigmas=schedule(k)
        for i,s in enumerate(sigmas):
            ss=z.new_full((x.shape[0],1,1),float(s));logits=self.denoise(c,z,ss)
            if i+1<len(sigmas):
                d=logits.softmax(-1)@self.codebook();z=z+(float(sigmas[i+1])-float(s))/float(s)*(z-d)
        return logits
    def train_denoise(self,x,y,noise,s):
        z=F.embedding(y,self.codebook())+s*noise
        return self.denoise(self.context(x),z,s)

def schedule(k):
    # Karras rho=7 schedule, k denoiser calls; last sigma .002.
    return (80.**(1/7)+torch.linspace(0,1,k)*(.002**(1/7)-80.**(1/7)))**7

class Transformer(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(V,D);self.pos=nn.Embedding(CONTEXT,D);self.blocks=nn.ModuleList([Block(),Block()]);self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,V)
    def forward(self,x):
        h=self.emb(x)+self.pos(torch.arange(x.shape[1]))
        for block in self.blocks:h=block(h)
        return self.head(self.norm(h))

def count(m):return sum(p.numel() for p in m.parameters())
