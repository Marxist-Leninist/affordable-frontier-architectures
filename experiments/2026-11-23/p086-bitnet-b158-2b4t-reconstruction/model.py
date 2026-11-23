"""Independent small-scale model from the b1.58 paper and published shape semantics."""
import torch
from torch import nn
from torch.nn import functional as F
D=192;H=4;H_KV=1;L=4;J=D//H;M=512;T=128;V=256;THETA=500000.;EPS=1.e-5
def ternary(w):
 raw=w.detach()
 beta=raw.abs().mean().clamp_min(EPS)
 mul=beta.reciprocal()
 code=(raw*mul).round().clamp(-1.,1.)
 return (w-raw)+code/mul,code,beta
def act8(x):
 raw=x.detach()
 alpha=127./raw.abs().amax(-1,keepdim=True).clamp_min(EPS)
 q=(raw*alpha).round().clamp(-128,127)
 return (x-raw)+q/alpha,q,alpha
class Norm(nn.Module):
 def __init__(self,d):
  super().__init__();self.gain=nn.Parameter(torch.ones(d))
 def forward(self,x):
  return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+EPS)*self.gain
class Linear(nn.Module):
 def __init__(self,d,f,quant):
  super().__init__();self.quant=quant;self.weight=nn.Parameter(torch.empty(f,d))
  nn.init.normal_(self.weight,std=.02)
  self.frozen=None
 def forward(self,x):
  if not self.quant:return F.linear(x,self.weight)
  x=act8(x)[0]
  w=ternary(self.weight)[0] if self.frozen is None else self.frozen
  return F.linear(x,w)
def rotate(v,c,s):
 a,b=v.chunk(2,-1)
 return v*c+torch.cat((-b,a),-1)*s
class Layer(nn.Module):
 def __init__(self,quant):
  super().__init__();self.pre_att=Norm(D);self.post_mix=Norm(D)
  self.q=Linear(D,D,quant);self.k=Linear(D,H_KV*J,quant);self.v=Linear(D,H_KV*J,quant);self.out=Linear(D,D,quant)
  self.pre_ff=Norm(D);self.gate=Linear(D,M,quant);self.up=Linear(D,M,quant)
  self.post_ff_hidden=Norm(M);self.down=Linear(M,D,quant)
 def forward(self,x,c,s):
  z=self.pre_att(x)
  def shape(a,n):return a.unflatten(-1,(n,J)).transpose(1,2)
  q=rotate(shape(self.q(z),H),c,s)
  # Explicit duplicate of K/V heads only for reference math, not optimized storage.
  k=rotate(shape(self.k(z),H_KV),c,s).expand(-1,H,-1,-1)
  v=shape(self.v(z),H_KV).expand(-1,H,-1,-1)
  u=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).flatten(-2)
  h=x+self.out(self.post_mix(u))
  z=self.pre_ff(h)
  z=F.relu(self.gate(z)).square()*self.up(z)
  return h+self.down(self.post_ff_hidden(z))
class Net(nn.Module):
 def __init__(self,quant):
  super().__init__();self.emb=nn.Embedding(V,D);self.layers=nn.ModuleList(Layer(quant) for _ in range(L));self.final=Norm(D);self.bit=quant
  nn.init.normal_(self.emb.weight,std=.02)
  phase=torch.outer(torch.arange(T,dtype=torch.float32),THETA**(-torch.arange(0,J,2,dtype=torch.float32)/J))
  phase=torch.cat((phase,phase),-1)
  self.register_buffer('c',phase.cos(),persistent=False);self.register_buffer('s',phase.sin(),persistent=False)
 def forward(self,t):
  h=self.emb(t);c,s=self.c[:t.shape[1]],self.s[:t.shape[1]]
  for l in self.layers:h=l(h,c,s)
  return F.linear(self.final(h),self.emb.weight)
