import math
import torch
from torch import nn
class Block(nn.Module):
 def __init__(self,d=32):
  super().__init__(); self.n1=nn.LayerNorm(d); self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d); self.n2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,d))
 def forward(self,x,local=False,final=False,reference=False):
  b,t,d=x.shape; z=self.n1(x); q,k,v=[a.view(b,t,4,d//4).transpose(1,2) for a in self.qkv(z).chunk(3,-1)]
  if local and not reference:
   kp=torch.cat([k[:,:,:1],k[:,:,:-1]],2); vp=torch.cat([v[:,:,:1],v[:,:,:-1]],2)
   score=torch.stack([(q*kp).sum(-1),(q*k).sum(-1)],-1)/math.sqrt(d//4); score[:,:,0,0]=-torch.inf
   w=score.softmax(-1); a=w[...,0,None]*vp+w[...,1,None]*v
  else:
   if final: q=q[:,:,-1:]
   score=q@k.transpose(-2,-1)/math.sqrt(d//4)
   if not final:
    p=torch.arange(t,device=x.device); mask=p[None,:]>p[:,None]
    if local: mask=mask|(p[None,:]<p[:,None]-1)
    score=score.masked_fill(mask,-torch.inf)
   a=score.softmax(-1)@v
  a=a.transpose(1,2).reshape(b,-1,d); x=(x[:,-1:] if final else x)+self.proj(a); return x+self.ff(self.n2(x))
class Transformer(nn.Module):
 def __init__(self):
  super().__init__(); self.emb=nn.Embedding(41,32); self.local=Block(); self.global_=Block(); self.norm=nn.LayerNorm(32); self.out=nn.Linear(32,16)
 def forward(self,x,reference=False):
  z=self.local(self.emb(x),local=True,reference=reference); z=self.global_(z,final=not reference,reference=reference); return self.out(self.norm(z[:,-1]))
class RNN(nn.Module):
 def __init__(self,h):
  super().__init__(); self.emb=nn.Embedding(41,32); self.rnn=nn.RNN(32,h,batch_first=True,nonlinearity='tanh'); self.out=nn.Linear(h,16)
 def forward(self,x): return self.out(self.rnn(self.emb(x))[1][-1])
def count(m): return sum(p.numel() for p in m.parameters())
def hidden(): return min(range(16,200),key=lambda h:abs(count(RNN(h))-count(Transformer())))
def make(name): return Transformer() if name=='transformer' else RNN(hidden())
