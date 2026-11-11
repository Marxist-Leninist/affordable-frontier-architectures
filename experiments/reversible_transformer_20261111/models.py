"""Deterministic additive-coupling Transformer, and a matched residual baseline.

Only the paired hidden-state stack is invertible. Embedding, readout, softmax,
argmax, and training updates are NOT asserted invertible. No stochastic dropout.
"""
import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self,d=48,heads=4):
        super().__init__(); self.heads=heads; self.d=d
        self.norm=nn.LayerNorm(d); self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
    def forward(self,x):
        b,t,d=x.shape; h=self.heads
        q,k,v=self.qkv(self.norm(x)).reshape(b,t,3,h,d//h).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
        return self.proj(a.transpose(1,2).reshape(b,t,d))

class FeedForward(nn.Module):
    def __init__(self,d=48,ff=96):
        super().__init__(); self.norm=nn.LayerNorm(d); self.up=nn.Linear(d,ff); self.down=nn.Linear(ff,d)
    def forward(self,x):return self.down(F.gelu(self.up(self.norm(x))))

class Block(nn.Module):
    def __init__(self,d=48,heads=4,ff=96):
        super().__init__(); self.f=Attention(d,heads); self.g=FeedForward(d,ff)
    def ordinary(self,x):
        x=x+self.f(x);return x+self.g(x)
    def forward(self,x1,x2):
        y1=x1+self.f(x2);return y1,x2+self.g(y1)
    def inverse(self,y1,y2):
        x2=y2-self.g(y1);return y1-self.f(x2),x2

def stack_forward(x1,x2,blocks):
    for block in blocks:x1,x2=block(x1,x2)
    return x1,x2

def stack_inverse(y1,y2,blocks):
    for block in reversed(blocks):y1,y2=block.inverse(y1,y2)
    return y1,y2

class InverseBackprop(torch.autograd.Function):
    """First-order reverse-mode AD with activation reconstruction.

Not inverse learning, target propagation, or a gradient-free optimizer.
A missing stochastic RNG replay would be a bug; this model has no dropout.
Parameter objects must not be mutated between forward and backward.
"""
    @staticmethod
    def forward(ctx,x1,x2,blocks,*parameters):
        ctx.blocks=blocks
        # Function.forward already runs without recording its internal graph.
        y1,y2=stack_forward(x1,x2,blocks)
        ctx.save_for_backward(y1,y2)
        return y1,y2
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,dy1,dy2):
        y1,y2=ctx.saved_tensors; grads=[]
        # Parameter order follows block f then block g, matching Module.parameters.
        for block in reversed(ctx.blocks):
            fp=tuple(block.f.parameters());gp=tuple(block.g.parameters())
            with torch.enable_grad():
                z1=y1.detach().requires_grad_();g=block.g(z1)
                dg=torch.autograd.grad(g,(z1,*gp),dy2)
            with torch.no_grad():
                x2=y2-g.detach(); dx1=dy1+dg[0]
            with torch.enable_grad():
                z2=x2.detach().requires_grad_();f=block.f(z2)
                df=torch.autograd.grad(f,(z2,*fp),dx1)
            with torch.no_grad():
                x1=y1-f.detach();dx2=dy2+df[0]
            grads.insert(0,(*df[1:],*dg[1:])); y1,y2=x1,x2;dy1,dy2=dx1,dx2
        return (dy1,dy2,None,*(p for bg in grads for p in bg))

class LanguageModel(nn.Module):
    def __init__(self,mode='ordinary',d=48,depth=4,heads=4,ff=96,context=64):
        super().__init__();self.mode=mode;self.context=context
        self.embedding=nn.Embedding(256,d);self.position=nn.Embedding(context,d)
        self.blocks=nn.ModuleList([Block(d,heads,ff) for _ in range(depth)])
        self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,256,bias=False)
        self.head.weight=self.embedding.weight
        nn.init.normal_(self.embedding.weight,std=.02);nn.init.normal_(self.position.weight,std=.02)
    def forward(self,idx):
        h=self.embedding(idx)+self.position(torch.arange(idx.shape[1],device=idx.device))
        if self.mode=='ordinary':
            for block in self.blocks:h=block.ordinary(h)
        else:
            if self.mode=='reversible_inverse' and torch.is_grad_enabled():
                a,b=InverseBackprop.apply(h,h,self.blocks,*tuple(self.blocks.parameters()))
            else:a,b=stack_forward(h,h,self.blocks)
            h=(a+b)*.5
        return self.head(self.norm(h))

def loss(logits,y):return F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
