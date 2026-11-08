"""Exactly parameter-matched AAA and attention/diagonal-SSM/attention models."""
import math,torch
from torch import nn
from torch.nn import functional as F
from base_models import Block,RMSNorm
from source_v38 import GQA

class DiagonalSSM(nn.Module):
    """Stable learned LTI state-space core, input/output projections and SiLU gate.

    h[t,d,n] = a[d,n] h[t-1,d,n] + (1-a[d,n]) B[d,n] u[t,d]
    y[t,d] = sum_n C[d,n] h[t,d,n] + D[d] u[t,d]
    Output = out(y * silu(gate)). No GRU, no selective/state-dependent poles.
    """
    def __init__(self,d=64,n=21):
        super().__init__();self.d=d;self.n=n
        self.in_proj=nn.Linear(d,2*d,bias=False);self.out=nn.Linear(d,d,bias=False)
        tau=torch.exp(torch.linspace(0,math.log(512),n));a=torch.exp(-1/tau)
        self.logit_decay=nn.Parameter(torch.logit(a).repeat(d,1))
        self.B=nn.Parameter(torch.randn(d,n));self.C=nn.Parameter(torch.randn(d,n)/math.sqrt(n));self.D=nn.Parameter(torch.ones(d))
    def forward(self,x,state=None,cache=False):
        u,gate=self.in_proj(x).chunk(2,-1);b,t,d=u.shape;a=self.logit_decay.sigmoid();coeff=(1-a)*self.B
        if t==1:
            h=coeff[None]*u[:,0,:,None] if state is None else a[None]*state+coeff[None]*u[:,0,:,None]
            y=(h*self.C[None]).sum(-1)[:,None]+u*self.D
            return self.out(y*F.silu(gate)),h if cache else None
        power=a[:,:,None]**torch.arange(t,device=x.device,dtype=x.dtype)[None,None,:]
        kernel=(coeff*self.C)[:,:,None].mul(power).sum(1)
        nfft=1<<(2*t-1).bit_length()
        y=torch.fft.irfft(torch.fft.rfft(u.transpose(1,2),n=nfft)*torch.fft.rfft(kernel,n=nfft)[None],n=nfft)[...,:t].transpose(1,2)
        if state is not None:y=y+torch.einsum('bdn,dnt,dn->btd',state,power*a[:,:,None],self.C)
        y=y+u*self.D
        h=None
        if cache:
            h=coeff[None]*torch.einsum('btd,dnt->bdn',u,power.flip(-1))
            if state is not None:h=h+state*a[None]**t
        return self.out(y*F.silu(gate)),h

class MixedBlock(Block):
    def __init__(self):
        super().__init__('transformer');self.mix=GQA(4)

class AlternatingLM(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(256,64);self.blocks=nn.ModuleList([MixedBlock() for _ in range(3)]);self.norm=RMSNorm(64);self.head=nn.Linear(64,256,bias=False)
    def forward(self,x,states=None,cache=False):
        x=self.emb(x);new=[]
        for i,b in enumerate(self.blocks):x,s=b(x,None if states is None else states[i],cache);new.append(s)
        logits=self.head(self.norm(x));return (logits,new) if cache else logits

def build(seed,arm):
    torch.manual_seed(seed);m=AlternatingLM()
    if arm=='ASA':
        torch.manual_seed(40000+seed);m.blocks[1].mix=DiagonalSSM()
    else:assert arm=='AAA'
    return m

def state_bytes(states):
    def tensors(s):
        if s is None:return []
        if torch.is_tensor(s):return [s]
        return sum((tensors(v) for v in s),[])
    tt=tensors(states);logical=sum(t.numel()*t.element_size() for t in tt)
    stores={(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()):t.untyped_storage().nbytes() for t in tt}
    return logical,sum(stores.values())
