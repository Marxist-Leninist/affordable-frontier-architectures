"""BAMG #259 adapted to the existing AGILLM mini hybrid interface.
Canonical mixer semantics, except dtype-preserving triangular solve replaces the
source's hard-coded BF16 intermediate. Binary qkv/o, block-wide output RMS, the
3/4 causal-mixer + 1/4 attention layout, and Dirac short-conv initialization are
shared comparison conventions, NOT the original full ASI-Arch model.
No incremental decode/cache implementation. No measured full-scale claims.
"""
import torch
from torch import nn
from torch.nn import functional as F

META = dict(id="bamg259", title="BAMG_MemoryGate #259 adapted hybrid", slot="hybrid",
            one_line="Bias-initialised delta-memory/value gate; 3/4 mixer layers and 1/4 attention")
SOURCE_PROGRAM_SHA256 = "08e3b4bb618c64be48db8cb402ab3dc46cbc1a204741710f25afb6fedc3b699e"
CHUNK = 32

def l2norm(x):
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)

def conv(x, w):
    b, h, t, d = x.shape
    z = x.transpose(-1, -2).reshape(b, h*d, t)
    z = F.conv1d(F.pad(z, (w.shape[-1]-1, 0)), w, groups=h*d)
    return F.silu(z.reshape(b,h,d,t).transpose(-1,-2))

def delta_chunked(q, k, v, beta, C=CHUNK):
    """Dtype-preserving plain delta recurrence, with source l2 normalization."""
    q, k = l2norm(q), l2norm(k)
    B,H,T,d = q.shape
    pad=(-T)%C
    if pad:
        q,k,v=(F.pad(x,(0,0,0,pad)) for x in (q,k,v))
        beta=F.pad(beta,(0,pad))
    N=(T+pad)//C
    q,k,v=(x.reshape(B,H,N,C,d) for x in (q,k,v))
    b=beta.reshape(B,H,N,C,1)
    eye=torch.eye(C,dtype=q.dtype,device=q.device)
    strict=torch.tril(torch.ones(C,C,dtype=torch.bool,device=q.device),-1)
    incl=torch.tril(torch.ones(C,C,dtype=torch.bool,device=q.device))
    lower=(b*(k@k.transpose(-1,-2))).masked_fill(~strict,0)+eye
    inv=torch.linalg.solve_triangular(lower,eye.expand_as(lower),upper=False,unitriangular=True)
    W,U0=inv@(b*k),inv@(b*v)
    Aqk=(q@k.transpose(-1,-2)).masked_fill(~incl,0)
    state=q.new_zeros(B,H,d,d)
    out=[]
    for n in range(N):
        U=U0[:,:,n]-W[:,:,n]@state
        out.append(q[:,:,n]@state+Aqk[:,:,n]@U)
        state=state+k[:,:,n].transpose(-1,-2)@U
    return torch.cat(out,2)[:,:,:T]

def patch(Block, Model, cfg):
    class BAMG(Block):
        def __init__(self,c,index):
            super().__init__(c,index)
            self.attn_layer=bool(index%4==3 or index==c["L"]-1)
            if self.attn_layer:
                return
            D,H=c["d"],self.H
            def dirac():
                w=torch.zeros(D,1,4);w[:,:,-1]=1
                return nn.Parameter(w)
            self.short_q,self.short_k,self.short_v=dirac(),dirac(),dirac()
            self.b=nn.Linear(D,H,bias=False)  # canonical #259 has no beta bias
            self.mix_proj=nn.Linear(D,H,bias=True)
            nn.init.constant_(self.mix_proj.bias,-1.0)
            self.mix_bias=nn.Parameter(torch.full((H,),-1.0)) # BOTH source biases are -1
        def mix(self,h,causal):
            if self.attn_layer:
                return super().mix(h,causal)
            B,T,D=h.shape;H=self.H;hd=D//H
            q,k,v=self.qkv(h).view(B,T,3,H,hd).permute(2,0,3,1,4)
            q,k,v=conv(q,self.short_q),conv(k,self.short_k),conv(v,self.short_v)
            beta=torch.sigmoid(self.b(h)).transpose(1,2)
            memory=delta_chunked(q,k,v,beta)
            gate=torch.sigmoid(self.mix_proj(h)+self.mix_bias).transpose(1,2).unsqueeze(-1)
            out=gate*memory+(1-gate)*v
            return out.transpose(1,2).reshape(B,T,D)
    return BAMG,Model
