"""ASI-Arch #641 HeadWiseGateNet: full-width AGILLM hybrid adapter.
Only mix() is replaced: host binary qkv/o and block-wide RMS are retained.
Source default four-path head-wise gate, FIR initialisation and plain delta rule.
No cache, packed-sequence API, extra loss, step counter or LR/LRK reduction.
See README.md for precise dtype, initialisation and normalisation contracts.
"""
import torch
from torch import nn
from torch.nn import functional as F

META = dict(id="hwg641", title="HeadWiseGateNet #641 full native-init hybrid", slot="hybrid",
            one_line="Full head-wise softmax over short FIR, long FIR, delta and value; 3/4 mixer, 1/4 attention",
            cohort="full-native-init", source_index=641)
SOURCE_PROGRAM_SHA256 = "c970077bad63f5829568cbcbe1019f91013416b2f10d634e363e421bc4077691"
CHUNK = 32

def l2norm(x):
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)

def conv(x, weight, activation=False):
    """B,H,T,hd -> B,H,T,hd; independent causal filter for every channel."""
    B,H,T,hd = x.shape
    y = x.permute(0,1,3,2).reshape(B,H*hd,T)
    y = F.conv1d(F.pad(y,(weight.shape[-1]-1,0)),weight,groups=H*hd)
    y = y.reshape(B,H,hd,T).permute(0,1,3,2)
    return F.silu(y) if activation else y

def delta_chunked(q, k, v, beta, C=CHUNK):
    """Source delta rule, equivalent triangular solve, no retention factor.
    Work in input dtype for fp32/fp64. Half/bfloat16 use fp32 work and cast
    the final result back; the source's forced BF16 inverse is not reproduced.
    """
    if C <= 0 or q.shape[2] == 0:
        raise ValueError("positive chunk size and nonempty sequence required")
    dtype = v.dtype
    if q.dtype in (torch.float16,torch.bfloat16):
        q,k,v,beta = (x.float() for x in (q,k,v,beta))
    q,k = l2norm(q),l2norm(k)
    B,H,T,dk = q.shape
    dv = v.shape[-1]
    pad = (-T) % C
    if pad:
        q,k,v = (F.pad(x,(0,0,0,pad)) for x in (q,k,v))
        beta = F.pad(beta,(0,pad))
    N=(T+pad)//C
    q,k = (x.reshape(B,H,N,C,dk) for x in (q,k))
    v=v.reshape(B,H,N,C,dv)
    b=beta.reshape(B,H,N,C,1)
    eye=torch.eye(C,device=q.device,dtype=q.dtype)
    strict=torch.tril(torch.ones(C,C,device=q.device,dtype=torch.bool),-1)
    inclusive=torch.tril(torch.ones(C,C,device=q.device,dtype=torch.bool))
    lower=(b*(k@k.transpose(-1,-2))).masked_fill(~strict,0)+eye
    inv=torch.linalg.solve_triangular(lower,eye.expand_as(lower),upper=False,unitriangular=True)
    W,U0=inv@(b*k),inv@(b*v)
    A=(q@k.transpose(-1,-2)).masked_fill(~inclusive,0)
    state=q.new_zeros(B,H,dk,dv)
    outputs=[]
    for n in range(N):
        U=U0[:,:,n]-W[:,:,n]@state
        outputs.append(q[:,:,n]@state+A[:,:,n]@U)
        state=state+k[:,:,n].transpose(-1,-2)@U
    return torch.cat(outputs,2)[:,:,:T].to(dtype)

def extra_parameter_count(d,H):
    """Per mixer, in addition to unchanged host block. Gate alone: 4Hd+12d+4H."""
    return 5*H*d + 58*d + 4*H

def patch(Block, Model, cfg):
    class HWG(Block):
        def __init__(self,c,index):
            super().__init__(c,index)
            self.attn_layer=bool(index%4==3 or index==c["L"]-1)
            if self.attn_layer:
                return
            D,H=c["d"],self.H
            if D%H:
                raise ValueError("d must be divisible by H")
            # Match pinned native FLA 0.2.2 nn.Conv1d initialization, NOT
            # the Dirac short-conv convention used by the older 458/879 ports.
            def short():
                return nn.Conv1d(D,D,4,groups=D,bias=False).weight
            self.short_q,self.short_k,self.short_v=short(),short(),short()
            self.b=nn.Linear(D,H,bias=False)
            def fir(taps):
                w=torch.zeros(H,D//H,taps)
                w[:,:,-1]=1
                w=w+torch.randn_like(w)*0.02
                return nn.Parameter(w)
            self.fs,self.fl=fir(3),fir(31)
            self.fusion_weight=nn.Parameter(torch.zeros(H,D+3*(D//H),4))
            bias=torch.zeros(H,4);bias[:,3]=2.0
            self.fusion_bias=nn.Parameter(bias)
        def mix(self,h,causal):
            if self.attn_layer:
                return super().mix(h,causal)
            B,T,D=h.shape
            if T==0:
                raise ValueError("nonempty sequence required")
            H,hd=self.H,D//self.H
            q,k,v=self.qkv(h).view(B,T,3,H,hd).permute(2,0,3,1,4)
            q,k,v=conv(q,self.short_q,True),conv(k,self.short_k,True),conv(v,self.short_v,True)
            beta=self.b(h).sigmoid().transpose(1,2)
            delta=delta_chunked(q,k,v,beta).transpose(1,2)
            short=conv(v,self.fs.reshape(D,1,3)).transpose(1,2)
            long=conv(v,self.fl.reshape(D,1,31)).transpose(1,2)
            direct=v.transpose(1,2)
            features=torch.cat((h.unsqueeze(2).expand(-1,-1,H,-1),short,long,delta),-1)
            logits=torch.einsum("bthf,hfc->bthc",features,self.fusion_weight)+self.fusion_bias
            weights=logits.softmax(-1)
            out=(weights[...,0:1]*short + weights[...,1:2]*long +
                 weights[...,2:3]*delta + weights[...,3:4]*direct)
            return out.reshape(B,T,D)
    return HWG,Model
