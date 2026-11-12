"""ASI-Arch #712 AdaptiveGateNet-AFP, full practical-Dirac AGILLM hybrid.
Only mix is replaced; host binary qkv/o and block-wide RMS remain sole owners.
Source gate/floor/temperature/statistic/FIR math is retained. Short convolution
Dirac initialization, output norm/projections, dtype work and diagnostic state
are explicit adaptations. This is not a trained or cached-decoding result.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

META = dict(id="afp712_dirac", title="AdaptiveGateNet-AFP #712 full practical-Dirac hybrid", slot="hybrid",
            one_line="Full head-wise adaptive-floor, per-path-temperature four-branch fusion",
            cohort="full-practical-dirac", source_index=712)
SOURCE_PROGRAM_SHA256 = "62d59f5bf7f19ba502343f2c1aeba41311659f981660cd0feb37a58940451b7e"
CHUNK = 32
BASE_FLOOR = 0.05

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

def gate_parameter_count(d,H):
    """Gate weight, bias, per-head/path floor and per-path temperature."""
    return 4*H*d + 48*d + 8*H + 4

def extra_parameter_count(d,H):
    """All additions per mixer; host tensors are unchanged."""
    return 5*H*d + 129*d + 8*H + 4

def branch_stats(t):
    """B,T,H,hd. Source averages across HEADS, never across sequence."""
    return torch.cat((t.mean(-2,keepdim=True).expand_as(t),
                      t.square().mean(-2,keepdim=True).expand_as(t),
                      t.abs().mean(-2,keepdim=True).expand_as(t),
                      t.norm(dim=-1,keepdim=True).expand_as(t)),-1)

def patch(Block, Model, cfg):
    class AFP(Block):
        def __init__(self,c,index):
            super().__init__(c,index)
            self.attn_layer=bool(index%4==3 or index==c["L"]-1)
            if self.attn_layer:
                return
            D,H=c["d"],self.H
            if D%H:
                raise ValueError("d must be divisible by H")
            def short():
                w=torch.zeros(D,1,4);w[:,:,-1]=1
                return nn.Parameter(w)
            self.short_q,self.short_k,self.short_v=short(),short(),short()
            self.b=nn.Linear(D,H,bias=False)
            # Source FIR initialization is random N(0,.02^2), not Dirac.
            self.fs=nn.Parameter(torch.randn(H,D//H,5)*0.02)
            self.fl=nn.Parameter(torch.randn(H,D//H,64)*0.02)
            self.gate_weight=nn.Parameter(torch.empty(H,D+12*(D//H),4))
            nn.init.kaiming_uniform_(self.gate_weight,a=math.sqrt(5))
            bias=torch.zeros(H,4);bias[:,0:2]=-.5;bias[:,2]=.5;bias[:,3]=1.5
            self.gate_bias=nn.Parameter(bias)
            self.log_temp=nn.Parameter(torch.zeros(4))
            self.floor_param=nn.Parameter(torch.zeros(H,4))
        def _mix_and_weights(self,h):
            B,T,D=h.shape
            if T==0:
                raise ValueError("nonempty sequence required")
            H,hd=self.H,D//self.H
            q,k,v=self.qkv(h).view(B,T,3,H,hd).permute(2,0,3,1,4)
            q,k,v=conv(q,self.short_q,True),conv(k,self.short_k,True),conv(v,self.short_v,True)
            beta=self.b(h).sigmoid().transpose(1,2)
            delta=delta_chunked(q,k,v,beta).transpose(1,2)
            short=conv(v,self.fs.reshape(D,1,5)).transpose(1,2)
            long=conv(v,self.fl.reshape(D,1,64)).transpose(1,2)
            direct=v.transpose(1,2)
            features=torch.cat((h.unsqueeze(2).expand(-1,-1,H,-1),
                branch_stats(short),branch_stats(long),branch_stats(delta)),-1)
            logits=torch.einsum("bthf,hfc->bthc",features,self.gate_weight)+self.gate_bias
            temp=torch.exp(self.log_temp).clamp(min=.1,max=10.)
            soft=(logits/temp).softmax(-1)
            floor=(BASE_FLOOR*self.floor_param.sigmoid())[None,None,:,:]
            weights=floor+(1.-floor.sum(-1,keepdim=True))*soft
            out=(weights[...,0:1]*short + weights[...,1:2]*long +
                 weights[...,2:3]*delta + weights[...,3:4]*direct)
            return out.reshape(B,T,D),weights
        def mix(self,h,causal):
            if self.attn_layer:
                return super().mix(h,causal)
            # Mixer stays causal even for masked/NAT host objectives.
            return self._mix_and_weights(h)[0]
        def mix_with_diagnostics(self,h,causal=True):
            """Explicit detached CPU gate diagnostics; no hidden retained state.
            Unlike source last_fusion_weights, diagnostics cannot train an
            external auxiliary loss. Such an integration is not supported here.
            """
            if self.attn_layer:
                return super().mix(h,causal),None
            out,weights=self._mix_and_weights(h)
            return out,weights.detach().cpu().clone()
    return AFP,Model
