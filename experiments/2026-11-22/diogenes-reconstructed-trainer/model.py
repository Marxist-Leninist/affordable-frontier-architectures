"""Independent Diogenes/Sable RLT forward, from the author's pinned public equations."""
import json,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

class RMS(nn.Module):
    def __init__(self,w):
        super().__init__();self.weight=nn.Parameter(torch.ones(w))
    def forward(self,x):
        return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6))*self.weight

class Loop(nn.Module):
    def __init__(self,w,h,k,v=256,feedback=True,alpha=.1):
        super().__init__(); self.w=w;self.h=h;self.k=k;self.v=v;self.alpha=alpha;self.feedback=feedback
        self.embedding=nn.Embedding(v,w);self.initial=nn.Parameter(torch.zeros(w))
        for key in ('en','efn','pn','dn','cn','dfn','mn','outn'):
            self.add_module(key,RMS(w))
        for key in ('eq','ek','ev','eo','merge','dq','dk','dv','do','cq','co','mk','mv'):
            self.add_module(key,nn.Linear(w,w,bias=False))
        self.gate=nn.Linear(2*w,w)
        self.eff=nn.Sequential(nn.Linear(w,2*w),nn.GELU(),nn.Linear(2*w,w))
        self.dff=nn.Sequential(nn.Linear(w,2*w),nn.GELU(),nn.Linear(2*w,w))
        self.read=nn.Linear(w,v)
    def heads(self,x):
        return x.reshape(x.shape[0],-1,self.h,self.w//self.h).transpose(1,2)
    def apply_angle(self,x,pos):
        f=10000**(-torch.arange(0,x.shape[-1],2,dtype=x.dtype,device=x.device)/x.shape[-1])
        t=torch.arange(pos,pos+x.shape[2],dtype=x.dtype,device=x.device)[:,None]*f
        a,b=x[...,::2],x[...,1::2];c,s=t.cos(),t.sin()
        return torch.stack([a*c-b*s,a*s+b*c],dim=-1).flatten(-2)
    def project(self,x,key,p,rotate=True):
        y=self.heads(getattr(self,key)(x))
        return self.apply_angle(y,p) if rotate else y
    def attn(self,q,k,v,causal=False):
        x=F.scaled_dot_product_attention(q,k,v,is_causal=causal)
        return x.transpose(1,2).reshape(x.shape[0],x.shape[2],self.w)
    def forward(self,ids):
        em=self.embedding(ids);x=self.en(em)
        y=em+self.eo(self.attn(self.project(x,'eq',0),self.project(x,'ek',0),
                                    self.project(x,'ev',0,False),True))
        y=y+self.eff(self.efn(y));n=self.mn(y)
        mk,mv=self.project(n,'mk',0),self.project(n,'mv',0,False)
        prev=self.initial.expand(ids.shape[0],-1)
        dks=[];dvs=[];out=[]
        for t in range(ids.shape[1]):
            a=z=y[:,t]
            if self.feedback:
                r=self.pn(prev);z=a+self.alpha*self.gate(torch.cat((a,r),dim=-1)).sigmoid()*self.merge(r)
            z0=self.dn(z)
            q,kk,vv=self.project(z0,'dq',t),self.project(z0,'dk',t),self.project(z0,'dv',t,False)
            dks.append(kk);dvs.append(vv)
            z=z+self.do(self.attn(q,torch.cat(dks,2),torch.cat(dvs,2))[:,0])
            z=z+self.co(self.attn(self.project(self.cn(z),'cq',t),mk[:,:,:t+1],mv[:,:,:t+1])[:,0])
            prev=z+self.dff(self.dfn(z));out.append(self.read(self.outn(prev)))
            dks=dks[-(self.k-1):] if self.k>1 else []
            dvs=dvs[-(self.k-1):] if self.k>1 else []
        return torch.stack(out,1)

@torch.no_grad()
def functional_step_eval(sd,ids,cfg):
    # Independently construct the published one-byte step path, explicit attention dot and softmax, no shared code above.
    w,h,window=cfg['hidden_size'],cfg['num_attention_heads'],cfg['decoder_window'];d=w//h
    def lx(k,x):
        b=sd.get(k+'.bias')
        return F.linear(x,sd[k+'.weight'],b)
    def normalize(k,x):
        return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6))*sd[k+'.weight']
    def head(x):
        return x.reshape(ids.shape[0],h,1,d)
    def proj(k,x,j=None):
        z=head(lx(k,x))
        if j is None:return z
        a=j*10000**(-torch.arange(0,d,2,dtype=x.dtype,device=x.device)/d)
        x0,x1=z[...,::2],z[...,1::2]
        return torch.stack((x0*a.cos()-x1*a.sin(),x0*a.sin()+x1*a.cos()),dim=-1).flatten(-2)
    def attend(q,ks,vs):
        # CPU float reference only no dropout, masks arise by actual prefix lists.
        k=torch.cat(ks,dim=2);v=torch.cat(vs,dim=2)
        z=(q@k.transpose(-2,-1))/math.sqrt(d)
        return (z.softmax(dim=-1)@v).reshape(ids.shape[0],w)
    def ff(k,z): return lx(k+'.2',F.gelu(lx(k+'.0',z)))
    ek,ev,dk,dv,mk,mv,out=[],[],[],[],[],[],[]
    prev=sd['initial'].expand(ids.shape[0],-1)
    for t in range(ids.shape[1]):
        em=sd['embedding.weight'][ids[:,t]];z=normalize('en',em)
        ek.append(proj('ek',z,t));ev.append(proj('ev',z))
        e=em+lx('eo',attend(proj('eq',z,t),ek,ev))
        e+=ff('eff',normalize('efn',e))
        zn=normalize('mn',e);mk.append(proj('mk',zn,t));mv.append(proj('mv',zn))
        rp=normalize('pn',prev);z=e+cfg['feedback_alpha']*lx('gate',torch.cat((e,rp),dim=-1)).sigmoid()*lx('merge',rp)
        yt=normalize('dn',z);dk.append(proj('dk',yt,t));dv.append(proj('dv',yt))
        z+=lx('do',attend(proj('dq',yt,t),dk[-window:],dv[-window:]))
        z+=lx('co',attend(proj('cq',normalize('cn',z),t),mk,mv))
        prev=z+ff('dff',normalize('dfn',z));out.append(lx('read',normalize('outn',prev)))
    return torch.stack(out,1)

def load(config,weights):
    cfg=json.loads(Path(config).read_text())
    assert cfg['model_type']=='sable-rlt' and cfg['hidden_size']==928 and cfg['num_attention_heads']==8
    assert cfg['decoder_window']==4 and cfg['feedback'] is True and cfg['vocab_size']==256
    m=Loop(cfg['hidden_size'],cfg['num_attention_heads'],cfg['decoder_window'],cfg['vocab_size'],cfg['feedback'],cfg['feedback_alpha'])
    state=load_file(str(weights),device='cpu')
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
    vals=m.load_state_dict(state,strict=True)
    assert not vals.missing_keys and not vals.unexpected_keys
    assert sum(p.numel() for p in m.parameters())==cfg['parameter_count']==20297472
    return m,cfg
