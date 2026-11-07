"""Direct-input causal conversion with neutral adapters and optional distillation.

The full-depth neutral variant preserves the pretrained function. Independent
two-layer partitions do not; that separate shock is explicitly tested. No prefix
network is needed to construct a selected block's clean/noisy inputs.
"""
import copy,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
# Process-local CPU configuration. Full-depth logits and gradient parity were
# checked before adopting it; original defaults had severe runtime variability.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.autograd.set_multithreading_enabled(False)
torch.set_flush_denormal(True)
from transformers import GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb,eager_attention_forward
P=Path(__file__).parent;D=128;BLOCKS=3;SMIN=.002;SMAX=80.;PMEAN=-1.2;PSTD=1.2
def load_original():
    return GPTNeoXForCausalLM.from_pretrained(P/'original',local_files_only=True,
        trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
def freeze_maps(m):
    m.gpt_neox.embed_in.requires_grad_(False);m.embed_out.requires_grad_(False);return m
def bands():
    n=torch.distributions.Normal(torch.tensor(PMEAN,dtype=torch.float64),torch.tensor(PSTD,dtype=torch.float64))
    qs=torch.linspace(float(n.cdf(torch.tensor(math.log(SMAX)))),float(n.cdf(torch.tensor(math.log(SMIN)))),4,dtype=torch.float64)
    ss=n.icdf(qs).exp();ss[0]=SMAX;ss[-1]=SMIN;return qs,ss
def sample_sigma(b,n,g):
    qs,_=bands();q=qs[b+1]+torch.rand(n,generator=g,dtype=torch.float64)*(qs[b]-qs[b+1])
    return (PMEAN+PSTD*torch.distributions.Normal(0.,1.).icdf(q)).exp().float().reshape(n,1,1)
def mask(t,device):
    # Query i predicts target token i+1. It reads clean keys j<=i, never any
    # noisy keys. Target-noise information remains in query residual/MLP paths.
    allowed=torch.zeros(2*t,2*t,dtype=torch.bool,device=device)
    allowed[:t,:t]=torch.ones(t,t,device=device,dtype=torch.bool).tril()
    allowed[t:,:t]=torch.ones(t,t,device=device,dtype=torch.bool).tril()
    return torch.zeros(2*t,2*t,device=device).masked_fill(~allowed,float('-inf'))[None,None]
def paired_layer(layer,clean,query,rot,attn_mask):
    """Two query streams share CLEAN K/V. No contaminated noisy prefix cache.

    Keep every GEMM shape and operation order identical to native GPT-NeoX for
    each stream, avoiding a full-depth identity test obscured by 2T GEMM rounding.
    """
    attention=layer.attention;bs,t,hidden=clean.shape;heads=hidden//attention.head_size
    cn=layer.input_layernorm(clean);qn=layer.input_layernorm(query)
    def qkv(h):
        v=attention.query_key_value(h).view(bs,t,heads,3*attention.head_size).transpose(1,2)
        return v.chunk(3,dim=-1)
    cq,ck,cv=qkv(cn);qq,qk,_=qkv(qn)
    cq,ck=apply_rotary_pos_emb(cq,ck,*rot);qq,_=apply_rotary_pos_emb(qq,qk,*rot)
    def attend(q):
        out,_=eager_attention_forward(attention,q,ck,cv,attn_mask,
            scaling=attention.scaling,dropout=attention.attention_dropout if layer.training else 0.)
        return layer.post_attention_dropout(attention.dense(out.reshape(bs,t,hidden)))
    ca=attend(cq);qa=attend(qq)
    assert layer.use_parallel_residual
    cm=layer.post_mlp_dropout(layer.mlp(layer.post_attention_layernorm(clean)))
    qm=layer.post_mlp_dropout(layer.mlp(layer.post_attention_layernorm(query)))
    return cm+ca+clean,qm+qa+query
def clean_layer_cache(layer,clean,rot,attn_mask):
    a=layer.attention;bs,t,d=clean.shape
    q,k,v=a.query_key_value(layer.input_layernorm(clean)).view(bs,t,d//a.head_size,3*a.head_size).transpose(1,2).chunk(3,-1)
    q,k=apply_rotary_pos_emb(q,k,*rot)
    out,_=eager_attention_forward(a,q,k,v,attn_mask,scaling=a.scaling,dropout=0.)
    attn=a.dense(out.reshape(bs,t,d));mlp=layer.mlp(layer.post_attention_layernorm(clean))
    return mlp+attn+clean,(k,v)
def noisy_layer_cached(layer,query,kv,rot,attn_mask):
    a=layer.attention;bs,t,d=query.shape
    q,k,_=a.query_key_value(layer.input_layernorm(query)).view(bs,t,d//a.head_size,3*a.head_size).transpose(1,2).chunk(3,-1)
    q,_=apply_rotary_pos_emb(q,k,*rot)
    out,_=eager_attention_forward(a,q,*kv,attn_mask,scaling=a.scaling,dropout=0.)
    attn=a.dense(out.reshape(bs,t,d));mlp=layer.mlp(layer.post_attention_layernorm(query))
    return mlp+attn+query
class DirectBlocks(nn.Module):
    def __init__(self,original,full_depth=False):
        super().__init__();self.embedding=original.gpt_neox.embed_in;self.head=original.embed_out
        self.embedding.requires_grad_(False);self.head.requires_grad_(False)
        self.groups=nn.ModuleList([nn.ModuleList(original.gpt_neox.layers)] if full_depth else
            [nn.ModuleList(original.gpt_neox.layers[b*2:(b+1)*2]) for b in range(3)])
        n=len(self.groups);self.norms=nn.ModuleList([copy.deepcopy(original.gpt_neox.final_layer_norm) for _ in range(n)])
        self.context_adapters=nn.ModuleList([nn.Linear(D,D) for _ in range(n)])
        self.noise_adapters=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(n)])
        self.times=nn.ModuleList([nn.Sequential(nn.Linear(16,D),nn.SiLU(),nn.Linear(D,D)) for _ in range(n)])
        for a in list(self.context_adapters)+list(self.noise_adapters):
            nn.init.zeros_(a.weight)
            if a.bias is not None:nn.init.zeros_(a.bias)
        for a in self.times:nn.init.zeros_(a[-1].weight);nn.init.zeros_(a[-1].bias)
        self.rotary=original.gpt_neox.rotary_emb
        self.register_buffer('codebook',F.normalize(self.embedding.weight.detach(),dim=-1))
        self.register_buffer('input_scale',self.embedding.weight.detach().norm(dim=-1).median())
    def active_parameters(self,b):
        return sum([list(a[b].parameters()) for a in [self.groups,self.norms,self.context_adapters,self.noise_adapters,self.times]],[])
    def denoise(self,x,z,s,b):
        bs,t=x.shape;clean=self.embedding(x);clean=clean+self.context_adapters[b](clean)
        freq=2.**torch.arange(8,device=z.device);a=.25*s.log().flatten(1)*freq[None,:]
        time=self.times[b](torch.cat([a.sin(),a.cos()],-1))[:,None,:]
        query=clean+self.noise_adapters[b](z*torch.rsqrt(1+D*s.square())*self.input_scale)+time
        ids=torch.arange(t,device=z.device)[None,:];rot=self.rotary(clean,ids)
        attn=torch.zeros(t,t,device=z.device).masked_fill(~torch.ones(t,t,dtype=torch.bool,device=z.device).tril(),float('-inf'))[None,None]
        for layer in self.groups[b]:clean,query=paired_layer(layer,clean,query,rot,attn)
        return self.head(self.norms[b](query))
    def training_logits(self,x,y,noise,s,b):return self.denoise(x,F.embedding(y,self.codebook)+s*noise,s,b)
    @torch.no_grad()
    def prepare_context(self,x,only_block=None):
        """Valid for this frozen prefix and model weights only. Never share
        distinct block caches. No sigma or noisy-target input enters this path.
        """
        out={};t=x.shape[1];ids=torch.arange(t,device=x.device)[None,:]
        attn=torch.zeros(t,t,device=x.device).masked_fill(~torch.ones(t,t,dtype=torch.bool,device=x.device).tril(),float('-inf'))[None,None]
        for b in (range(len(self.groups)) if only_block is None else [only_block]):
            initial=self.embedding(x);initial=initial+self.context_adapters[b](initial)
            clean=initial;rot=self.rotary(clean,ids);kvs=[]
            for layer in self.groups[b]:clean,kv=clean_layer_cache(layer,clean,rot,attn);kvs.append(kv)
            out[b]=(initial,kvs,rot,attn)
        return out
    def denoise_cached(self,context,z,s,b):
        clean,kvs,rot,attn=context[b];freq=2.**torch.arange(8,device=z.device)
        a=.25*s.log().flatten(1)*freq[None,:];time=self.times[b](torch.cat([a.sin(),a.cos()],-1))[:,None,:]
        query=clean+self.noise_adapters[b](z*torch.rsqrt(1+D*s.square())*self.input_scale)+time
        for layer,kv in zip(self.groups[b],kvs):query=noisy_layer_cached(layer,query,kv,rot,attn)
        return self.head(self.norms[b](query))
    def predict(self,x,noise,decoder='normalized_rollout',context=None):
        if context is None:context=self.prepare_context(x,0 if decoder=='high_noise' else None)
        z=noise*SMAX
        if decoder=='high_noise':return self.denoise_cached(context,z,z.new_full((len(x),1,1),SMAX),0)
        _,ss=bands()
        for b in range(3):
            s=z.new_full((len(x),1,1),float(ss[b]));logits=self.denoise_cached(context,z,s,b)
            estimate=logits.softmax(-1)@self.codebook
            if decoder=='normalized_rollout':estimate=F.normalize(estimate,dim=-1,eps=1e-8)
            z=z+(float(ss[b+1])-float(ss[b]))/float(ss[b])*(z-estimate)
        return self.denoise_cached(context,z,z.new_full((len(x),1,1),SMIN),2)
def count(m):return sum(p.numel() for p in m.parameters() if p.requires_grad)
def coarse_kl(logits,indices,q):
    """Exact KL on teacher-top32 categories plus one aggregated complement."""
    logp=logits.log_softmax(-1);lp=logp.gather(-1,indices.long())
    p_tail=(1-lp.exp().sum(-1)).clamp_min(1e-7)
    q_tail=(1-q.sum(-1)).clamp_min(1e-7)
    return (q*(q.clamp_min(1e-30).log()-lp)).sum(-1)+q_tail*(q_tail.log()-p_tail.log())
