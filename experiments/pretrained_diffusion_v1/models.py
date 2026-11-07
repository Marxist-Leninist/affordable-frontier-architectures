"""Independent, inspectable Pythia -> locally trained diffusion-block adapter.

Not official DiffusionBlocks LM code (none supplied by the paper). Frozen shared
embedding/output maps keep block updates independent. This is an intentionally
small feasibility conversion, not a reproduction or claim about large models.
"""
import copy, math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPTNeoXForCausalLM

P=Path(__file__).parent
BLOCKS=3; D=128; SMIN=.002; SMAX=80.; PMEAN=-1.2; PSTD=1.2

def load_original():
    return GPTNeoXForCausalLM.from_pretrained(P/'original', local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval()

def frozen_maps(m):
    m.gpt_neox.embed_in.requires_grad_(False);m.embed_out.requires_grad_(False)
    return m

def probability_bounds():
    n=torch.distributions.Normal(torch.tensor(PMEAN,dtype=torch.float64),torch.tensor(PSTD,dtype=torch.float64))
    qlo=n.cdf(torch.tensor(math.log(SMIN))); qhi=n.cdf(torch.tensor(math.log(SMAX)))
    qs=torch.linspace(float(qhi),float(qlo),BLOCKS+1,dtype=torch.float64)
    ss=n.icdf(qs).exp();ss[0]=SMAX;ss[-1]=SMIN
    return qs,ss

def sample_sigma(b,batch,generator):
    qs,_=probability_bounds();q=qs[b+1]+torch.rand(batch,generator=generator,dtype=torch.float64)*(qs[b]-qs[b+1])
    return (PMEAN+PSTD*torch.distributions.Normal(0.,1.).icdf(q)).exp().float().reshape(batch,1,1)

def clean_past_mask(t,device):
    # Clean x[i] is token i. Noisy z[i] predicts token i+1.
    # Noisy positions see clean j<=i and themselves, NEVER another noisy token.
    a=torch.zeros(2*t,2*t,device=device,dtype=torch.bool)
    a[:t,:t]=torch.ones(t,t,device=device,dtype=torch.bool).tril()
    a[t:,:t]=torch.ones(t,t,device=device,dtype=torch.bool).tril()
    a[t:,t:]=torch.eye(t,device=device,dtype=torch.bool)
    return torch.zeros(2*t,2*t,device=device).masked_fill(~a,float('-inf'))[None,None]

class DiffusionBlocks(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.embedding=original.gpt_neox.embed_in;self.head=original.embed_out
        self.embedding.requires_grad_(False);self.head.requires_grad_(False)
        self.groups=nn.ModuleList([nn.ModuleList(original.gpt_neox.layers[b*2:(b+1)*2]) for b in range(BLOCKS)])
        self.norms=nn.ModuleList([copy.deepcopy(original.gpt_neox.final_layer_norm) for _ in range(BLOCKS)])
        self.times=nn.ModuleList([nn.Sequential(nn.Linear(16,D),nn.SiLU(),nn.Linear(D,D)) for _ in range(BLOCKS)])
        for m in self.times:nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
        self.rotary=original.gpt_neox.rotary_emb
        self.register_buffer('codebook',F.normalize(self.embedding.weight.detach(),dim=-1))
        self.register_buffer('input_scale',self.embedding.weight.detach().norm(dim=-1).median())
    def block_parameters(self,b):
        return list(self.groups[b].parameters())+list(self.norms[b].parameters())+list(self.times[b].parameters())
    def denoise(self,x,z,s,b):
        bs,t=x.shape;freq=2.**torch.arange(8,device=z.device)
        a=.25*s.log().flatten(1)*freq[None,:]
        time=self.times[b](torch.cat((a.sin(),a.cos()),-1))[:,None,:]
        # Unit-L2 target codes have per-coordinate variance approx 1/D.
        # Input preconditioning preserves the original embedding's typical norm.
        h=torch.cat((self.embedding(x),(z*torch.rsqrt(1.+D*s.square()))*self.input_scale+time),dim=1)
        ids=torch.cat((torch.arange(t,device=z.device),torch.arange(1,t+1,device=z.device)))[None,:]
        pos=self.rotary(h,ids);mask=clean_past_mask(t,z.device)
        for layer in self.groups[b]:
            h=layer(h,attention_mask=mask,position_ids=ids,position_embeddings=pos,use_cache=False)[0]
        return self.head(self.norms[b](h[:,t:]))
    def training_logits(self,x,y,noise,s,b):
        return self.denoise(x,F.embedding(y,self.codebook)+s*noise,s,b)
    def predict(self,x,noise,steps_per_block=2):
        _,bounds=probability_bounds();z=noise*SMAX
        for b in range(BLOCKS):
            ss=torch.logspace(math.log10(float(bounds[b])),math.log10(float(bounds[b+1])),steps_per_block+1)
            for j in range(steps_per_block):
                s=z.new_full((x.shape[0],1,1),float(ss[j]))
                logits=self.denoise(x,z,s,b)
                estimate=logits.softmax(-1)@self.codebook
                # Descending sigma gives a NEGATIVE Euler dt.
                z=z+(float(ss[j+1])-float(ss[j]))/float(ss[j])*(z-estimate)
        return self.denoise(x,z,z.new_full((x.shape[0],1,1),SMIN),BLOCKS-1)

def trainable_count(m):return sum(p.numel() for p in m.parameters() if p.requires_grad)
