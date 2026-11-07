"""Dense mechanism comparison; every treatment keeps native causal inference.

AGILLM-inspired local masked reconstruction is ported to Pythia layers, not the
ternary MoE runtime. The direct branch uses the tested v2 clean-KV denoiser as an
auxiliary objective, not as the inference decoder.
"""
from direct import *
MASK=50303
class Treatment(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.base=freeze_maps(load_original())
        if kind.startswith('direct'):
            self.local=DirectBlocks(self.base)
        elif kind.startswith('masked'):
            self.conditions=nn.ModuleList([nn.Linear(1,2*D) for _ in range(3)])
            self.norms=nn.ModuleList([copy.deepcopy(self.base.gpt_neox.final_layer_norm) for _ in range(3)])
            for c in self.conditions:nn.init.zeros_(c.weight);nn.init.zeros_(c.bias)
    def forward(self,x):return self.base(x,use_cache=False).logits
    def masked(self,inputs,selected,sigma,noise,group):
        x=self.base.gpt_neox.embed_in(inputs)
        x=math.sqrt(max(0.,1-sigma*sigma))*x+sigma*noise.to(x)
        scale,shift=self.conditions[group](x.new_tensor([[sigma]])).chunk(2,-1)
        x=x*(1+.1*scale.tanh())+shift
        t=x.shape[1];ids=torch.arange(t)[None,:];rot=self.base.gpt_neox.rotary_emb(x,ids)
        mask=x.new_zeros((1,1,t,t))
        for layer in self.base.gpt_neox.layers[group*2:group*2+2]:
            x=layer(x,attention_mask=mask,position_embeddings=rot,use_cache=False)[0]
        x=self.norms[group](x)
        return self.base.embed_out(x[selected])

def batch_objective(kind,step,seed,x,y):
    local=kind!='normal' and ('_local_' in kind or step%9==0)
    if not local:return {'local':False,'family':'ar','labels':y.reshape(-1),'selected_count':y.numel()}
    cursor=step-1 if '_local_' in kind else step//9-1
    group_count=1 if kind.endswith('_parent') else 3;group=cursor%group_count
    if kind.startswith('masked'):
        import numpy as np
        rng=np.random.default_rng(seed*100000+step)
        low=.05+.9*(1-(group+1)/group_count);high=.05+.9*(1-group/group_count)
        sigma=float(rng.uniform(low,high));chosen=rng.random(tuple(x.shape))<sigma
        if not chosen.any():chosen.flat[int(rng.integers(chosen.size))]=True
        selected=torch.from_numpy(chosen);inputs=x.clone();inputs[selected]=MASK
        noise=torch.from_numpy(rng.standard_normal((*x.shape,D),dtype=np.float32))
        return {'local':True,'family':'masked','group':group,'inputs':inputs,'selected':selected,'sigma':sigma,'noise':noise,'labels':x[selected],'selected_count':int(selected.sum())}
    g=torch.Generator().manual_seed(seed*100000+step)
    noise=torch.randn(*x.shape,D,generator=g);sigma=sample_sigma(group,len(x),g)
    return {'local':True,'family':'direct','group':group,'sigma':sigma,'noise':noise,'labels':y.reshape(-1),'selected_count':y.numel()}

def objective_logits(m,x,y,o):
    if not o['local']:return m(x).flatten(0,1)
    if o['family']=='masked':return m.masked(o['inputs'],o['selected'],o['sigma'],o['noise'],o['group'])
    return m.local.training_logits(x,y,o['noise'],o['sigma'],o['group']).flatten(0,1)
