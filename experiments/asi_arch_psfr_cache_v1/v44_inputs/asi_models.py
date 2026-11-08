"""Tiny full-stack architecture proxies. No cached inference enabled."""
import torch
from torch import nn
from cpu_adapter import load
from prior_models import AlternatingLM
MODS={n:load(n) for n in ('psfr','csm')}
class SourceMixer(nn.Module):
    def __init__(self,name):
        super().__init__();self.layer=MODS[name].DeltaNet(hidden_size=64,num_heads=2)
    def forward(self,x,state=None,cache=False):
        assert state is None and not cache,'Original ASI cache fails equivalence; full recomputation only'
        y,aux,_=self.layer(x)
        # The original API uses the attention-output slot for auxiliary entropy.
        # Standard LM block ignores that slot; do not silently reverse its sign.
        return y,None

def build(seed,arm,target=190848):
    assert arm in ('transformer','psfr','csm')
    torch.manual_seed(seed);m=AlternatingLM()
    # Preserve common embeddings, norms, output head and shared FFN submatrices.
    if arm!='transformer':
        torch.manual_seed(44000+seed)
        for b in m.blocks:b.mix=SourceMixer(arm)
    nonff=sum(p.numel() for n,p in m.named_parameters() if not any(s in n for s in ('.gate.weight','.up.weight','.down.weight')))
    # FFN unit contributes 3*64 params. Nearest total across three layers.
    totalinner=round((target-nonff)/192);inners=[totalinner//3+(i<totalinner%3) for i in range(3)]
    for b,inner in zip(m.blocks,inners):
        old=(b.gate,b.up,b.down);new=(nn.Linear(64,inner,bias=False),nn.Linear(64,inner,bias=False),nn.Linear(inner,64,bias=False))
        with torch.no_grad():
            for a,c in zip(old[:2],new[:2]):c.weight[:min(inner,a.out_features)].copy_(a.weight[:min(inner,a.out_features)])
            new[2].weight[:,:min(inner,old[2].in_features)].copy_(old[2].weight[:,:min(inner,old[2].in_features)])
        b.gate,b.up,b.down=new
    return m
if __name__=='__main__':
    for arm in ('transformer','psfr','csm'):
        m=build(1,arm);print(arm,sum(p.numel() for p in m.parameters()),[b.gate.out_features for b in m.blocks])
