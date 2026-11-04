"""Prune only structurally unreachable prefix; preserve absolute RoPE coordinates."""
import torch
from prefill import prefill

def local_prefill(model,tokens,output_tail=1,prune=False):
    states=None
    if prune:
        required=output_tail+sum(b.mix.window-1 for b in model.blocks)
        skipped=max(0,tokens.shape[1]-required)
        if skipped:
            states=[]
            for b in model.blocks:
                a=b.mix
                empty=torch.empty((tokens.shape[0],a.h,0,a.d//a.h),dtype=model.emb.weight.dtype,device=tokens.device)
                states.append((empty,empty.clone(),torch.tensor(skipped,dtype=torch.int64,device=tokens.device)))
            tokens=tokens[:,-required:]
    return prefill(model,tokens,128,output_tail,initial_states=states)
