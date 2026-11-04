"""Last-output-only prefill, with optional exact chunking through existing model cache.
Both monolithic and chunked paths omit unused vocabulary projections equally.
"""
import torch

def prefill(model,tokens,chunk_size=None,output_tail=1):
    n=tokens.shape[1]
    if n<1 or not isinstance(output_tail,int) or not 1<=output_tail<=n:raise ValueError('Invalid output tail')
    if chunk_size is not None and (not isinstance(chunk_size,int) or chunk_size<1):raise ValueError('Invalid chunk')
    chunk=n if chunk_size is None else chunk_size
    states=None;tail=[];needed=n-output_tail
    for start in range(0,n,chunk):
        end=min(n,start+chunk);hidden=model.emb(tokens[:,start:end]);new=[]
        for i,block in enumerate(model.blocks):
            hidden,state=block(hidden,None if states is None else states[i],cache=True);new.append(state)
        states=new
        if end>needed:tail.append(hidden[:,max(0,needed-start):])
    features=tail[0] if len(tail)==1 else torch.cat(tail,dim=1)
    logits=model.head(model.norm(features))
    return logits,states
