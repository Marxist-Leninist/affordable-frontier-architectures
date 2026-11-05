"""Fixed-span differentiable attention intervention. v19 continues training its weights."""
import copy
import torch
from torch.nn import functional as F
from models import Attention

class LocalAttention(Attention):
    def __init__(self,d=64,heads=4,window=128):
        super().__init__(d,heads)
        if not isinstance(window,int) or window<1:raise ValueError('window must be positive integer')
        self.window=window
    def forward(self,x,state=None,cache=False):
        b,t,d=x.shape
        q,k,v=self.qkv(x).chunk(3,-1)
        q,k,v=[a.reshape(b,t,self.h,d//self.h).transpose(1,2) for a in (q,k,v)]
        offset=0 if state is None else int(state[2].item())
        q,k=self.rope(q,offset),self.rope(k,offset)
        old=0 if state is None else state[0].shape[-2]
        if state is not None:k=torch.cat((state[0],k),-2);v=torch.cat((state[1],v),-2)
        qpos=torch.arange(offset,offset+t,device=x.device)[:,None]
        kpos=torch.arange(offset-old,offset+t,device=x.device)[None,:]
        mask=(kpos<=qpos)&(kpos>qpos-self.window)
        z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        new=None
        if cache:
            keep=min(self.window-1,k.shape[-2])
            # Clone even when slicing could be a view; retain no hidden long-prefix storage.
            if keep:knew=k[:,:,-keep:,:].clone(memory_format=torch.contiguous_format);vnew=v[:,:,-keep:,:].clone(memory_format=torch.contiguous_format)
            else:knew=k[:,:,:0,:].clone();vnew=v[:,:,:0,:].clone()
            new=(knew,vnew,torch.tensor(offset+t,dtype=torch.int64,device=x.device))
        return self.out(z.transpose(1,2).reshape(b,t,d)),new

def localize(model,window=128):
    out=copy.deepcopy(model)
    if out.kind!='transformer':raise ValueError('Only Transformer has attention to localize')
    for block in out.blocks:
        prior=block.mix;replacement=LocalAttention(prior.d,prior.h,window)
        replacement.load_state_dict(prior.state_dict());block.mix=replacement
    return out

def tensor_bytes(state):
    if state is None:return 0
    if isinstance(state,torch.Tensor):return state.numel()*state.element_size()
    return sum(tensor_bytes(x) for x in state)

def allocated_tensor_bytes(state):
    seen=set()
    def walk(s):
        if s is None:return 0
        if isinstance(s,torch.Tensor):
            storage=s.untyped_storage();key=(storage.data_ptr(),storage.nbytes())
            if key in seen:return 0
            seen.add(key);return storage.nbytes()
        return sum(walk(x) for x in s)
    return walk(state)
