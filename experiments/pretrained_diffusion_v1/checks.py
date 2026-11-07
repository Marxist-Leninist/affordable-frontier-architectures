import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import json,hashlib,copy
import torch
from models import *
torch.set_num_threads(1);torch.manual_seed(91)
def main():
    m=load_original();x=torch.randint(5,1000,(2,8));a=m(x).logits
    h=m.gpt_neox.embed_in(x);posids=torch.arange(8)[None,:];rot=m.gpt_neox.rotary_emb(h,posids)
    mask=torch.zeros(8,8).masked_fill(~torch.ones(8,8,dtype=torch.bool).tril(),float('-inf'))[None,None]
    for layer in m.gpt_neox.layers:h=layer(h,attention_mask=mask,position_ids=posids,position_embeddings=rot,use_cache=False)[0]
    a2=m.embed_out(m.gpt_neox.final_layer_norm(h));assert torch.allclose(a,a2,atol=2e-5,rtol=2e-5)
    del a,a2,h
    q=DiffusionBlocks(m);z=torch.randn(2,8,D);s=torch.ones(2,1,1)
    with torch.no_grad():
        for b in range(BLOCKS):
            base=q.denoise(x,z,s,b)
            xx=x.clone();xx[:,5:]=(xx[:,5:]+31)%1000
            assert torch.allclose(base[:,:5],q.denoise(xx,z,s,b)[:,:5],atol=2e-5,rtol=2e-5),'future clean leakage'
            zz=z.clone();zz[:,[0,1,2,3,5,6,7]]+=3
            assert torch.allclose(base[:,4],q.denoise(x,zz,s,b)[:,4],atol=2e-5,rtol=2e-5),'noisy token leakage'
    q.zero_grad(set_to_none=True);yy=torch.randint(5,1000,(2,8));loss=F.cross_entropy(q.training_logits(x,yy,z,s,1).flatten(0,1),yy.flatten());loss.backward()
    assert all(p.grad is None for p in q.block_parameters(0)+q.block_parameters(2))
    assert all(p.grad is None for p in q.embedding.parameters())
    assert all(p.grad is None for p in q.head.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in q.groups[1].parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in q.parameters())
    before={n:p.detach().clone() for n,p in q.named_parameters() if n.startswith(('groups.0.','groups.2.'))}
    opt=torch.optim.AdamW(q.block_parameters(1),lr=.0002);opt.step()
    for n,p in q.named_parameters():
        if n in before:assert torch.equal(before[n],p)
    qs,ss=probability_bounds();assert bool(torch.all(torch.diff(ss)<0));assert torch.allclose(torch.diff(qs),torch.diff(qs)[0].expand(3))
    for b in range(3):
        sampled=sample_sigma(b,100,torch.Generator().manual_seed(b))
        assert float(sampled.min())>=float(ss[b+1]) and float(sampled.max())<=float(ss[b])
    est=torch.tensor([2.]);state=torch.tensor([80.]);nextstate=state+(0.-80.)/80.*(state-est);assert torch.equal(nextstate,est)
    with torch.no_grad():out=q.predict(x,z);assert torch.isfinite(out).all()
    result={'passed':True,'checks':['original manual-forward equivalence','no future clean leakage for all blocks','no other noisy target leakage for all blocks','gradient isolation','optimizer isolation','frozen embedding and head','finite gradients and inference','equal probability descending ranges','sigma sampling ranges','negative-dt oracle endpoint'],'torch':torch.__version__,'trainable_diffusion':trainable_count(q)}
    (P/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
