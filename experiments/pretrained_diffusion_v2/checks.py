import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import json,copy
import torch
from models import *
torch.set_num_threads(1);torch.manual_seed(708)
def main():
    original=load_original();x=torch.randint(10,1000,(2,8));z=torch.randn(2,8,D);s=torch.ones(2,1,1)
    with torch.no_grad():reference=original(x,use_cache=False).logits
    full=DirectBlocks(original,full_depth=True)
    with torch.no_grad():
        same=full.denoise(x,z,s,0);identity=float((reference-same).abs().max())
        assert torch.allclose(reference,same,rtol=2e-5,atol=2e-5)
        assert torch.equal(same,full.denoise(x,z*7,s*.02,0))
    del full,same
    model=DirectBlocks(original)
    # Activate the noise path so a zero adapter cannot make leakage tests vacuous.
    with torch.no_grad():
        for a in model.noise_adapters:torch.nn.init.normal_(a.weight,std=.02)
        for b in range(3):
            before=model.denoise(x,z,s,b)
            xx=x.clone();xx[:,5:]+=5
            assert torch.allclose(before[:,:5],model.denoise(xx,z,s,b)[:,:5],rtol=2e-5,atol=2e-5)
            zz=z.clone();zz[:,[0,1,2,3,5,6,7]]+=2
            assert torch.allclose(before[:,4],model.denoise(x,zz,s,b)[:,4],rtol=2e-5,atol=2e-5)
            zz=z.clone();zz[:,4]+=2
            assert float((before[:,4]-model.denoise(x,zz,s,b)[:,4]).abs().max())>1e-4
    counts=[0,0,0];handles=[]
    def hook(b):
        def h(*args):counts[b]+=1
        return h
    for b,group in enumerate(model.groups):
        for layer in group:handles.append(layer.attention.query_key_value.register_forward_hook(hook(b)))
    y=torch.randint(10,1000,(2,8));model.zero_grad(set_to_none=True)
    out=model.training_logits(x,y,z,s,1);F.cross_entropy(out.flatten(0,1),y.flatten()).backward()
    assert counts==[0,4,0],counts
    assert all(p.grad is None for p in model.active_parameters(0)+model.active_parameters(2))
    assert model.embedding.weight.grad is None and model.head.weight.grad is None
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.groups[1].parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for h in handles:h.remove()
    ll=torch.randn(2,8,100);q,idx=ll.softmax(-1).topk(32,-1);k=coarse_kl(ll,idx,q);assert float(k.abs().max())<1e-5
    qs,ss=bands();assert torch.all(torch.diff(ss)<0)
    assert torch.allclose(torch.diff(qs),torch.diff(qs)[0].expand(3))
    for b in range(3):
        sampled=sample_sigma(b,100,torch.Generator().manual_seed(b));assert sampled.min()>=ss[b+1] and sampled.max()<=ss[b]
    with torch.no_grad():
        contexts=model.prepare_context(x)
        cached_diffs=[]
        for b in range(3):
            for sigma in [.002,.1,80.]:
                ss=s*sigma;d=float((model.denoise(x,z,ss,b)-model.denoise_cached(contexts,z,ss,b)).abs().max())
                cached_diffs.append(d);assert d<1e-5
        for dec in ['high_noise','normalized_rollout','raw_rollout']:assert torch.isfinite(model.predict(x,z,dec)).all()
    result={'passed':True,'full_depth_neutral_max_abs':identity,'cached_uncached_max_abs':max(cached_diffs),'selected_group_qkv_forward_counts':counts,'trainable_parameters':count(model),'checks':['Full-depth neutral adapters preserve pretrained logits','Neutral output independent of noise and sigma','Future-clean exclusion in every block with active noise path','Other noisy-target exclusion in every block','Own-noise sensitivity','Selected block executes without preceding or later block forwards','Gradient isolation and frozen lexical maps','Finite gradients and all decoders','Coarsened teacher KL identity','Equal-probability bands and bounded noise samples','Fixed-prefix clean KV reuse matches uncached denoisers for all blocks at three noise levels'],'torch':torch.__version__}
    (P/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
