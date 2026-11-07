"""Supplemental optimizer-isolation and immutable-cache tests; no training study changes."""
import run
from run import *
def main():
    torch.manual_seed(617);m=DirectBlocks(load_original());x=torch.randint(10,1000,(2,8));y=torch.randint(10,1000,(2,8));z=torch.randn(2,8,D);s=torch.ones(2,1,1)
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    loss=F.cross_entropy(m.training_logits(x,y,z,s,1).flatten(0,1),y.flatten());loss.backward();opt.step()
    changed=[];unchanged=[]
    for name,param in m.named_parameters():
        changed_now=not torch.equal(param,before[name])
        active=any(name.startswith(f'{a}.1.') for a in ['groups','norms','context_adapters','noise_adapters','times'])
        if not active:assert not changed_now,name
        (changed if changed_now else unchanged).append(name)
    assert any(n.startswith('groups.1.') for n in changed)
    m.eval()
    with torch.inference_mode():
        ctx=m.prepare_context(x);cache_before={(b,i,j):a.clone() for b,c in ctx.items() for i,kv in enumerate(c[1]) for j,a in enumerate(kv)}
        for sigma in [.002,.1,80.]:
            for b in range(3):m.denoise_cached(ctx,z,s*sigma,b)
        assert all(torch.equal(ctx[b][1][i][j],v) for (b,i,j),v in cache_before.items())
        # Exercise the actual rollout with a constant denoiser: descending Euler
        # must contract the initial noise toward this fixed clean estimate.
        constant=torch.zeros(2,8,m.codebook.shape[0]);constant[...,37]=10.
        estimate=F.normalize(constant.softmax(-1)@m.codebook,dim=-1)
        seen=[];original=m.denoise_cached
        def oracle(context,state,sigma,b):
            seen.append(state.clone());return constant
        m.denoise_cached=oracle
        m.predict(x,z,'normalized_rollout',ctx)
        m.denoise_cached=original
        expected=estimate+(SMIN/SMAX)*(z*SMAX-estimate)
        error=float((seen[-1]-expected).abs().max())
        assert len(seen)==4 and error<2e-6,(len(seen),error)
    write('extra_checks.json',dict(passed=True,optimizer_changed_only_active_group=True,changed_parameter_count=len(changed),unchanged_parameter_count=len(unchanged),clean_cache_unchanged_across_noise_calls=True,descending_euler_constant_oracle_max_abs=error))
    print('Optimizer isolation and cache immutability passed',flush=True)
if __name__=='__main__':main()
