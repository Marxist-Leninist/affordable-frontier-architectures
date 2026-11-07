"""Pre-outcome structural and leakage tests for common causal inference."""
from experiment import *
def main():
    torch.manual_seed(190);x=torch.randint(10,1000,(2,8));y=torch.randint(10,1000,(2,8));results=[]
    with torch.no_grad():reference=load_original()(x,use_cache=False).logits
    for kind in KINDS:
        torch.manual_seed(912);m=Treatment(kind);m.eval()
        with torch.no_grad():
            err=float((m(x)-reference).abs().max());assert err==0.,(kind,err)
            xx=x.clone();xx[:,5:]+=3;assert torch.equal(m(x)[:,:5],m(xx)[:,:5])
        if kind=='normal':results.append({'kind':kind,'initial_native_logit_error':err});continue
        step=18 if 'hybrid' in kind else 2;o=batch_objective(kind,step,17,x,y);b=o['group'];counts=[0]*6;handles=[]
        def hook(i):
            def h(*args):counts[i]+=1
            return h
        for i,layer in enumerate(m.base.gpt_neox.layers):handles.append(layer.attention.query_key_value.register_forward_hook(hook(i)))
        m.zero_grad(set_to_none=True);loss=F.cross_entropy(objective_logits(m,x,y,o),o['labels']);loss.backward()
        expected=[(2 if kind.startswith('direct') else 1) if i//2==b else 0 for i in range(6)]
        assert counts==expected,(kind,counts,expected)
        assert all(p.grad is None for i,layer in enumerate(m.base.gpt_neox.layers) if i//2!=b for p in layer.parameters())
        assert m.base.gpt_neox.embed_in.weight.grad is None and m.base.embed_out.weight.grad is None
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        for h in handles:h.remove()
        before={n:p.detach().clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4);opt.step()
        for i,layer in enumerate(m.base.gpt_neox.layers):
            if i//2!=b:
                for n,p in layer.named_parameters():assert torch.equal(p,before[f'base.gpt_neox.layers.{i}.{n}'])
        if kind.startswith('masked'):
            selected=torch.zeros_like(x,dtype=torch.bool);selected[:,1]=True;inputs=x.clone();inputs[selected]=MASK;noise=torch.randn(2,8,D)
            with torch.no_grad():
                a=m.masked(inputs,selected,.2,noise,0);future=inputs.clone();future[:,7]+=12
                assert float((a-m.masked(future,selected,.2,noise,0)).abs().max())>1e-5
                changed=x.clone();changed[selected]+=29;changed[selected]=MASK
                assert torch.equal(a,m.masked(changed,selected,.2,noise,0))
        else:
            with torch.no_grad():
                for a in m.local.noise_adapters:nn.init.normal_(a.weight,std=.02)
                z=torch.randn(2,8,D);s=torch.ones(2,1,1);before=m.local.denoise(x,z,s,b)
                xx=x.clone();xx[:,5:]+=10;zz=z.clone();zz[:,[0,1,2,3,5,6,7]]+=1
                assert torch.equal(before[:,:5],m.local.denoise(xx,z,s,b)[:,:5])
                assert torch.equal(before[:,4],m.local.denoise(x,zz,s,b)[:,4])
        results.append({'kind':kind,'initial_native_logit_error':err,'selected_group':b,'qkv_forward_counts':counts,'gradient_optimizer_isolation':True})
        del m,before,opt
    assert MASK==50303 and all(not (a==MASK).any() for a in data().values())
    for kind in KINDS[1:]:
        objs=[batch_objective(kind,k,31,x,y) for k in range(1,649)]
        assert sum(o['local'] for o in objs)==(648 if '_local_' in kind else 72)
        if kind.endswith('_parent'):assert {o['group'] for o in objs if o['local']}=={0}
        else:assert {o['group'] for o in objs if o['local']}=={0,1,2}
    write('checks.json',{'passed':True,'results':results,'common_native_causal_inference':True,'masked_targets_absent_from_inputs':True,'masked_training_bidirectional_by_design':True,'direct_training_causal_clean_past_only':True,'reserved_mask_absent_in_all_data':True,'cadence_and_restricted_coverage_verified':True})
    print('PASS',results)
if __name__=='__main__':main()
