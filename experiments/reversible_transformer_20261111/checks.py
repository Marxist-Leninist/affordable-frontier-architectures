import copy,json,math
from pathlib import Path
import torch
from models import Block,LanguageModel,InverseBackprop,stack_forward,stack_inverse,loss
P=Path(__file__).resolve().parent

def err(a,b):
    d=(a-b).detach();return {'max_abs':float(d.abs().max()),'relative_l2':float(d.norm()/a.detach().norm().clamp_min(1e-30))}

def gradients(xs,ys):
    return err(torch.cat([x.detach().reshape(-1) for x in xs]),torch.cat([y.detach().reshape(-1) for y in ys]))

def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out={'stack':[],'model':[]}
    for dtype in (torch.float32,torch.float64):
        for depth in (1,4,12):
            torch.manual_seed(998+depth);blocks=torch.nn.ModuleList([Block(16,4,32) for _ in range(depth)]).to(dtype)
            x1=torch.randn(2,8,16,dtype=dtype);x2=torch.randn_like(x1)
            with torch.no_grad():
                y=stack_forward(x1,x2,blocks);back=stack_inverse(*y,blocks)
                arbitrary=(torch.randn_like(x1),torch.randn_like(x2));forth=stack_forward(*stack_inverse(*arbitrary,blocks),blocks)
            a=x1.clone().requires_grad_();b=x2.clone().requires_grad_();r1,r2=stack_forward(a,b,blocks)
            upstream=(torch.randn_like(r1),torch.randn_like(r2))
            gp=torch.autograd.grad((r1,r2),(a,b,*blocks.parameters()),upstream)
            c=x1.clone().requires_grad_();d=x2.clone().requires_grad_();i1,i2=InverseBackprop.apply(c,d,blocks,*tuple(blocks.parameters()))
            gi=torch.autograd.grad((i1,i2),(c,d,*blocks.parameters()),upstream)
            rec=gradients((x1,x2),back);rt=gradients(arbitrary,forth);ge=gradients(gp,gi)
            tol=2e-5 if dtype==torch.float32 else 1e-11
            assert rec['max_abs']<tol and rt['max_abs']<tol,(dtype,depth,rec,rt)
            assert ge['relative_l2']<(2e-5 if dtype==torch.float32 else 1e-11),(dtype,depth,ge)
            assert all(torch.isfinite(v).all() for v in gi)
            out['stack'].append({'dtype':str(dtype),'depth':depth,'inverse_forward':rec,'forward_inverse':rt,'all_input_parameter_gradients':ge})
    torch.manual_seed(34);a=LanguageModel('reversible_graph',d=24,depth=4,heads=4,ff=48,context=16);b=copy.deepcopy(a);b.mode='reversible_inverse'
    oa=torch.optim.AdamW(a.parameters(),lr=.002,betas=(.9,.95));ob=torch.optim.AdamW(b.parameters(),lr=.002,betas=(.9,.95))
    for step in range(6):
        x=torch.randint(256,(4,16));y=torch.randint(256,(4,16));oa.zero_grad();ob.zero_grad();la=a(x);lb=b(x)
        loss(la,y).backward();loss(lb,y).backward()
        row={'step':step,'logits':err(la,lb),'all_parameter_gradients':gradients([p.grad for p in a.parameters()],[p.grad for p in b.parameters()])}
        assert row['logits']['max_abs']<1e-4 and row['all_parameter_gradients']['relative_l2']<2e-4,row
        out['model'].append(row);oa.step();ob.step()
    counts={}
    for mode in ('ordinary','reversible_graph','reversible_inverse'):
        torch.manual_seed(34);m=LanguageModel(mode);counts[mode]=sum(p.numel() for p in m.parameters())
        x=torch.randint(256,(2,64));z=x.clone();z[:,32:]=torch.randint(256,(2,32))
        with torch.no_grad():e=err(m(x)[:,:32],m(z)[:,:32])
        assert e['max_abs']==0,(mode,e)
        out.setdefault('causal_prefix',{})[mode]=e
    assert len(set(counts.values()))==1;out['parameter_counts']=counts;out['passed']=True
    (P/'correctness.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
