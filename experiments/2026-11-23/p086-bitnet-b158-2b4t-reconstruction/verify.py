"""Verify six exact P086 master/packed tensor-only checkpoints on CPU."""
import argparse, hashlib,json
from pathlib import Path
import torch
import train as t
from model import Net,Linear
def sha(b):return hashlib.sha256(b).hexdigest()
def get(path,h):
    assert path.is_file() and sha(path.read_bytes())==h
    return torch.load(path,map_location='cpu',weights_only=True)
def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path('.'));c=a.parse_args()
    torch.set_num_threads(1);d=c.root
    rows=json.loads((d/'evidence/FIT-FROZEN.json').read_text())
    assert {(x['arm'],x['seed']) for x in rows}=={(a,s) for a in t.ARMS for s in t.SEEDS}
    with torch.inference_mode():
        tok=(torch.arange(32)*19+9).remainder(256).view(1,-1)
        for r in rows:
            arm,s=r['arm'],r['seed'];isbit=arm=='BitNet-b1.58'
            x=get(d/'weights'/f'{arm}-{s}.pt',r['checkpoint_sha'])
            assert set(x)=={'net','optimizer_retained'} and x['optimizer_retained'] is False
            m=Net(isbit).eval();m.load_state_dict(x['net'],strict=True)
            o=m(tok);q=tok.clone();q[:,16:]=(q[:,16:]+1)%256
            assert o.isfinite().all() and torch.equal(o[:,:16],m(q)[:,:16])
            print(arm,s,'strict CPU reload and causal prefix PASS')
            if not isbit:continue
            z=get(d/'weights'/f'{arm}-{s}-inference.pt',r['inference_payload']['sha'])
            assert set(z)=={'shape_by_tensor','ternary_pack','inv_scale','non_quantized'}
            calc=Net(True).eval();merged={}; cpu_reconstructed={}
            for k,m0 in calc.named_modules():
                if not isinstance(m0,Linear):continue
                packed=z['ternary_pack'][k];assert packed.dtype==torch.uint8
                code=t.open2(packed).view_as(m0.weight)
                assert tuple(z['shape_by_tensor'][k])==tuple(m0.weight.shape)
                assert bool(torch.all((code>=-1)&(code<=1)))
                src=x['net'][k+'.weight'];inv=z['inv_scale'][k]
                assert torch.allclose(src.abs().mean().clamp_min(1.e-5).reciprocal(),inv,rtol=2e-6,atol=1e-9)
                s2=(src*inv).round().clamp(-1.,1.)
                assert torch.equal(code,s2)
                m0.frozen=code/inv
                cpu_reconstructed[k]=code/src.abs().mean().clamp_min(1.e-5).reciprocal()
                m.get_submodule(k).frozen=s2/inv
                merged[k+'.weight']=torch.zeros_like(m0.weight)
            assert set(merged).isdisjoint(z['non_quantized'])
            for k,v in z['non_quantized'].items():assert torch.equal(v,x['net'][k])
            calc.load_state_dict(dict(z['non_quantized'],**merged),strict=True)
            oo=calc(tok);assert torch.equal(oo,m(tok))
            print(arm,s,'CPU master-derived ternary codes exact with archived scales, full logits identical, max vs fresh CPU reductions',float((o-oo).abs().max()))
            for k,v in cpu_reconstructed.items():calc.get_submodule(k).frozen=v
            assert torch.equal(o,calc(tok))
            print(arm,s,'isolating fresh CPU scales restores bit-exact output')
if __name__=='__main__':main()
