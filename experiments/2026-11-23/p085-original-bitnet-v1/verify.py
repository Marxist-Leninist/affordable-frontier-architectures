"""Strict CPU test of exact master and packed inference checkpoints."""
import argparse, hashlib,json
from pathlib import Path
import torch
import train as t
import run as p
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
            arm,s=r['arm'],r['seed'];isbit=arm=='BitNet-v1'
            x=get(d/'weights'/f'{arm}-{s}.pt',r['checkpoint_sha'])
            assert set(x)=={'net','optimizer_retained'} and x['optimizer_retained'] is False
            m=t.Net(isbit).eval(); m.load_state_dict(x['net'],strict=True)
            o=m(tok); q=tok.clone(); q[:,16:]=(q[:,16:]+1)%256
            assert o.isfinite().all() and torch.equal(o[:,:16],m(q)[:,:16])
            print(arm,s,'CPU strict checkpoint reload and causal prefix PASS')
            if not isbit:continue
            z=get(d/'weights'/f'{arm}-{s}-inference.pt',r['inference_payload']['sha'])
            assert set(z)=={'shape_by_tensor','sign_pack','beta','non_binary'}
            calc=t.Net(True).eval()
            merged={}
            for k,m0 in calc.named_modules():
                if not isinstance(m0,t.Proj):continue
                sgn=p.bits_open(z['sign_pack'][k]).float()*2-1
                assert tuple(z['shape_by_tensor'][k])==tuple(m0.weight.shape)
                sgn=sgn.view_as(m0.weight)
                src=x['net'][k+'.weight'];_,s2,b=t.effective(src)
                assert torch.equal(sgn,s2) and torch.allclose(b,z['beta'][k],rtol=2e-6,atol=1e-9)
                m0.exported=sgn*z['beta'][k]
                master_projection=m.get_submodule(k)
                master_projection.exported=s2*z['beta'][k]
                # Runtime override for binary projection; the generated loader
                # fills unused FP32 placeholders locally, not from the file.
                merged[k+'.weight']=torch.zeros_like(m0.weight)
            assert set(merged).isdisjoint(z['non_binary'])
            for k,v in z['non_binary'].items():assert torch.equal(v,x['net'][k])
            calc.load_state_dict(dict(z['non_binary'],**merged),strict=True)
            oo=calc(tok)
            err=float((o-oo).abs().max())
            assert torch.equal(m(tok),oo)
            print(arm,s,'CPU signs exact, identical full logits using the same archived betas; max abs vs freshly recomputed CPU betas:',err)
if __name__=='__main__':main()
