"""Integrity, finite tensors, frozen weights, and exact equal-data score replay."""
import run
from run import *
def restore(kind,seed,endpoint):
    torch.manual_seed(seed);orig=load_original();m=DiffusionBlocks(orig) if kind=='pretrained_diffusion' else frozen_maps(orig)
    f=P/'checkpoints'/f'{kind}_{seed}_{endpoint}.pt';s=torch.load(f,weights_only=True)
    expected={n for n,p in m.named_parameters() if p.requires_grad}
    assert set(s)==expected
    assert all(torch.isfinite(v).all() for v in s.values())
    old={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    r=m.load_state_dict(s,strict=False);assert not r.unexpected_keys
    assert all(torch.equal(p,old[n]) for n,p in m.named_parameters() if n in old)
    return m
def main():
    original=json.loads((P/'model_manifest.json').read_text())
    for f,v in original['files'].items():assert sha(P/'original'/f)==v['sha256']
    protocol=json.loads((P/'protocol.json').read_text())
    for f,h in protocol['files'].items():assert sha(P/f)==h
    ds=data();checked=[];replays=[]
    for kind in ['normal_finetune','pretrained_diffusion']:
        for seed in SEEDS:
            for endpoint in (['equal_data'] if kind=='normal_finetune' else ['equal_data','equal_cpu']):
                stem=f'{kind}_{seed}_{endpoint}';r=json.loads((P/f'{stem}.json').read_text())
                assert sha(P/'checkpoints'/f'{stem}.pt')==r['checkpoint_sha256']
                m=restore(kind,seed,endpoint)
                if seed==1 and endpoint=='equal_data':
                    e=evaluate(m,kind,ds['test'],64)
                    maxdiff=max(abs(a-b) for a,b in zip(e['window_nll_nats'],r['test']['window_nll_nats']))
                    assert maxdiff<1e-7;replays.append(dict(checkpoint=stem,max_window_difference=maxdiff))
                checked.append(stem);del m
    write('validation.json',dict(passed=True,checkpoint_count=len(checked),checkpoints=checked,original_files_unchanged=True,source_matches_frozen_protocol=True,frozen_maps_unchanged=True,all_trainable_tensor_keys_and_finiteness_verified=True,heldout_replays=replays))
    print('PASSED',len(checked),'checkpoints',replays,flush=True)
if __name__=='__main__':main()
