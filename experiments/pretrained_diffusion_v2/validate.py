"""Source, data separation, immutable weights, tensors, and score replay."""
import run
from run import *
from accounting import restore
def main():
    protocol=json.loads((P/'protocol.json').read_text())
    for f,h in protocol['source_sha256'].items():assert sha(P/f)==h
    for f,h in [('model_manifest.json',protocol['model_manifest_sha256']),('token_manifest.json',protocol['token_manifest_sha256']),('source_data_manifest.json',protocol['source_data_manifest_sha256']),('window_manifest.json',protocol['window_manifest_sha256'])]:assert sha(P/f)==h
    for f,v in json.loads((P/'model_manifest.json').read_text())['files'].items():assert sha(P/'original'/f)==v['sha256']
    window=json.loads((P/'window_manifest.json').read_text())
    for s,v in window['splits'].items():
        assert all(abs(a-b)>33 for a in v['starts'] for b in v['excluded_v1_starts'])
        assert all(abs(a-b)>33 for i,a in enumerate(v['starts']) for b in v['starts'][i+1:])
    ds=data();checked=[];replayed=[]
    for kind in ['normal_ce','normal_kd','direct_ce','direct_kd']:
        for seed in SEEDS:
            endpoints=['equal_data'] if kind.startswith('normal') else ['equal_data','equal_cpu']
            for endpoint in endpoints:
                stem=f'{kind}_{seed}_{endpoint}';r=json.loads((P/f'{stem}.json').read_text());f=P/'checkpoints'/f'{stem}.pt'
                assert sha(f)==r['checkpoint_sha256'];state=torch.load(f,weights_only=True)
                assert all(torch.isfinite(t).all() for t in state.values())
                model=restore(kind,seed,endpoint)
                if seed==11 and endpoint=='equal_data':
                    replay=evaluate(model,kind,ds['test'],'test');difference=max(abs(a-b) for a,b in zip(replay['window_nll_nats'],r['test']['window_nll_nats']))
                    assert difference==0.;replayed.append(dict(checkpoint=stem,max_window_difference=difference))
                checked.append(stem);del model,state
    for mode in ['normal','block']:
        a=json.loads((P/f'accounting_{mode}.json').read_text())
        assert a['loaded_subset_reproduces_full_source_logits']
        assert a['loaded_weight_files']==(['shared.safetensors','normal_body.safetensors'] if mode=='normal' else ['shared.safetensors','block_1.safetensors'])
        assert a['cold_component_logical_read_bytes']==sum(x['logical_bytes'] for x in a['initial_reads'])
    write('validation.json',dict(passed=True,checkpoint_count=len(checked),checked=checked,score_replays=replayed,original_weights_and_protocol_unchanged=True,new_eval_windows_exclude_all_v1_contexts=True,weight_read_trace_matches_selected_block=True,finite_trainable_tensors_and_exact_keys=True))
    print('PASSED',len(checked),'checkpoints',replayed,flush=True)
if __name__=='__main__':main()
