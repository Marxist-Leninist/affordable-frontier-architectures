from experiment import *
from accounting import restore

def main():
    p=json.loads((P/'protocol.json').read_text());ds=data();records=[];replays=[];cache_checks=[]
    for f,h in p['source_sha256'].items():assert sha(P/f)==h
    assert sha(P/'launch.py')==p['runtime_launcher_sha256']
    assert json.loads((P/'startup_parity.json').read_text())['passed']
    assert sha(P/'token_manifest.json')==p['data_manifest_sha256'];assert sha(P/'window_manifest.json')==p['window_manifest_sha256']
    for f,m in json.loads((P/'model_manifest.json').read_text())['files'].items():assert sha(P/'original'/f)==m['sha256']
    windows=json.loads((P/'window_manifest.json').read_text())
    for split,r in windows.items():
        assert all(abs(a-b)>33 for a in r['starts'] for b in r['excluded_v1_v2_starts'])
        assert all(abs(a-b)>33 for i,a in enumerate(r['starts']) for b in r['starts'][i+1:])
    for kind in KINDS:
        for seed in SEEDS:
            for endpoint in (['equal_data'] if kind=='normal' else ['equal_data','equal_cpu']):
                stem=f'{kind}_{seed}_{endpoint}';r=json.loads((P/f'{stem}.json').read_text());f=P/'checkpoints'/f'{stem}.pt';assert sha(f)==r['checkpoint_sha256']
                state=torch.load(f,weights_only=True);assert all(torch.isfinite(t).all() for t in state.values());m=restore(kind,seed,endpoint)
                if seed==101 and endpoint=='equal_data':
                    ev=evaluate(m,ds['test'],'test');assert ev['window_nll_nats']==r['test']['window_nll_nats'];replays.append(stem)
                    # All treatments retain the native inference/cache contract.
                    # Check one actual trained prefix continuation per treatment.
                    m.eval();ss=torch.tensor(windows['test']['starts'][:2]);x,y=batch(ds['test'],ss)
                    with torch.inference_mode():
                        full=m.base(x,use_cache=False).logits[:,16:]
                        cache=m.base(x[:,:16],use_cache=True).past_key_values
                        continued=m.base(x[:,16:],past_key_values=cache,use_cache=True).logits
                        error=float((full-continued).abs().max())
                        assert torch.allclose(full,continued,atol=1e-4,rtol=1e-5),(kind,error)
                        assert torch.equal(full.argmax(-1),continued.argmax(-1))
                        m(x);t=time.process_time();w=time.perf_counter()
                        for _ in range(8):m(x)
                        cpu=(time.process_time()-t)/8;wall=(time.perf_counter()-w)/8
                    cache_checks.append({'checkpoint':stem,'max_abs_logits':error,'argmax_equal':True,'warmed_full_prefix_cpu_seconds':cpu,'warmed_full_prefix_wall_seconds':wall,'timing_scope':'Batch2 context32, native causal forward without cache; eight warmed repetitions. Continuation parity separately checks16prefill+16continuation positions.'})
                records.append(stem);del state,m
    for family in ['normal','masked','direct']:assert json.loads((P/f'accounting_{family}.json').read_text())['subset_logits_exact']
    write('validation.json',{'passed':True,'checkpoint_count':len(records),'checkpoints':records,'exact_score_replays':replays,'native_cache_continuation_checks':cache_checks,'source_and_original_unchanged':True,'new_eval_contexts_exclude_v1_v2':True,'selected_file_loading_verified':True});print('PASS',len(records),'checkpoints',len(replays),'score replays',flush=True)
if __name__=='__main__':main()
