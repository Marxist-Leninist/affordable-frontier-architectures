"""Read completed results, keep all seeds and raw uncertainty; no model selection."""
import json, math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
def summary():
    r=json.loads((P/'results.json').read_text()); rows=r['runs']; out={'architectures':{}}
    for kind in ['transformer','mamba']:
        rs=[x for x in rows if x['kind']==kind]
        d={'parameters':rs[0]['parameters'],'training_seconds':[x['wall_seconds'] for x in rs],'targets_per_second':[x['targets_per_second'] for x in rs],'mean_training_seconds':np.mean([x['wall_seconds'] for x in rs]),'mean_targets_per_second':np.mean([x['targets_per_second'] for x in rs]),'time_budget_steps':[x['time_budget_receipt']['step'] for x in rs],'time_budget_actual_seconds':[x['time_budget_receipt']['train_wall_seconds'] for x in rs]}
        for mode in ['test_fixed_steps','test_fixed_time']:
            d[mode]={}
            for task in ['text','copy','reverse','sort']:
                d[mode][task]={}
                for metric in ['bpb','token_accuracy','exact_match']:
                    values=[x[mode][task][metric] for x in rs]
                    d[mode][task][metric]={'mean':float(np.mean(values)),'sd_across_3_seeds':float(np.std(values,ddof=1)),'seeds':values}
        for batch in [1,16]:
            ts=[x['ms'] for x in r['inference_trials'] if x['name'].startswith(kind) and x['batch']==batch]
            d[f'inference_batch{batch}']={'p50_ms':float(np.median(ts)),'p95_ms':float(np.percentile(ts,95)),'byte_outputs_per_second_at_median':batch*16/(np.median(ts)/1000),'trials':len(ts)}
        out['architectures'][kind]=d
    out['training_speed_transformer_over_mamba']=out['architectures']['mamba']['mean_training_seconds']/out['architectures']['transformer']['mean_training_seconds']
    rng=np.random.default_rng(20261111)
    paired=[]
    for seed in [1,2,3]:
        a=next(x for x in rows if x['kind']=='transformer' and x['seed']==seed)
        b=next(x for x in rows if x['kind']=='mamba' and x['seed']==seed)
        assert a['batch_sha256']==b['batch_sha256']
        paired.append((np.array(b['test_fixed_steps']['text']['per_example_loss'])-np.array(a['test_fixed_steps']['text']['per_example_loss']))/math.log(2))
    means=np.mean(paired,axis=0)
    draws=np.mean(means[rng.integers(0,len(means),(5000,len(means)))],axis=1)
    out['paired_text_mamba_minus_transformer_bpb']={'mean':float(means.mean()),'example_bootstrap_95_interval':np.quantile(draws,[.025,.975]).tolist(),'limitation':'Conditional on these 3 models and sampled windows; windows can share documents, so not a document/population or model-seed uncertainty interval.'}
    L,D,H,N,I,R,K,O,V=48,48,266,8,96,3,4,16,256
    out['forward_arithmetic_per_example']={
        'transformer_matrix_MACs':2*(4*L*D*D+2*L*L*D+2*L*D*H)+O*D*V,
        'mamba_matrix_and_depthwise_conv_MACs':4*(3*L*D*I+L*I*(2*R+2*N)+L*I*K)+O*D*V,
        'mamba_parallel_scan_scalar_multiplies_and_adds':4*3*sum(L-s for s in [1,2,4,8,16,32])*I*N,
        'limitations':'MAC counts omit embedding, normalizations, nonlinearities, Mamba dt/exponential/discretization and gating, attention softmax, backward and optimizer. Count one MAC as 2 FLOPs if needed. Not an equal-compute experiment.'}
    (P/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':summary()
