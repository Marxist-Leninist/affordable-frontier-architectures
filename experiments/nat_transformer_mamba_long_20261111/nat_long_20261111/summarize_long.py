"""All-seed summary with paired document-cluster uncertainty, no selection."""
import json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def describe(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1)) if len(a)>1 else None,'values':a.tolist()}

def main():
    e=json.loads((P/'evaluation_results.json').read_text());selection=e['selection'];manifest=json.loads((P/'evaluation'/'test_manifest.json').read_text());out={'models':{},'paired_document_bootstrap':{},'unigram_baseline':e['unigram_bpb']}
    for kind in ['transformer','mamba']:
        trials=[json.loads((P/'runs'/f'final_{kind}_s{s}_lr{selection[kind]}'/'result.json').read_text()) for s in [1,2,3]]
        tasks=[json.loads((P/'runs'/f'final_{kind}_s{s}_lr{selection[kind]}'/'task_result.json').read_text()) for s in [1,2,3]]
        d={'parameters':trials[0]['parameters'],'selected_lr':selection[kind],'training_cpu_seconds':describe([x['training_cpu_seconds'] for x in trials]),'training_wall_seconds':describe([x['training_wall_seconds'] for x in trials]),'end_to_end_text_wall_seconds':describe([x['end_to_end_wall_seconds'] for x in trials]),'peak_rss_kib':describe([x['peak_rss_kib'] for x in trials]),'task_training_cpu_seconds':describe([x['training_cpu_seconds'] for x in tasks]),'metrics':{}}
        for budget in ['fixed_data','cpu_900s','separate_task_adaptation']:
            rows=[x for x in e['rows'] if x['kind']==kind and x['budget']==budget and x['status']=='completed']
            if len(rows)!=3:
                d['metrics'][budget]={'status':'incomplete','completed_seeds':[x['seed'] for x in rows]};continue
            d['metrics'][budget]={}
            for dataset in rows[0]['metrics']:
                d['metrics'][budget][dataset]={metric:describe([x['metrics'][dataset][metric] for x in rows]) for metric in ['bpb','byte_accuracy','exact_match']}
            if budget=='cpu_900s':
                d['time_budget_steps']=[x['checkpoint_step'] for x in rows]
                d['time_budget_actual_cpu_seconds']=[x['checkpoint_extra']['training_cpu_seconds'] for x in rows]
            if budget=='fixed_data':
                d['conditioning_bpb_difference_shuffled_minus_real']=describe([x['conditioning_diagnostic']['shuffled_prefix']['bpb']-x['conditioning_diagnostic']['real_prefix']['bpb'] for x in rows])
        out['models'][kind]=d
    # Pair seeds and held-out documents; average fitted seeds before bootstrap.
    rng=np.random.default_rng(20261111)
    for budget in ['fixed_data','cpu_900s']:
        for dataset in ['wikitext','fineweb']:
            pairs=[]
            for seed in [1,2,3]:
                rows={k:next((x for x in e['rows'] if x['kind']==k and x['seed']==seed and x['budget']==budget and x['status']=='completed'),None) for k in ['transformer','mamba']}
                if any(x is None for x in rows.values()):continue
                pairs.append((np.array(rows['mamba']['metrics'][dataset]['per_example_loss'])-np.array(rows['transformer']['metrics'][dataset]['per_example_loss']))/math.log(2))
            if len(pairs)!=3:continue
            difference=np.mean(pairs,axis=0);groups=np.array(manifest[dataset]['group_ids']);ids=np.unique(groups)
            sums=np.array([difference[groups==g].sum() for g in ids]);counts=np.array([(groups==g).sum() for g in ids])
            draws=rng.integers(0,len(ids),(5000,len(ids)));boot=sums[draws].sum(1)/counts[draws].sum(1)
            out['paired_document_bootstrap'][budget+'_'+dataset]={'mamba_minus_transformer_bpb':float(difference.mean()),'conditional_document_95_interval':np.quantile(boot,[.025,.975]).tolist(),'documents':len(ids),'seed_paired_means':np.mean(pairs,axis=1).tolist(),'scope':'Conditional on these three fitted seeds and this corpus cohort. Document resampling, target-count weighted. Not population uncertainty over arbitrary architectures; secondary comparisons exploratory.'}
    # Assert actual paired data, including separate capability adaptation.
    paired={}
    for seed in [1,2,3]:
        rr={k:json.loads((P/'runs'/f'final_{k}_s{seed}_lr{selection[k]}'/'result.json').read_text()) for k in ['transformer','mamba']}
        tr={k:json.loads((P/'runs'/f'final_{k}_s{seed}_lr{selection[k]}'/'task_result.json').read_text()) for k in ['transformer','mamba']}
        assert rr['transformer']['batch_sha256']==rr['mamba']['batch_sha256']
        assert tr['transformer']['batch_sha256']==tr['mamba']['batch_sha256']
        assert rr['transformer']['initial_common_hashes']==rr['mamba']['initial_common_hashes']
        paired[str(seed)]={'text_hash':rr['transformer']['batch_sha256'],'task_hash':tr['transformer']['batch_sha256'],'common_initialization':rr['transformer']['initial_common_hashes']}
    out['pairing_verified']=paired
    if (P/'standalone_timings.json').exists():out['standalone_timings']=json.loads((P/'standalone_timings.json').read_text())
    (P/'summary_long.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['models'],indent=2))

if __name__=='__main__':main()
