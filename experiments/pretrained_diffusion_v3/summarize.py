from pathlib import Path
import numpy as np,json
P=Path(__file__).parent;SEEDS=[101,202,303];KINDS=['normal','masked_hybrid_parent','masked_hybrid_all','masked_local_all','direct_hybrid_all','direct_local_all']
def read(n):return json.loads((P/n).read_text())
def main():
    controls=[read(f'normal_{s}_equal_data.json') for s in SEEDS];control=np.array([r['test']['window_nll_nats'] for r in controls]);cpu=np.mean([r['cpu_seconds'] for r in controls]);boot=np.random.default_rng(71338).integers(0,64,(10000,64));rows=[]
    for kind in KINDS:
        for endpoint in (['equal_data'] if kind=='normal' else ['equal_data','equal_cpu']):
            rs=[read(f'{kind}_{s}_{endpoint}.json') for s in SEEDS];a=np.array([r['test']['window_nll_nats'] for r in rs]);d=(a-control).mean(0);ci=np.quantile(d[boot].mean(1),[.025,.975]).tolist()
            rows.append({'kind':kind,'endpoint':endpoint,'test_nll_by_seed':a.mean(1).tolist(),'mean_test_nll':float(a.mean()),'seed_sd_test_nll':float(a.mean(1).std(ddof=1)),'paired_gap_nats':float(d.mean()),'conditional_window_bootstrap95':ci,'mean_training_cpu_seconds':float(np.mean([r['cpu_seconds'] for r in rs])),'cpu_fraction_of_ordinary':float(np.mean([r['cpu_seconds'] for r in rs])/cpu),'steps':[r['steps'] for r in rs],'predicted_targets':[r['stats']['predicted_targets'] for r in rs],'mean_gap_within_0_05':bool(d.mean()<=.05),'conditional_upper_ci_within_0_05':bool(ci[1]<=.05),'scope':'Bootstrap conditions on these three seeds and64windows; not production noninferiority or generalization to other data/models.'})
    io={f:read(f'accounting_{f}.json') for f in ['normal','masked','direct']}
    sizes={f:io[f]['warm_body_reads'][0]['logical_bytes'] for f in io}
    for row in rows:
        kind=row['kind'];family='normal' if kind=='normal' else kind.split('_')[0]
        totals=[]
        for seed in SEEDS:
            r=read(f'{kind}_{seed}_{row["endpoint"]}.json');s=r['stats']
            totals.append(s['ar_updates']*sizes['normal']+(s['local_updates']+s['local_skipped_above35'])*sizes[family])
        row['mean_modeled_body_read_bytes']=float(np.mean(totals))
        row['modeled_body_read_fraction_of_ordinary']=float(np.mean(totals)/(648*sizes['normal']))
    validation=read('validation.json');assert validation['passed']
    passes=[r['kind'] for r in rows if r['kind']!='normal' and r['endpoint']=='equal_data' and r['conditional_upper_ci_within_0_05'] and r['cpu_fraction_of_ordinary']<1]
    write={'complete':True,'untouched_test_nll':read('untouched.json')['test']['nll_nats'],'rows':rows,'conservative_quality_and_cpu_screen_passes':passes,'offload_body_bytes':sizes,'modeled_hybrid_mean_body_bytes':{f:(8*sizes['normal']+sizes[f])/9 for f in ['masked','direct']},'native_inference_checks':validation['native_cache_continuation_checks'],'io_scope':'Actual isolated file sizes/reads verified; hybrid-weighted totals are a schedule model, not measured primary-run offloading. Primary training stays in RAM. Shared vocabulary loading and optimizer transfers excluded from modeled body figures.','interpretation':'Prefer measured common causal quality and useful end-to-end cost, not paper fidelity. Dense Pythia proxy isolates local mechanisms with common AR anchors; does not reproduce native ternary MoE, SAT policy, rotating write shards, or exact ancestor ratio.'}
    (P/'summary.json').write_text(json.dumps(write,indent=2)+'\n');print(json.dumps(write,indent=2))
if __name__=='__main__':main()
