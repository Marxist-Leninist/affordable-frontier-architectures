from run import *
r=json.loads((P/'results.json').read_text());assert r['complete'];z=np.load(P/'losses.npz');items=json.loads((P/'items.json').read_text());docs=sorted({i['doc'] for i in items});idx=np.array([i['doc'] for i in items]);rng=np.random.default_rng(25);resamples=rng.integers(0,len(docs),(5000,len(docs)));counts=np.array([(idx==d).sum() for d in docs])
means={a:{c:float(np.mean([z[f'{a}_seed{s}_{c}'] for s in (1,2,3)])) for c in CONDS} for a in ARMS}
def diff(a,c,b):
 v=np.mean([z[f'{a}_seed{s}_{c}'].mean(1)-z[f'{a}_seed{s}_{b}'].mean(1) for s in (1,2,3)],axis=0);summ=np.array([v[idx==d].sum() for d in docs]);boot=summ[resamples].sum(1)/counts[resamples].sum(1)
 return dict(arm=a,comparison=c+' minus '+b,effect_bpb=float(v.mean()),document_cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),documents_positive=int(sum(v[idx==d].mean()>0 for d in docs)),seed_effects=[float(z[f'{a}_seed{s}_{c}'].mean()-z[f'{a}_seed{s}_{b}'].mean()) for s in (1,2,3)],max_absolute_target_loss_difference=float(max(np.max(np.abs(z[f'{a}_seed{s}_{c}']-z[f'{a}_seed{s}_{b}'])) for s in (1,2,3))))
write('summary.json',dict(means=means,comparisons=[diff(a,c,b) for a in ARMS for c,b in [('donor2048','original2048'),('original2048','suffix512')]],windows=len(items),documents=len(docs),targets_per_condition_seed=128*len(items),main_cpu_seconds=r['cpu_seconds']))
print(json.dumps(means,indent=2))
