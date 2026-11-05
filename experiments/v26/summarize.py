from run import *
z=np.load(P/'losses.npz');items=data();mask=np.load(P/'subgroup.npy');idx=np.array([i['doc'] for i in items]);docs=np.unique(idx);rng=np.random.default_rng(26);resamples=rng.integers(0,len(docs),(5000,len(docs)))
def arr(arm,cond):return z['copy_'+cond] if arm=='copy' else np.mean([z[f'{arm}_seed{s}_{cond}'] for s in (1,2,3)],axis=0)
cohorts={'all':np.ones_like(mask),'distant_repeat':mask};out={}
for cohort,m in cohorts.items():
 counts=np.array([m[idx==d].sum() for d in docs]);den=counts[resamples].sum(1);valid=den>0;means={};effects=[]
 for arm in list(ARMS)+[a+'Hybrid' for a in ARMS]+['copy']:
  conditions=list(CONDS)+(['suffixCopyOnly'] if arm.endswith('Hybrid') else [])
  means[arm]={c:float(arr(arm,c)[m].mean()) if m.any() else None for c in conditions}
  for c,b in [('donor2048','original2048'),('original2048','suffix512')]+([('suffixCopyOnly','original2048')] if arm.endswith('Hybrid') else []):
   diff=arr(arm,c)-arr(arm,b);sums=np.array([diff[idx==d][m[idx==d]].sum() for d in docs]);boot=sums[resamples].sum(1)[valid]/den[valid]
   effects.append(dict(arm=arm,comparison=c+' minus '+b,effect_bpb=float(diff[m].mean()) if m.any() else None,bootstrap95=np.quantile(boot,[.025,.975]).tolist() if len(boot) else None,zero_denominator_bootstraps=int((~valid).sum())))
 out[cohort]=dict(targets=int(m.sum()),fraction=float(m.mean()),documents=int(sum(counts>0)),windows=int(sum(m.any(1))),means=means,effects=effects)
write('summary.json',out);print(json.dumps(out,indent=2))
