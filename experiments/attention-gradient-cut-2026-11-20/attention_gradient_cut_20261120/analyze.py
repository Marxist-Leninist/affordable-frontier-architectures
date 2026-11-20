from bench import *
p044=DATA.parent; manifest=json.loads((p044/'protocol.json').read_text())
dataset={k:manifest['data'][k] for k in ('train_524288.bin','dev.bin',*(m+'.bin' for m in DOM))}
assert all(sha(DATA/k)==r['sha256'] for k,r in dataset.items())
by={k:json.loads((OUT/f'{k}.json').read_text()) for k in KINDS}
def get(k,field):return np.array([[row['test'][d][field] for d in DOM] for row in by[k]])
mu=lambda q:float(np.mean(q))
for k,a in by.items():
 assert [r['seed'] for r in a]==list(SEEDS)
 for i,r in enumerate(a):
  assert r['common_input_sha256']==by['full'][i]['common_input_sha256']
  assert r['checkpoint_readback_bitexact'] and sha(OUT/r['checkpoint'])==r['sha256']
  assert r['targets']==460800 and all(x['scored_targets']==4032 for x in r['test'].values())
for l,r in zip(by['full'],by['exact_recompute']):
 a=torch.load(OUT/l['checkpoint'],weights_only=True);b=torch.load(OUT/r['checkpoint'],weights_only=True)
 assert list(a)==list(b) and all(torch.equal(v,b[n]) for n,v in a.items())
ref_bpb=mu(get('full','bpb'));ref_top1=mu(get('full','top1'))
rg=np.random.default_rng(21239);ix=rg.integers(0,3,(5000,3));win=rg.integers(0,21,(5000,3,21))
T={}
for k,r in by.items():
 diff=get(k,'by_window_bpb')-get('full','by_window_bpb');sample=diff[ix[:,:,None,None],np.arange(3)[None,None,:,None],win[:,None,:,:]].mean((1,2,3))
 b=mu(get(k,'bpb')); top=mu(get(k,'top1'))
 pp=json.loads((OUT/f'profile_{k}.json').read_text())
 T[k]={'bpb':b,'delta_vs_full':float(diff.mean()),'conditional_cohort_ci95':np.quantile(sample,[.025,.975]).tolist(),'top1':top,'quality_index_full100':100*math.sqrt((top/ref_top1)*2**(ref_bpb-b)),'per_domain_bpb':dict(zip(DOM,np.mean(get(k,'bpb'),0).tolist())),'late_context_half_bpb':mu(get(k,'bpb_late_half')),'train_step_wall_s_mean':mu([z['measured_train_step_time_s'] for z in r]),'profile':pp}
rank=sorted(T,key=lambda k:T[k]['quality_index_full100'],reverse=True)
out={'table':T,'q_index_order':rank,'exact_recompute_checkpoint_tensors_bitexact':True,'real_data_verified':dataset,'scope':'3 seeds, 460800 train targets each, 12096 historical heldout targets each, width48 and two layers, context up to192, CPU','proxy_caveat':'saved nonparameter storage bytes at forward loss boundary, exact tensor-storage dedup within graph + detached cache memory; not full peak allocation; fresh process RSS/runtime separate'}
(OUT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for k in rank:print(k,T[k])
