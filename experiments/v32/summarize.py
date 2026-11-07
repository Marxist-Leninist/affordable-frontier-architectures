"""Post-scoring deterministic summaries; no selection changes."""
from run import *
verify()
items = data('evaluation')
z = np.load(P/'losses.npz')
docs = sorted({i['doc'] for i in items})
masks = [np.array([i['doc']==d for i in items]) for d in docs]
den = np.array([m.sum()*128 for m in masks])
draw = np.random.default_rng(32).integers(0,len(docs),(5000,len(docs)))
out = dict(windows=len(items),documents=len(docs),targets=128*len(items),models={},contrasts={},order_subgroups={})
for arm in ARMS:
    means = {variant:np.mean([z[f'{arm}_seed{s}_{variant}'] for s in SEEDS],axis=0) for variant in VARIANTS}
    out['models'][arm]={v:float(x.mean()) for v,x in means.items()}
    for v,ref in [('by_order','constant'),('by_order','historical'),('by_order','neural'),('constant','neural')]:
        diff=means[v]-means[ref]
        total=np.array([diff[m].sum() for m in masks])
        bootstrap=total[draw].sum(1)/den[draw].sum(1)
        out['contrasts'][f'{arm}_{v}_minus_{ref}']=dict(mean_bpb=float(diff.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),document_sums=total.tolist(),document_target_counts=den.tolist(),per_seed_deltas=[float((z[f'{arm}_seed{s}_{v}']-z[f'{arm}_seed{s}_{ref}']).mean()) for s in SEEDS])
    out['order_subgroups'][arm]={str(k):dict(targets=int((z['order']==k).sum()),constant_bpb=float(means['constant'][z['order']==k].mean()),by_order_bpb=float(means['by_order'][z['order']==k].mean())) for k in (0,4,8,16) if (z['order']==k).any()}
write('summary.json',out)
print(json.dumps(dict(models=out['models'],primary={k:v for k,v in out['contrasts'].items() if k.endswith('minus_constant')}),indent=2))
if (P/'timing.json').exists():
    rows=json.loads((P/'timing.json').read_text())['rows']
    groups={}
    pairs={}
    for arm in ARMS:
        grouped={}
        for r in rows:
            if r['arm']==arm:
                grouped.setdefault((r['seed'],r['doc'],r['rep']),{})[r['variant']]=r
        ratios=[(p['constant']['prefill_seconds']+p['constant']['decode_seconds'])/(p['by_order']['prefill_seconds']+p['by_order']['decode_seconds']) for p in grouped.values()]
        pairs[arm]=dict(pairs=len(ratios),median_constant_over_by_order_total_ratio=float(np.median(ratios)),by_order_faster_pairs=sum(x>1 for x in ratios))
        for variant in ('neural','constant','by_order'):
            rr=[r for r in rows if r['arm']==arm and r['variant']==variant]
            groups[arm+'_'+variant]=dict(samples=len(rr),total_ms_p50_p95=np.quantile([1000*(r['prefill_seconds']+r['decode_seconds']) for r in rr],[.5,.95]).tolist(),prefill_ms_p50=float(np.median([1000*r['prefill_seconds'] for r in rr])),update32_ms_p50=float(np.median([1000*r['decode_seconds'] for r in rr])),index_bytes_median=float(np.median([r['index_bytes'] for r in rr])),state_bytes=rr[0]['state_bytes'],weight_bytes=rr[0]['weight_bytes'])
    write('timing_summary.json',dict(groups=groups,paired=pairs))
