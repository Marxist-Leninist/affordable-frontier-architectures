from run import *
verify();items=data();z=np.load(P/'losses.npz');docs=sorted({i['doc'] for i in items});rng=np.random.default_rng(27);draw=rng.integers(0,len(docs),(5000,len(docs)));masks=[np.array([i['doc']==d for i in items]) for d in docs];den=np.array([int(m.sum())*128 for m in masks]);out={'windows':len(items),'documents':len(docs),'targets':len(items)*128,'models':{},'contrasts':{}}
for a in ARMS:
 means={str(v):np.mean([z[f'{a}_seed{s}_{v}'] for s in (1,2,3)],axis=0) for v in ('neural',*SCOPES)}
 out['models'][a]={v:float(x.mean()) for v,x in means.items()}
 for v in SCOPES:
  for ref in ('full','neural'):
   if str(v)==ref:continue
   diff=means[str(v)]-means[ref];tot=np.array([diff[m].sum() for m in masks]);boot=tot[draw].sum(1)/den[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975]);out['contrasts'][f'{a}_{v}_minus_{ref}']=dict(mean=float(diff.mean()),ci95=[float(lo),float(hi)],passes_fixed_gate=bool(hi<=.01) if ref=='full' else None,document_sums=tot.tolist(),document_target_counts=den.tolist())
write('summary.json',out);print(json.dumps(out['models'],indent=2));print({k:(v['mean'],v['ci95'],v['passes_fixed_gate']) for k,v in out['contrasts'].items()})
if (P/'timing.json').exists():
 rows=json.loads((P/'timing.json').read_text())['rows'];summ={}
 for a in ARMS:
  for v in ('neural',*SCOPES):
   rr=[r for r in rows if r['arm']==a and r['scope']==v];summ[a+'_'+str(v)]={'samples':len(rr),'total_wall_ms_p50_p95':np.quantile([1000*(r['prefill_wall_seconds']+r['decode_wall_seconds']) for r in rr],[.5,.95]).tolist(),'final_copy_index_bytes_median':float(np.median([r['final_copy_index_bytes'] for r in rr])),'final_state_bytes':rr[0]['final_state_bytes'],'weight_bytes':rr[0]['weight_bytes']}
 write('timing_summary.json',summ);print(json.dumps(summ,indent=2))
