"""Post-run arithmetic, source, pairing and causality audit."""
from run import *
verify();items=data();saved=json.loads((P/'items.json').read_text());assert saved==[{k:v for k,v in i.items() if k not in ('raw','changed')} for i in items]
r=json.loads((P/'results.json').read_text());assert r['complete'] and len(r['rows'])==45;z=np.load(P/'losses.npz');assert len(z.files)==45;err=0
for row in r['rows']:
 a=z[f"{row['arm']}_seed{row['seed']}_{row['condition']}"];assert a.shape==(len(items),128) and np.isfinite(a).all();err=max(err,abs(float(a.mean())-row['bpb']))
assert err<1e-12
errors=[]
with torch.inference_mode():
 torch.manual_seed(250125);x=torch.randint(256,(1,2048));y=x.clone();y[:,1985:]=(y[:,1985:]+1)%256
 for arm in ARMS:
  m=load(arm,1);full=m(x);future=m(y);opt,_=local_prefill(m,x,128,prune=True) if arm=='local' else prefill(m,x,None,128)
  causal=float((full[:,:1985]-future[:,:1985]).abs().max());parity=float((full[:,-128:]-opt).abs().max());assert causal<=1e-5 and parity<=1e-4
  errors.append(dict(arm=arm,causal_logit_error=causal,optimized_full_logit_error=parity))
write('audit.json',dict(aggregate_max_error_bpb=err,aggregate_rows=45,arrays=45,source_and_checkpoint_hashes_verified=True,items_and_donors_reconstructed=True,causal_and_parity=errors,all_local_donor_losses_identical=all(np.array_equal(z[f'local_seed{s}_original2048'],z[f'local_seed{s}_donor2048']) for s in (1,2,3))))
print('Audit passed',errors)
