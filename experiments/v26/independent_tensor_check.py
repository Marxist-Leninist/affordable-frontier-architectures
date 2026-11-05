"""Run after producer timing. Frozen inference validation only; no training/timing rerun."""
from run import *
from models import state_bytes
out={};items=data();raw=torch.tensor([list(items[0]['raw'])]);x=raw[:,:-1];y=raw[:,-128:]
z=np.load(P/'losses.npz')
with torch.inference_mode():
 for arm in ARMS:
  records=[]
  for seed in (1,2,3):
   m=load(arm,seed);dtypes=sorted({str(t.dtype) for t in m.state_dict().values()});assert dtypes==['torch.float32']
   native=m(x);optimized,state=local_prefill(m,x,128,prune=True) if arm=='local' else prefill(m,x,None,128)
   error=float((native[:,-128:]-optimized).abs().max());assert error<=1e-4
   probs=native[:,-128:].double().softmax(-1);assert float((probs.sum(-1)-1).abs().max())<1e-14
   manual=(-probs.log().gather(-1,y[:,:,None]).squeeze(-1)/math.log(2)).numpy()
   stored=z[f'{arm}_seed{seed}_original2048'][0];loss_error=float(np.max(np.abs(stored-manual[0])));assert loss_error<=1e-4
   changed=x.clone();changed[:,1921:]=(changed[:,1921:]+1)%256
   causal_error=float((native[:,1920]-m(changed)[:,1920]).abs().max());assert causal_error<=1e-5
   records.append(dict(seed=seed,weight_dtype=dtypes,weight_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()),native_optimized_max_logit_error=error,independent_native_saved_max_loss_error=loss_error,first_target_causal_max_logit_error=causal_error))
  out[arm]=records
r=json.loads((P/'results.json').read_text());assert r['complete'] and len(r['rows'])==66
for row in r['rows']:assert abs(row['bpb']-float(z[row['key']].mean()))<1e-12
for seed in (1,2,3):assert np.max(np.abs(z[f'local_seed{seed}_original2048']-z[f'local_seed{seed}_donor2048']))<=1e-10
for cond in CONDS:
 cp,active=copy_scores(items,cond)
 for a in ARMS:
  for s in (1,2,3):
   p=np.exp2(-z[f'{a}_seed{s}_{cond}']);w=ALPHAS[a]*active;expected=-np.log2((1-w)*p+w*cp);assert np.max(np.abs(expected-z[f'{a}Hybrid_seed{s}_{cond}']))<1e-12
   if cond=='suffix512':
    expected=-np.log2((1-w)*np.exp2(-z[f'{a}_seed{s}_original2048'])+w*cp);assert np.max(np.abs(expected-z[f'{a}Hybrid_seed{s}_suffixCopyOnly']))<1e-12
tr=json.loads((P/'timing.json').read_text());assert len(tr['rows'])==144
assert len({(r['seed'],r['arm'],r['variant'],r['doc']) for r in tr['rows']})==144
for r in tr['rows']:
 assert r['fp32_weight_bytes']==out[r['arm']][r['seed']-1]['weight_bytes']
 expected=2*2*2048*64*4 if r['arm']=='context2048' else 2*(2*127*64*4+8) if r['arm']=='local' else 2*64*4
 assert r['prefill_state_bytes']==expected
 assert r['final_state_bytes']==(expected+2*2*32*64*4 if r['arm']=='context2048' else expected)
 assert (r['final_copy_index_bytes']==0)==(r['variant']=='neural')
out['all66_loss_means_verified']=True;out['all_hybrid_losses_verified']=True;out['local_all_seed_all_target_donor_invariance']=True;out['timing144_unique_records_weight_state_bytes_verified']=True
(P/'independent_tensor_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
