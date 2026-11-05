"""Run once after producer timing ends. No training, timing, or benchmark rerun."""
from run import *
from local_attention import allocated_tensor_bytes
from models import state_bytes
out={};items=data();raw=torch.tensor([list(items[0]['raw'])]);x=raw[:,:-1];y=raw[:,-128:]
with torch.inference_mode():
 for arm in ARMS:
  records=[]
  for seed in (1,2,3):
   m=load(arm,seed); dtypes=sorted({str(t.dtype) for t in m.state_dict().values()});assert dtypes==['torch.float32']
   # One real document, full native model versus optimized final128 logit path.
   native=m(x);optimized,state=local_prefill(m,x,128,prune=True) if arm=='local' else prefill(m,x,None,128)
   error=float((native[:,-128:]-optimized).abs().max());assert error<=1e-4
   manual=(-native[:,-128:].double().log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)/math.log(2)).numpy()
   stored=np.load(P/'losses.npz')[f'{arm}_seed{seed}_original2048'][0]
   loss_error=float(np.max(np.abs(stored-manual[0])));assert loss_error<=1e-4
   changed=x.clone();changed[:,1921:]=(changed[:,1921:]+1)%256
   # No output predicting first scored target may depend on that or later target bytes.
   causal_error=float((native[:,1920]-m(changed)[:,1920]).abs().max());assert causal_error<=1e-5
   records.append(dict(seed=seed,weight_dtype=dtypes,weight_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()),native_optimized_max_logit_error=error,independent_native_saved_max_loss_error=loss_error,first_target_causal_max_logit_error=causal_error,logical_state_bytes=state_bytes(state),allocated_state_bytes=allocated_tensor_bytes(state)))
  out[arm]=records
z=np.load(P/'losses.npz');r=json.loads((P/'results.json').read_text());assert r['complete'] and len(r['rows'])==45
for row in r['rows']:assert abs(row['bpb']-float(z[f"{row['arm']}_seed{row['seed']}_{row['condition']}"].mean()))<1e-12
for seed in (1,2,3):assert np.max(np.abs(z[f'local_seed{seed}_original2048']-z[f'local_seed{seed}_donor2048']))<=1e-10
summary=json.loads((P/'summary.json').read_text());
for row in summary['comparisons']:
 a,b=row['comparison'].split(' minus ');effect=float(np.mean([z[f"{row['arm']}_seed{s}_{a}"]-z[f"{row['arm']}_seed{s}_{b}"] for s in (1,2,3)]));assert abs(effect-row['effect_bpb'])<1e-12
tr=json.loads((P/'timing.json').read_text());assert len(tr['rows'])==240
assert len({(r['seed'],r['arm'],r['context'],r['doc']) for r in tr['rows']})==240
for r in tr['rows']:
 assert r['fp32_weight_bytes']==out[r['arm']][r['seed']-1]['weight_bytes']
 expected=2*2*r['context']*64*4 if r['arm'] not in ('local','gru') else 2*(2*127*64*4+8) if r['arm']=='local' else 2*64*4
 assert r['prefill_state_bytes']==expected
 assert r['final_state_bytes']==(expected+2*2*32*64*4 if r['arm'] not in ('local','gru') else expected)
out['all_45_loss_means_and_10_comparison_means_verified']=True;out['local_all_seed_all_target_donor_invariance']=True;out['timing_240_unique_records_weight_state_bytes_verified']=True
(P/'independent_tensor_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
