"""Post-run artifact and long-context causality checks; no model selection."""
import time
from run import *
cpu=time.process_time();p=json.loads((P/'protocol.json').read_text());r=json.loads((P/'results.json').read_text());assert r['complete'];assert len(r['rows'])==30;assert len(r['training'])==6
for f,h in p['source_sha256'].items():assert sha(P/f)==h
for f,h in json.loads((P/'checkpoint_manifest.json').read_text()).items():assert sha(P/'checkpoints'/f)==h
z=np.load(P/'losses.npz');items=json.loads((P/'items.json').read_text());maxerr=0
for row in r['rows']:
 a=z[f"{row['label']}_seed{row['seed']}_ctx{row['context']}"];assert a.shape==(len(items),128);assert np.isfinite(a).all();maxerr=max(maxerr,abs(float(a.mean())-row['bpb']))
assert maxerr<1e-12
m=LM('transformer');m.load_state_dict(torch.load(P/'checkpoints/context2048_seed1_step250.pt',weights_only=True));m.eval();torch.manual_seed(823);x=torch.randint(0,256,(1,2048));y=x.clone();y[:,1921:]=(y[:,1921:]+17)%256
with torch.no_grad():
 a=m(x);b=m(y);causal=float((a[:,:1921]-b[:,:1921]).abs().max());zz,_=prefill(m,x,None,128);optimized=float((a[:,-128:]-zz).abs().max())
assert causal<1e-4;assert optimized<1e-4
write('artifact_audit.json',dict(complete_rows=len(r['rows']),training_runs=len(r['training']),optimizer_updates=sum(t['steps'] for t in r['training']),maximum_aggregate_reconstruction_error=maxerr,final2048_causal_future_suffix_error=causal,final2048_optimized_logits_error=optimized,source_and_checkpoint_hashes=True,cpu_seconds=time.process_time()-cpu));print((P/'artifact_audit.json').read_text())
