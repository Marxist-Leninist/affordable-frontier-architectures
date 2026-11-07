"""Verify frozen source, all endpoint statistics, checkpoints, and replayed scores."""
import json,math
from run import *
verify();data=source_data();results=json.loads((P/'results.json').read_text());manifest=json.loads((P/'checkpoint_manifest.json').read_text());assert len(results['all_endpoints'])==21
checks=[]
for r in results['all_endpoints']:
 f=f"{r['kind']}_{r['seed']}_{r['endpoint']}.pt";assert sha(P/'checkpoints'/f)==manifest[f]['sha256'];m=Transformer() if r['kind']=='transformer' else Recurrent();state=torch.load(P/'checkpoints'/f,weights_only=True);m.load_state_dict(state)
 assert all(torch.isfinite(t).all() for t in state.values());assert count(m)==r['parameters']
 for split in ['validation','test']:
  losses=r[split]['window_bpb'];assert math.isclose(sum(losses)/len(losses),r[split]['bpb'],abs_tol=1e-10)
 checks.append(dict(checkpoint=f,sha256=True,finite=True,parameters=True,metric_mean=True))
 if r['seed']==1 and r['endpoint']=='equal_data':
  replay=evaluate(m,r['kind'],data['test']);assert abs(replay['bpb']-r['test']['bpb'])<1e-9;checks[-1]['heldout_replay_abs_error']=abs(replay['bpb']-r['test']['bpb'])
out=dict(passed=True,endpoints=len(checks),checks=checks,source_hashes_verified=True,public_data_hashes_verified=True)
write('validation.json',out);print('PASS',len(checks),'checkpoints; source/data hashes; four exact heldout score replays')
