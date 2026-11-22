"""Equal-or-greater persistent runtime-state bytes control. No sidecar/gate in output."""
import json,time,sys
import numpy as np
from pilot import SEEDS,HERE,init,train,classify,gen,metrics,digest
p=HERE/'wide_results.json'
def eval_one(seed):
    f=HERE/f'wide_{seed}.npz'
    with np.load(f,allow_pickle=False) as weights: w={k:weights[k] for k in weights}
    r={};r['sha256']=digest(f);r['params']=sum(v.size for v in w.values());r['dynamic_state_bytes']=56*4
    for x,l,shift in [('id',48,False),('ood',128,True)]:
      z=classify(w,gen((8000 if shift else 7000)+seed,1200,l,shift))
      r[x]=metrics(z,np.zeros(len(z['y']),bool))
    return r
if '--verify' in sys.argv:
    old=json.loads(p.read_text())
    for seed in SEEDS:
      assert old[str(seed)]==eval_one(seed)
    (HERE/'wide_replay.json').write_text(json.dumps({'fresh_process_exact_metrics':3,'results_sha256':digest(p)})+'\n')
else:
    out={}
    for seed in SEEDS:
      w=train(init(seed,56),1000+seed);np.savez(HERE/f'wide_{seed}.npz',**w)
      out[seed]=eval_one(seed)
    p.write_text(json.dumps(out,indent=2)+'\n')
    for split in ('id','ood'):print(split,np.mean([v[split]['acc'] for v in out.values()]),flush=True)
