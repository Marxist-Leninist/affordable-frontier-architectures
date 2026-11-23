import json,math,statistics,random,sys
from pathlib import Path
A=('FP32','BitNet-b1.58');S=(1733,1741)
def run(p):
 m=statistics.mean
 a={(k,s):json.loads((p/f'{k}-{s}.json').read_text()) for k in A for s in S}
 pz=json.loads((p/'FIT-FROZEN.json').read_text())
 assert len(pz)==4
 for r in pz:
  k,s=r['arm'],r['seed']
  assert r['checkpoint_sha']==a[k,s]['checkpoint_sha']
  assert r['train_offsets_sha']==a['FP32',s]['train_offsets_sha']
  assert r['curve'][-1]['targets']==4915200
  assert r['params']==1601984 and len(a[k,s]['test_bpb_by_window'])==256
 cz={s:min(a[k,s]['curve'][-1]['fit_s'] for k in A) for s in S}
 out={}
 for k in A:
  q=[a[k,s] for s in S]
  d=[a[k,s]['test_bpb']-a['FP32',s]['test_bpb'] for s in S]
  vv=[m(a[k,s]['test_bpb_by_window'][j]-a['FP32',s]['test_bpb_by_window'][j] for s in S) for j in range(256)]
  vs=[m(vv[j:j+8]) for j in range(0,256,8)];rng=random.Random(71249)
  z=sorted(m(rng.choices(vs,k=32)) for i in range(10000))
  u=dict(test_bpb=m(r['test_bpb'] for r in q),fit_s=m(r['curve'][-1]['fit_s'] for r in q),
        latency_measured=False,
        delta_vs_base=m(vv),per_seed_BPB_delta=d,ci95_blocks_fixed_seeds=[z[249],z[9749]],qualifies=max(d)<=.15,
        working_training_state_B=m(sum(r['real_training_state'].values()) for r in q),
        inference_payload_B=m(r['inference_payload']['param_bytes'] for r in q))
  ss=[]
  for s,r in zip(S,q):
   t=r['curve'][-1]['fit_s']
   ss.append({**{j:r[j] for j in ['seed','initial_bpb','test_bpb','cuda_fit_peak_B','inference_payload','real_training_state','params','checkpoint_sha','stage2_proxy_eligible_windows']},
            'fit_s':t,'BPB_gain_per_fit_min':(r['initial_bpb']-r['test_bpb'])*60/t,'bytes_targets_per_s':4915200/t,
            'first_dev_target':next((x for x in r['curve'] if x['dev_bpb']<=3.25),None),
            'common_fit_time_observed':next(x for x in reversed(r['curve']) if x['fit_s']<=cz[s])})
  u['seeds']=ss;out[k]=u
 for k,u in out.items():
  b=out['FP32']
  u['index']=100*math.sqrt(b['test_bpb']*b['fit_s']/(u['test_bpb']*u['fit_s'])) if u['qualifies'] else None
 return dict(arms=out,ranking=sorted([k for k in A if out[k]['index'] is not None],key=lambda k:-out[k]['index']),
          engineering=json.loads((p/'preflight.json').read_text())['qa'],
          prescore_holdout_exclusions=json.loads((p/'preflight.json').read_text())['prescore_holdout_exclusions'],
          elapsed_s=json.loads((p/'DONE.json').read_text())['elapsed_seconds'])
if __name__=='__main__':print(json.dumps(run(Path(sys.argv[1])),allow_nan=False,separators=(',',':')))
