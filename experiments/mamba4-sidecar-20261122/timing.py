"""CPU host microbench: identical prepared sidecar state + same precomputed model probabilities."""
import json,time,statistics
import numpy as np
from pilot import HERE,SEEDS,gen,forward,make_sidecar,probe,features
rows=json.loads((HERE/'results.json').read_text())['arms']
out=[]
for b in rows:
  seed=b['seed']
  with np.load(HERE/f'model_{seed}.npz',allow_pickle=False) as ar:
    w={k:ar[k] for k in ('u','g','w','b')}; mean=ar['gate_mean'];sd=ar['gate_sd'];gw=ar['gate_w']
  for tag,L in [('id',48),('ood',128)]:
    t,q,y,_=gen((8000 if tag=='ood' else 7000)+seed,1200,L,tag=='ood')
    p=np.concatenate([forward(w,t[i:i+64],q[i:i+64])[0] for i in range(0,len(q),64)])
    a=[make_sidecar(row) for row in t];pred=p.argmax(-1)
    ql=q.tolist();n=len(t)
    def always():
      h=np.array([probe(x,y) for x,y in zip(a,ql)])
      return np.where(h>=0,h,pred)
    def gated():
      X=features(p,t,q,a)
      take=((X-mean)/sd)@gw[:-1]+gw[-1]>b['cuts']['learned']
      res=pred.copy()
      for j in np.flatnonzero(take):
        h=probe(a[j],ql[j])
        if h>=0:res[j]=h
      return res
    assert (always()==y).sum()==round(n*b['result'][tag]['fixed_dev_threshold']['always_exact']['acc'])
    assert (gated()==y).sum()==round(n*b['result'][tag]['fixed_dev_threshold']['learned']['acc'])
    u={name:[] for name in ('always','learned')}
    for k in range(15):
      f=[('always',always),('learned',gated)]
      for name,fn in (f[::-1] if k%2 else f):
        s=time.perf_counter_ns();fn();u[name].append((time.perf_counter_ns()-s)/1000/n)
    out.append(dict(seed=seed,split=tag,median_us_per_query={n:statistics.median(v) for n,v in u.items()}))
(HERE/'timing.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
