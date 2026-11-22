"""Unofficial retrieval-sidecar design, trainable selective recurrent proxy (not Mamba)."""
import struct, json, time, math, hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
K,V,D,BUF=12,8,48,4
NONE=V; RESET=K*V; FILL=RESET+1; VOC=FILL+8
SEEDS=(17,29,43)
# Fixed application record layout: byte key, byte value, uint16 epoch, uint16 total event sequence.
# 4*6 byte ring + byte next-write, byte used, uint16 query epoch, uint16 last-write-sequence, byte last-write-key.
RECORD=struct.Struct('<BBHH')
HEADER=struct.Struct('<BBHHB')
BASE=BUF*RECORD.size
SIDE_BYTES=BASE+HEADER.size
MODES=('local','far','overwrite','absent','stale','evicted')
def gen(seed,n,L,shift=False):
    """Generate observations and distinct labelled cases. Answer is computed from a replay, never an input."""
    rng=np.random.default_rng(seed); T=np.empty((n,L),np.uint8); Q=rng.integers(K,size=n,dtype=np.int32)
    Y=np.empty(n,np.int32); G=np.empty(n,np.int8)
    for i,q in enumerate(Q):
      seq=rng.integers(FILL,VOC,size=L,dtype=np.uint8)
      g=i%len(MODES); G[i]=g
      choices=[x for x in range(K) if x!=q]
      # Deterministic case proportions; shifted families move events further away and add distractors.
      nw=int(rng.integers(5,10))+(3 if shift else 0)
      e=sorted(rng.choice(np.arange(1,L-1),size=nw,replace=False))
      for p in e:
        seq[p]=rng.choice(choices)*V+rng.integers(V)
      # Reserve a reset location independent of labels/results, with nonempty current epoch.
      reset=min(L-3,max(1,int(L*(0.65 if shift else 0.55))+rng.integers(-4,5)))
      seq[reset]=RESET
      for p in range(reset+1,min(L,reset+max(5,L//6))):
        # New authenticated record events near epoch start.
        if rng.random()<.37: seq[p]=rng.choice(choices)*V+rng.integers(V)
      val=int(rng.integers(V))
      if g==0:   seq[-1]=q*V+val
      elif g==1: seq[max(reset+1,L-(17 if shift else 9))]=q*V+val
      elif g==2:
        seq[reset+1]=q*V+((val+1)%V)
        seq[-2 if shift else -3]=q*V+val
      elif g==4: seq[max(0,reset-2)]=q*V+val
      elif g==5:
        seq[reset+1]=q*V+val
        for p in range(L-5,L): seq[p]=rng.choice(choices)*V+rng.integers(V)
      T[i]=seq
      r=NONE
      for x in seq:
        if x==RESET:r=NONE
        elif x<RESET and x//V==q:r=x%V
      Y[i]=r
    return T,Q,Y,G
def make_sidecar(seq):
    ring=bytearray(SIDE_BYTES); at=used=ep=last_t=0; last_key=255
    for p,x in enumerate(seq,1):
      if x==RESET:ep+=1
      elif x<RESET:
        k,v=divmod(int(x),V)
        RECORD.pack_into(ring,at*RECORD.size,k,v,ep,p)
        at=(at+1)%BUF; used=min(used+1,BUF)
        last_key,last_t=k,p
    HEADER.pack_into(ring,BASE,at,used,ep,last_t,last_key)
    return bytes(ring)
def probe(r,q,unsafe=False):
    at,used,ep,*_=HEADER.unpack_from(r,BASE)
    for j in range(used):
      k,v,when,_=RECORD.unpack_from(r,((at-1-j)%BUF)*RECORD.size)
      if k==q and (unsafe or when==ep):return v
    return -1
def features(p,T,Q,A):
    margin=np.sort(p,axis=-1)[:,-1]-np.sort(p,axis=-1)[:,-2]
    last=np.array([HEADER.unpack_from(a,BASE)[3:] for a in A])
    last_age=np.minimum(16,T.shape[1]-last[:,0])/16
    same=last[:,1]==Q
    entropy=-np.sum(p*np.log(np.maximum(1e-10,p)),axis=-1)/math.log(V+1)
    # These are observation features, not probe result/ring contents/episode family.
    return np.column_stack((1-p.max(-1),1-margin,entropy,p[:,NONE],last_age,same,last_age*same,
                            (1-margin)*same,entropy*last_age)).astype('f')
def init(seed,d):
    rng=np.random.default_rng(seed)
    return {n:(rng.normal(size=sh)*.05).astype('f') for n,sh in
       [('u',(VOC,d)),('g',(VOC,d)),('w',(K,d,V+1)),('b',(K,V+1))]}
def forward(w,T,Q):
    B,L=T.shape;d=w['u'].shape[1]
    h=np.zeros((L+1,B,d),'f')
    u=w['u'][T]; g=1/(1+np.exp(-w['g'][T]))
    for j in range(L):
      h[j+1]=g[:,j]*h[j]+(1-g[:,j])*u[:,j]
    W=w['w'][Q]; logits=(h[-1,:,None,:]@W)[:,0,:]+w['b'][Q]
    e=np.exp(logits-logits.max(-1,keepdims=True)); p=e/e.sum(-1,keepdims=True)
    return p,(h,u,g,W)
def train(w,seed,steps=960):
    m={k:np.zeros_like(v) for k,v in w.items()};s={k:np.zeros_like(v) for k,v in w.items()}
    pop=gen(seed,4608,32);r=np.random.default_rng(seed+100)
    for step in range(1,steps+1):
      ii=r.choice(len(pop[0]),size=96,replace=False)
      T,Q,Y=map(lambda v:v[ii],pop[:3]);B,L=T.shape
      p,(h,u,g,W)=forward(w,T,Q)
      p[np.arange(B),Y]-=1;p/=B
      grads={k:np.zeros_like(v) for k,v in w.items()}
      np.add.at(grads['w'],Q,h[-1,:,:,None]*p[:,None,:]);np.add.at(grads['b'],Q,p)
      dh=(W@p[...,None])[...,0]
      for j in range(L-1,-1,-1):
        ij=T[:,j];gj=g[:,j]
        np.add.at(grads['u'],ij,dh*(1-gj))
        np.add.at(grads['g'],ij,dh*(h[j]-u[:,j])*gj*(1-gj))
        dh*=gj
      for k,x in w.items():
        a=grads[k];m[k]=.9*m[k]+.1*a;s[k]=.99*s[k]+.01*a*a
        x-=.012*(m[k]/(1-.9**step))/(np.sqrt(s[k]/(1-.99**step))+1e-8)
    return w
def classify(w,pop,infer=None):
    T,Q,Y,G=pop; t0=time.perf_counter()
    if infer is None: infer=lambda t,q:forward(w,t,q)[0]
    p=np.concatenate([infer(T[i:i+64],Q[i:i+64]) for i in range(0,len(Q),64)])
    assert p.shape==(len(Q),V+1) and np.all(np.isfinite(p)) and np.max(np.abs(p.sum(-1)-1))<2e-6
    forward_us=(time.perf_counter()-t0)*1e6/len(Q)
    t0=time.perf_counter();rings=[make_sidecar(row) for row in T]
    sidecar_us=(time.perf_counter()-t0)*1e6/len(Q)
    t0=time.perf_counter()
    hits=np.array([probe(a,int(q)) for a,q in zip(rings,Q)])
    probe_us=(time.perf_counter()-t0)*1e6/len(Q)
    unsafe=np.array([probe(a,int(q),True) for a,q in zip(rings,Q)])
    x=features(p,T,Q,rings)
    pred=p.argmax(-1); correct=(pred==Y)
    target=(hits>=0)&(~correct)
    assert np.all(Y[hits>=0]==hits[hits>=0]), 'stale/evicted lookup returned as hit'
    z=np.log(np.maximum(p,1e-10))
    return dict(X=x,Q=Q,y=Y,G=G,pred=pred,correct=correct,hits=hits,unsafe=unsafe,better=target,
        forward_us=forward_us,sidecar_build_us=sidecar_us,probe_us=probe_us)
def fit_gate(z):
    x=z['X'].astype('d'); y=z['better'];mean=x.mean(0);sd=np.maximum(x.std(0),.025)
    x=np.column_stack(((x-mean)/sd,np.ones(len(x))))
    w=np.zeros(x.shape[1]); t0=time.perf_counter()
    for _ in range(350):
      p=1/(1+np.exp(-np.clip(x@w,-30,30)))
      w-=.3*((x.T@(p-y))/len(x)+.01*w)
    return dict(mean=mean,sd=sd,w=w,fit_us=(time.perf_counter()-t0)*1e6)
def scores(z,g):
    x=z['X']; t=time.perf_counter()
    neural=((x-g['mean'])/g['sd'])@g['w'][:-1]+g['w'][-1]
    ns=(time.perf_counter()-t)*1e6/len(x)
    # Strong hand-designed gate covering likely errors and older records without reading the ring.
    hand=x[:,1]*(1-.55*x[:,5])*(1-.30*x[:,3])
    return dict(confidence=x[:,1],heuristic=hand,learned=neural),ns
def metrics(z,s,unsafe=False):
    h=z['unsafe'] if unsafe else z['hits']; y=z['y']
    pred=np.where(s&(h>=0),h,z['pred'])
    regret=~(pred==y); lookup=s.mean(); accuracy=1-regret.mean()
    return dict(acc=float(accuracy),rate=float(lookup),utility=float(accuracy-.025*lookup),
        false_calls=int(np.sum(s&(h<0))),missed_improvements=int(np.sum(~s&z['better'])),
        false_present=int(np.sum((pred!=NONE)&(y==NONE))),
        accuracy_by_family={name:float(np.mean(pred[z['G']==g]==y[z['G']==g])) for g,name in enumerate(MODES)},
        n=len(y))
def decide(sc,threshold):return sc>threshold
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(seed):
    t0=time.perf_counter(); w=train(init(seed,D),1000+seed)
    fit=classify(w,gen(2000+seed,624,48)); dev=classify(w,gen(3000+seed,624,48))
    g=fit_gate(fit); ds,_=scores(dev,g)
    # Development calibration targets 40% reads, not test quantiles or labels.
    cuts={n:float(np.quantile(sc,.60)) for n,sc in ds.items()}
    target=HERE/f'model_{seed}.npz'
    np.savez(target,**w,gate_mean=g['mean'],gate_sd=g['sd'],gate_w=g['w'])
    return dict(seed=seed,cuts=cuts,sha256=digest(target),train_and_calibrate_sec=time.perf_counter()-t0)
def evaluate(b):
    seed=b['seed']; path=HERE/f'model_{seed}.npz';assert digest(path)==b['sha256']
    with np.load(path,allow_pickle=False) as data:
      w={k:data[k] for k in ('u','g','w','b')};g={k:data['gate_'+k] for k in ('mean','sd','w')}
    return evaluate_gate(b,g,lambda t,q:forward(w,t,q)[0])
def evaluate_gate(b,g,infer):
    seed=b['seed']
    all_runs={}
    for key,steps,shift in [('id',48,False),('ood',128,True)]:
      z=classify(None,gen((8000 if shift else 7000)+seed,1200,steps,shift),infer); ss,ns=scores(z,g)
      n=len(z['y']); all_on=np.ones(n,bool); off=~all_on
      policies={'recurrent':off,'always_exact':all_on,**{k:decide(ss[k],v) for k,v in b['cuts'].items()}}
      fixed={k:metrics(z,m) for k,m in policies.items()}
      fixed['unsafe_ignore_epoch']=metrics(z,all_on,True)
      # Separate offline matched-call discrimination; test-feature-only top-k; not deployable streaming.
      limit=n*2//5
      def ranked(sc):
        yes=np.zeros(n,bool);yes[np.argsort(-sc,kind='stable')[:limit]]=True;return yes
      equal={k:metrics(z,ranked(s)) for k,s in ss.items()}
      equal['seeded_random']=metrics(z,ranked(np.random.default_rng(9000+seed).random(n)))
      equal['oracle_not_deployable']=metrics(z,ranked(z['better'].astype(float)))
      all_runs[key]=dict(fixed_dev_threshold=fixed,offline_exact_40pct=equal,
        timing_us={k:z[k] for k in ('forward_us','sidecar_build_us','probe_us')}|{'gate_matrix_batched_us':ns},
        exact_hits=int(np.sum(z['hits']>=0)),gain_opportunities=int(np.sum(z['better'])))
    return all_runs
def main():
    import sys
    path=HERE/'results.json'
    if '--verify' in sys.argv:
      r=json.loads(path.read_text())
      def strip(x): return {k:{a:b for a,b in v.items() if a!='timing_us'} for k,v in x.items()}
      for k in r['arms']: assert strip(evaluate(k))==strip(k['result'])
      (HERE/'replay.json').write_text(json.dumps({'fresh_process_exact_metrics':len(r['arms']),'run_sha256':digest(path)})+'\n')
      return
    arms=[]
    for seed in SEEDS:
      r=build(seed);r['result']=evaluate(r);arms.append(r)
      print(seed, {split:{name:round(x['acc'],4) for name,x in v['fixed_dev_threshold'].items()} for split,v in r['result'].items()},flush=True)
    ranks={}
    for scope in ('fixed_dev_threshold','offline_exact_40pct'):
      names=arms[0]['result']['id'][scope]
      for split in ('id','ood'):
        m={n:{k:float(np.mean([a['result'][split][scope][n][k] for a in arms])) for k in ('acc','rate','utility')}
           for n in names if 'unsafe' not in n and 'oracle' not in n}
        best=max(x['utility'] for x in m.values())
        ranks[f'{scope}/{split}']=[dict(method=n,score=100*v['utility']/best,**v) for n,v in sorted(m.items(),key=lambda i:-i[1]['utility'])]
    path.write_text(json.dumps(dict(arms=arms,ranks=ranks,recurrent_state_bytes=D*4,
      sidecar_and_header_bytes=SIDE_BYTES,gate_weight_and_norm_bytes=28*8,
      limits='CPU NumPy selective diagonal recurrent proxy, not Mamba-3/4 implementation. Synthetic KV event case fixture only. Matrix timings batch-amortized.'),indent=2)+'\n')
    print(json.dumps(ranks))
if __name__=='__main__':main()
