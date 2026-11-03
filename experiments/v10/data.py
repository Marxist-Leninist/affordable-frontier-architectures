import numpy as np, hashlib

def generate(seed,n,length,facts,balanced=False):
 r=np.random.default_rng(seed); xs=[]; ys=[]; positions=[]
 # Separate structural and value RNG streams prevents target affecting positions/keys/filler.
 rv=np.random.default_rng(seed+1000000007)
 qs=np.arange(n)%16; ts=np.arange(n)%16
 if balanced: r.shuffle(qs); rv.shuffle(ts)
 for i in range(n):
  q=int(qs[i]) if balanced else int(r.integers(16)); keys=np.r_[q,r.choice([k for k in range(16) if k!=q],facts-1,replace=False)]; r.shuffle(keys)
  vals=rv.integers(16,size=facts); qi=int(np.where(keys==q)[0][0])
  if balanced: vals[qi]=ts[i]
  # Uniformly choose nonoverlapping dimer starts through compressed coordinates.
  starts=np.sort(r.choice(length-2-facts,facts,replace=False))+np.arange(facts)
  x=r.integers(32,40,size=length,dtype=np.int64); x[-2:]=[40,q]
  for s,k,v in zip(starts,keys,vals): x[s:s+2]=[k,16+v]
  xs.append(x); ys.append(vals[qi]); positions.append(int(starts[qi]))
 return np.stack(xs),np.array(ys,dtype=np.int64),positions

def digest(x,y): return hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
def oracle(x):
 maps={int(x[i]):int(x[i+1])-16 for i in range(len(x)-2) if x[i]<16 and 16<=x[i+1]<32};return maps[int(x[-1])]
