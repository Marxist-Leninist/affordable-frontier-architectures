"""Bounded NumPy neural character-LM training; synthetic language, no frontier claims."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np, time,json,hashlib
from pathlib import Path

def data(seed,n):
 r=np.random.default_rng(seed);names=['alice','bob','carol','david','eve','frank'];colors=['red','blue','green','gold'];items=['book','ball','cup','key'];places=['room','garden','hall','house']
 return ''.join(f"{r.choice(names)} put the {r.choice(colors)} {r.choice(items)} in the {r.choice(places)}. then {r.choice(names)} went home.\n" for _ in range(n))
TRAIN=data(11,1500);seen=set(TRAIN.splitlines());held=[]
for line in data(73,3000).splitlines():
 if line not in seen:
  held.append(line);seen.add(line)
 if len(held)==300:break
assert len(held)==300
VALID='\n'.join(held)+'\n';VOCAB=sorted(set(TRAIN+VALID));V=len(VOCAB);ENC={c:i for i,c in enumerate(VOCAB)}
def encode(s):return np.array([ENC[c] for c in s])
def softmax(x):
 e=np.exp(x-x.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
class LM:
 def __init__(self,kind,d,seed):
  self.kind=kind;self.d=d;r=np.random.default_rng(seed);self.p={}
  def w(k,shape,scale=None):self.p[k]=r.normal(0,scale or shape[0]**-.5,shape)
  w('E',(V,d),.1);w('U',(d,V))
  if kind=='attention':
   for k in ('Q','K','W','O'):w(k,(d,d))
   w('F',(d,2*d));w('G',(2*d,d))
  else:w('R',(d,d),.1)
 def count(self):return sum(x.size for x in self.p.values())
 def loss(self,x,y,grad=True):
  p=self.p;b,t=x.shape;d=self.d;z=p['E'][x]
  if self.kind=='attention':
   pos=np.arange(t)[:,None];freq=10000**(-np.arange(d)[None,:]/d);z=z+.1*np.sin(pos*freq)
   q=z@p['Q'];k=z@p['K'];v=z@p['W'];score=q@k.transpose(0,2,1)/np.sqrt(d);score=np.where(np.triu(np.ones((t,t),bool),1),-1e30,score);a=softmax(score);c=a@v;h=z+c@p['O'];f=np.tanh(h@p['F']);out=h+f@p['G']
  else:
   hs=[];prev=np.zeros((b,d))
   for i in range(t):prev=np.tanh(z[:,i]+prev@p['R']);hs.append(prev)
   out=np.stack(hs,1)
  prob=softmax(out@p['U']);
  if y is None:return prob
  loss=-np.log(prob[np.arange(b)[:,None],np.arange(t),y]).mean()
  if not grad:return float(loss)
  dl=prob.copy();dl[np.arange(b)[:,None],np.arange(t),y]-=1;dl/=b*t
  g={'U':out.reshape(-1,d).T@dl.reshape(-1,V)};do=dl@p['U'].T
  if self.kind=='attention':
   g['G']=f.reshape(-1,2*d).T@do.reshape(-1,d);df=(do@p['G'].T)*(1-f*f);g['F']=h.reshape(-1,d).T@df.reshape(-1,2*d);dh=do+df@p['F'].T
   g['O']=c.reshape(-1,d).T@dh.reshape(-1,d);dc=dh@p['O'].T;da=dc@v.transpose(0,2,1);dv=a.transpose(0,2,1)@dc;ds=a*(da-(da*a).sum(-1,keepdims=True))/np.sqrt(d);dq=ds@k;dk=ds.transpose(0,2,1)@q
   dz=dh.copy()
   for name,delta in [('Q',dq),('K',dk),('W',dv)]:g[name]=z.reshape(-1,d).T@delta.reshape(-1,d);dz+=delta@p[name].T
  else:
   dz=np.zeros_like(z);g['R']=np.zeros_like(p['R']);future=np.zeros((b,d))
   for i in range(t-1,-1,-1):
    dd=(do[:,i]+future)*(1-out[:,i]**2);dz[:,i]=dd
    if i:g['R']+=out[:,i-1].T@dd
    future=dd@p['R'].T
  g['E']=np.zeros_like(p['E']);np.add.at(g['E'],x,dz)
  return float(loss),g

def check():
 rng=np.random.default_rng(9);x=rng.integers(V,size=(2,5));y=rng.integers(V,size=(2,5));errors={}
 for kind in ('attention','recurrent'):
  m=LM(kind,4,1);_,g=m.loss(x,y);worst=0
  for name,p in m.p.items():
   for _ in range(3):
    ix=tuple(rng.integers(n) for n in p.shape);old=p[ix];eps=1e-5;p[ix]=old+eps;hi=m.loss(x,y,False);p[ix]=old-eps;lo=m.loss(x,y,False);p[ix]=old;num=(hi-lo)/(2*eps);worst=max(worst,abs(num-g[name][ix]));assert abs(num-g[name][ix])<1e-7,(kind,name,num,g[name][ix])
  errors[kind]=worst
 return errors

def run(kind,d,seed,steps=400):
 m=LM(kind,d,seed);r=np.random.default_rng(100+seed);tr=encode(TRAIN);va=encode(VALID);batch=8;seq=32;ms={k:np.zeros_like(v) for k,v in m.p.items()};vs={k:np.zeros_like(v) for k,v in m.p.items()}
 def evaluate():
  losses=[]
  for start in range(0, min(len(va)-seq-1,8192),batch*seq):
   idx=start+np.arange(batch)[:,None]*seq+np.arange(seq);losses.append(m.loss(va[idx],va[idx+1],False))
  return float(np.mean(losses))
 initial=evaluate();start=time.perf_counter();history=[]
 for step in range(1,steps+1):
  starts=r.integers(len(tr)-seq-1,size=batch);ix=starts[:,None]+np.arange(seq);loss,g=m.loss(tr[ix],tr[ix+1]);norm=np.sqrt(sum(np.sum(a*a) for a in g.values()));scale=min(1,1/(norm+1e-12))
  for k in m.p:
   a=g[k]*scale;ms[k]=.9*ms[k]+.1*a;vs[k]=.999*vs[k]+.001*a*a;m.p[k]-=.003*(ms[k]/(1-.9**step))/(np.sqrt(vs[k]/(1-.999**step))+1e-8)
  if step%100==0:history.append({'step':step,'train_loss':loss})
 seconds=time.perf_counter()-start;start=time.perf_counter();final=evaluate();eval_seconds=time.perf_counter()-start
 latencies=[];context=list(va[:32]);sample=[]
 for _ in range(192):
  tick=time.perf_counter();prob=m.loss(np.array([context[-32:]]),None,False)[0,-1];token=int(np.argmax(prob));context.append(token);latencies.append(time.perf_counter()-tick);sample.append(VOCAB[token])
 return dict(decode_mode='batch1 greedy sliding32 full-window recomputation, no KV/recurrent caching',decode_p50_ms=float(np.median(latencies[2:]))*1000,decode_p95_ms=float(np.quantile(latencies[2:],.95))*1000,greedy_sample=''.join(sample[:120]),kind=kind,width=d,seed=seed,parameters=m.count(),training_tokens=steps*batch*seq,training_seconds=seconds,training_tokens_per_second=steps*batch*seq/seconds,initial_nats=initial,final_nats=final,final_bpc=final/np.log(2),evaluation_seconds=eval_seconds,history=history)
if __name__=='__main__':
 checks=check();target=LM('attention',16,0).count();rd=min(range(1,100),key=lambda d:abs(LM('recurrent',d,0).count()-target));rows=[run(k,d,s) for s in (1,2,3) for k,d in [('attention',16),('recurrent',rd)]]
 result=dict(classification='actual tiny neural LM training on synthetic English templates; CPU only',gradient_max_abs_errors=checks,vocab=V,train_sha256=hashlib.sha256(TRAIN.encode()).hexdigest(),validation_sha256=hashlib.sha256(VALID.encode()).hexdigest(),numpy=np.__version__,rows=rows)
 Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
