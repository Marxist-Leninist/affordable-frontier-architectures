import os,sys,json,math,time,hashlib,resource
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'p044_counts_scaling/data'
N=192; B=6; STEPS=640; W=48; SEEDS=(17,29,43)
SIG=(2.,.85,.05)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return torch.from_numpy(np.fromfile(DATA/name,dtype=np.uint8)).long()
def grab(data,starts):
 a=data[torch.from_numpy(np.array(starts))[:,None]+torch.arange(N+1)]
 return a[:,:-1].contiguous(),a[:,1:].contiguous()
def evalsets(data):
 # disjoint target blocks separated by 1 byte; context begins at start of each block
 return grab(data,np.arange(0,len(data)-N,N+1))
def blockeval(x,y):return zip(x.split(B),y.split(B))
class Block(nn.Module):
 def __init__(self):
  super().__init__(); self.la=nn.LayerNorm(W);self.lb=nn.LayerNorm(W);self.qkv=nn.Linear(W,W*3);self.o=nn.Linear(W,W)
  self.ff=nn.Sequential(nn.Linear(W,W*3),nn.GELU(),nn.Linear(W*3,W))
 def project(self,h):
  b,t,w=h.shape
  return [v.view(b,t,4,W//4).transpose(1,2) for v in self.qkv(self.la(h)).chunk(3,-1)]
 def residual(self,h,a):
  h=h+self.o(a.transpose(1,2).contiguous().view_as(h));return h+self.ff(self.lb(h))
 def forward(self,h):
  q,k,v=self.project(h);return self.residual(h,F.scaled_dot_product_attention(q,k,v,is_causal=True))
 def cross(self,h,c,mask):
  q,k,v=self.project(h); qc,kc,vc=self.project(c)
  a=F.scaled_dot_product_attention(q,torch.cat([kc,k],2),torch.cat([vc,v],2),attn_mask=mask)
  return self.residual(h,a)
class Model(nn.Module):
 def __init__(self,kind):
  super().__init__(); self.kind=kind
  self.emb=nn.Embedding(256,W);self.pos=nn.Parameter(torch.randn(N,W)*.02)
  self.blocks=nn.ModuleList(Block() for _ in range(2 if kind in ('ar','db_ar') else 1)) if kind!='gru' else None
  if kind=='gru': self.gru=nn.GRU(W,W,2,batch_first=True)
  if kind.startswith('db'):
   self.zin=nn.Linear(W,W);self.sigma=nn.Sequential(nn.Linear(1,W),nn.SiLU(),nn.Linear(W,W))
   g=torch.Generator().manual_seed(588);self.register_buffer('codes',math.sqrt(W)*F.normalize(torch.randn(256,W,generator=g),dim=-1))
  self.norm=nn.LayerNorm(W);self.head=nn.Linear(W,W if kind.startswith('db') else 256)
  self.register_buffer('mask',torch.cat([torch.ones(N,N,dtype=torch.bool).tril(),torch.eye(N,dtype=torch.bool)],1),persistent=False)
 def forward(self,x):
  c=self.emb(x)+self.pos[:x.size(1)]
  if self.kind=='gru': c=self.gru(c)[0]
  else:
   for b in ([self.blocks[0]]*2 if self.kind=='loop' else self.blocks):c=b(c)
  return self.head(self.norm(c))
 def den(self,x,z,s,i):
  n=x.size(1)
  c=self.emb(x)+self.pos[:n]
  h=self.zin(z/math.sqrt(1+s*s))+self.pos[:n]+self.sigma(z.new_full((1,1,1),math.log(s)))
  m=self.mask[:n,torch.cat([torch.arange(n),N+torch.arange(n)])]
  return self.head(self.norm(self.blocks[i if self.kind=='db_ar' else 0].cross(h,c,m)))
 def approx(self,x,g,draw=4):
  # An operational predictive distribution from prefix-only samples; not diffusion NLL/ELBO.
  out=[]
  for _ in range(draw):
   z=torch.randn((*x.shape,W),generator=g)*SIG[0]
   for i,(s,t) in enumerate(zip(SIG,SIG[1:])):
    d=self.den(x,z,s,i);z=(t/s)*z+(1-t/s)*d
   out.append((F.normalize(z,dim=-1)@(self.codes/math.sqrt(W)).T*12).softmax(-1))
  return torch.stack(out).mean(0).clamp_min(1e-12).log()
@torch.no_grad()
def measure(m,pair,seed):
 g=torch.Generator().manual_seed(seed)
 losses=[];acc=[];last=[]
 for x,y in blockeval(*pair):
  q=m.approx(x,g) if m.kind.startswith('db') else m(x).log_softmax(-1)
  l=-q.gather(-1,y[...,None])[...,0]/math.log(2)
  losses.extend(l.mean(-1).tolist());acc.extend(q.argmax(-1).eq(y).float().mean(-1).tolist());last.extend(l[:,N//2:].mean(-1).tolist())
 return {'bpb':float(np.mean(losses)),'top1':float(np.mean(acc)),'bpb_late_half':float(np.mean(last)),'by_window_bpb':losses,'scored_targets':len(losses)*N}
def assert_causal(m,x):
 xp=x.clone();xp[:,90:]=(xp[:,90:]+1)%256
 if m.kind.startswith('db'):
  z=torch.randn((*x.shape,W),generator=torch.Generator().manual_seed(22));zp=z.clone();zp[:,90:]+=20
  for i in (0,1):
   # Future true targets/noise and past target states in other noisy positions cannot alter this output.
   a=m.den(x,z,.95,i);v=m.den(xp,zp,.95,i);torch.testing.assert_close(a[:,:90],v[:,:90],rtol=1e-5,atol=1e-6)
   zp_past=z.clone();zp_past[:,:80]+=20
   torch.testing.assert_close(a[:,80:],m.den(x,zp_past,.95,i)[:,80:],rtol=1e-5,atol=1e-6)
 else: torch.testing.assert_close(m(x)[:,:90],m(xp)[:,:90],rtol=1e-5,atol=1e-6)
@torch.no_grad()
def generation(m,fixture):
 # From held-out past only; no teacher-forcing inside continuation. Deterministically seeded, all outputs retained hex.
 result=[]
 gen=torch.Generator().manual_seed(937)
 for row in fixture[0][:2]:
  x=row[:64][None,:].clone()
  for _ in range(24):
   v=m.approx(x,gen,2)[0,-1] if m.kind.startswith('db') else m(x)[0,-1].log_softmax(-1)
   x=torch.cat([x,torch.multinomial(v.exp(),1,generator=gen).view(1,1)],-1)
  result.append(bytes(x[0,64:].tolist()).hex())
 return result
def run(kind,seed,train,dev,tests):
 torch.manual_seed(seed)
 t=time.perf_counter()
 m=Model(kind);opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
 rng=np.random.default_rng(seed); noise=torch.Generator().manual_seed(seed+1904); curve=[]
 starts=rng.integers(0,len(train)-N,size=(STEPS,B)); causal_x=dev[0][:2]
 assert_causal(m,causal_x)
 mx=0; bands=[0,0]; primary_device='cpu'; clip=1.
 for i,idx in enumerate(starts):
  x,y=grab(train,idx)
  for g in opt.param_groups:g['lr']=.002*min(1.,(i+1)/35)*(.2+.8*(1+math.cos(math.pi*i/STEPS))/2)
  opt.zero_grad(set_to_none=True)
  if kind.startswith('db'):
   band=i%2;bands[band]+=1
   s=math.exp(math.log(SIG[band+1])+float(torch.rand((),generator=noise))*(math.log(SIG[band])-math.log(SIG[band+1])))
   c=m.codes[y];z=c+s*torch.randn(c.shape,generator=noise)
   loss=F.mse_loss(m.den(x,z,s,band),c)
  else:loss=F.cross_entropy(m(x).flatten(0,1),y.flatten())
  loss.backward()
  if kind=='db_ar':
   assert all(p.grad is None for p in m.blocks[1-band].parameters())
  mx=max(mx,sum(p.numel() for p in m.parameters() if p.grad is not None))
  nn.utils.clip_grad_norm_(m.parameters(),clip)
  opt.step()
  if i+1 in (160,320,480,640):
   with torch.no_grad(): val=measure(m,dev,55)
   curve.append({'step':i+1,'train_objective_last_batch':loss.item(),'dev_bpb':val['bpb']})
   print(f'PROGRESS {kind} {seed} {i+1}/{STEPS} {time.perf_counter()-t:.1f}s dev {val["bpb"]:.4f}',flush=True)
 path=HERE/f'{kind}_{seed}.pt';torch.save(m.state_dict(),path);r=Model(kind);r.load_state_dict(torch.load(path,weights_only=True));r.eval()
 with torch.no_grad():
  f=lambda a: a.approx(causal_x,torch.Generator().manual_seed(999)) if kind.startswith('db') else a(causal_x)
  assert torch.equal(f(m),f(r))
 assert_causal(r,causal_x)
 elapsed=time.perf_counter()-t
 v={k:measure(r,value,3117) for k,value in tests.items()}
 gt=time.perf_counter()
 sampled={k:generation(r,value) for k,value in tests.items()}
 return {'kind':kind,'seed':seed,'steps':STEPS,'batch':B,'context':N,'targets':STEPS*B*N,'curve':curve,'tests':v,'fit_dev_save_wall_s':elapsed,'generation_wall_s':time.perf_counter()-gt,'generation_hex':sampled,'checkpoint':path.name,'checkpoint_sha256':sha(path),'params':sum(p.numel() for p in m.parameters()),'max_active_grad_params':mx,'peak_process_rss_kib_cumulative':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'band_updates':bands,'device':primary_device,'causality_and_reload_pass':True}
if __name__=='__main__':
 k=sys.argv[1]
 train=read('train_524288.bin'); dev=evalsets(read('dev.bin')) 
 tests={x:evalsets(read(x+'.bin')) for x in ['wiki','shakespeare','marlowe']}
 r=[]
 for seed in SEEDS:
  r.append(run(k,seed,train,dev,tests)); (HERE/f'{k}.json').write_text(json.dumps(r,indent=2)+'\n')
