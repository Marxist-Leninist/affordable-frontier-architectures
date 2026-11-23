"""Small 2023 original BitNet W1A8 equation study, not BitNet b1.58 or native binary speed."""
import argparse,math,statistics,time,json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import run as p
D=192;H=4;L=4;T=128;B=16;SEEDS=(1733,1741);STEPS=2400;WARM=45;PEAK=.0024;EVAL=300
ARMS=('FP32','BitNet-v1')

def q8(x,per_token,positive=False):
    x0=x.detach()
    axes=(-1,) if per_token else tuple(range(x.ndim))
    v=x0.abs().amax(dim=axes,keepdim=True)
    scale=torch.where(v>0,v/128.,1.)
    if positive:
        off=x0.amin(dim=axes,keepdim=True)
        q=((x0-off)/scale).round().clamp(0,127)
        xx=q*scale+off
    else:
        q=(x0/scale).round().clamp(-127,127)
        xx=q*scale
    return (x-x0)+xx, q, scale
def effective(w):
    raw=w.detach(); center=raw.mean()
    b=raw.abs().mean()
    s=torch.where(raw>center,1.,-1.)
    return (w-raw)+s*b,s,b

class Proj(nn.Module):
 def __init__(self,d,f,bit):
  super().__init__();self.n=nn.LayerNorm(d)
  self.weight=nn.Parameter(torch.empty(f,d));nn.init.normal_(self.weight,std=d**-.5)
  self.bit=bit;self.exported=None
 def forward(self,x):
  x=self.n(x)
  if not self.bit:return F.linear(x,self.weight)
  xx,_,_=q8(x,not self.training)
  return F.linear(xx,effective(self.weight)[0] if self.exported is None else self.exported)
class Block(nn.Module):
 def __init__(self,bit):
  super().__init__();self.q=Proj(D,D*3,bit);self.a=Proj(D,D,bit)
  self.f=Proj(D,D*4,bit);self.o=Proj(D*4,D,bit);self.bit=bit
 def forward(self,x):
  q,k,v=(z.unflatten(-1,(H,D//H)).transpose(1,2) for z in self.q(x).chunk(3,-1))
  u=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).flatten(-2)
  h=x+self.a(u)
  z=self.f(h)
  if self.bit:z=q8(z,not self.training,True)[0]
  return h+self.o(F.gelu(z))
class Net(nn.Module):
 def __init__(self,bit):
  super().__init__();self.emb=nn.Embedding(256,D);self.pos=nn.Embedding(T,D)
  self.layers=nn.ModuleList(Block(bit) for _ in range(L));self.n=nn.LayerNorm(D);self.head=nn.Linear(D,256,bias=False);self.bit=bit
  nn.init.normal_(self.emb.weight,std=.02);nn.init.normal_(self.pos.weight,std=.02);nn.init.normal_(self.head.weight,std=.02)
 def forward(self,x):
  h=self.emb(x)+self.pos(torch.arange(x.shape[1],device=x.device))
  for layer in self.layers:h=layer(h)
  return self.head(self.n(h))
def opt(m,dev):
 return torch.optim.AdamW(m.parameters(),lr=PEAK,betas=(.9,.98),eps=1.e-8,weight_decay=.01,fused=(dev=='cuda'))
def loss(m,x,y):return F.cross_entropy(m(x).transpose(1,2),y)
def trainstep(m,x,y,o,lr):
 for g in o.param_groups:g['lr']=lr
 o.zero_grad(set_to_none=True); l=loss(m,x,y);l.backward();o.step();o.zero_grad(set_to_none=True)
 return l.detach()
def cfg(u):
 if u<WARM:return PEAK*(u+1)/WARM
 return PEAK*(.10+.90*(1-(u-WARM)/(STEPS-1-WARM)))
def bytes_state(m,o):
 return dict(master_B=sum(x.untyped_storage().nbytes() for x in m.state_dict().values()),
 opt_B=sum(v.untyped_storage().nbytes() for s in o.state.values() for v in s.values() if isinstance(v,torch.Tensor)))
def pick(n,k):
 q=p.offsets(n,k);z=q.copy();z[:-1]+=12*(T+1);z[-1]-=12*(T+1)
 for f in (0,2,4,6,8,10):
  r=q.copy();r[:-1]+=f*(T+1);r[-1]-=f*(T+1);assert np.abs(z[:,None]-r[None,:]).min()>T
 assert np.diff(z).min()>T and z.min()>=0 and z.max()+T<n
 return z
@torch.inference_mode()
def evaluate(m,tape,i):
 m.eval();rows=[]
 for ii in np.array_split(i,len(i)//B):
  x,y=p.pull(tape,ii);z=m(x)
  rows.extend((F.cross_entropy(z.transpose(1,2),y,reduction='none').mean(-1).double()/math.log(2)).cpu().tolist())
 return rows
@torch.inference_mode()
def packed_export(m,path,x):
 m.eval()
 st=m.state_dict()
 names={k for k,v in m.named_modules() if isinstance(v,Proj)}
 remove={x+'.weight' for x in names} if m.bit else set()
 non={k:v.cpu() for k,v in st.items() if k not in remove}
 if not m.bit:return dict(param_bytes=sum(v.untyped_storage().nbytes() for v in non.values()),method='FP32 tensors, no packed artifact')
 signs={};sc={}
 for k,v in m.named_modules():
  if k not in names:continue
  _,s,b=effective(v.weight)
  signs[k]=p.bits_pack(s>0).cpu();sc[k]=b.reshape(1).cpu()
 p.atomic(path,lambda f:torch.save(dict(shape_by_tensor={k:list(st[k+'.weight'].shape) for k in signs},sign_pack=signs,beta=sc,non_binary=non),f))
 r=torch.load(path,map_location='cpu',weights_only=True)
 before=m(x)
 for k,v in m.named_modules():
  if k not in names:continue
  s=p.bits_open(r['sign_pack'][k]).to(x.device,dtype=torch.float32)*2.-1.
  b=r['beta'][k].to(x.device)
  v.exported=s.view_as(v.weight)*b
 assert torch.equal(before,m(x))
 for v in m.modules():
  if isinstance(v,Proj):v.exported=None
 return dict(method='bit-packed projection signs and FP32 scales/high-precision remainder, fake dequantized replay only',
   param_bytes=sum(v.numel()*v.element_size() for v in list(non.values())+list(sc.values())+list(signs.values())),
   onebit_B=sum(w.numel() for w in signs.values()),scale_B=sum(w.numel()*w.element_size() for w in sc.values()),
   fp32_remainder_B=sum(w.numel()*w.element_size() for w in non.values()),file_B=path.stat().st_size,sha=p.digest(path.read_bytes()))
@torch.inference_mode()
def timed(m,tape,i):
 m.eval()
 x,_=p.pull(tape,i[:1]);ts=[]
 for _ in range(4):m(x)
 for _ in range(21):
  torch.cuda.synchronize();t=time.perf_counter();m(x);torch.cuda.synchronize();ts.append((time.perf_counter()-t)*1000)
 return dict(ref_p50_ms=statistics.median(ts),ref_p95_ms=sorted(ts)[19],native_binary_kernel=False)

def qa(dev):
 torch.manual_seed(714)
 w=torch.tensor([[3.,3.,5.,1.],[5.,1.,3.,3.]],requires_grad=True,device=dev)
 wq,s,b=effective(w)
 assert b.item()==3. and torch.equal(s,torch.tensor([[-1.,-1.,1.,-1.],[1.,-1.,-1.,-1.]],device=dev))
 x=torch.tensor([[2.,-1.,3.,-2.],[1.,-1.,1.,-1.]],requires_grad=True,device=dev)
 n=F.layer_norm(x,(4,))
 a,q,h=q8(n,True)
 exact=(q.int()[:,None,:]*s.int()[None,:,:]).sum(-1).float()*h*b
 assert torch.allclose(F.linear(a,wq),exact,atol=2e-6)
 gw,gn=torch.autograd.grad((wq.sum()+a.sum()),(w,n))
 assert torch.equal(gw,torch.ones_like(w)) and torch.equal(gn,torch.ones_like(n))
 stat={}
 for bit in (False,True):
  torch.manual_seed(754);m=Net(bit).to(dev);o=opt(m,dev)
  tt=torch.tensor([[1,7,1,3,44,18,12,9],[5,7,3,8,2,9,6,90]],device=dev)
  m.train()
  trainstep(m,tt[:,:-1],tt[:,1:],o,.0003)
  from io import BytesIO
  f=BytesIO();torch.save(dict(net=m.state_dict(),opt=o.state_dict()),f);f.seek(0)
  ch=torch.load(f,map_location=dev,weights_only=True)
  r=Net(bit).to(dev);r.load_state_dict(ch['net'],strict=True);oo=opt(r,dev);oo.load_state_dict(ch['opt'])
  trainstep(m,tt[:,:-1],tt[:,1:],o,.0003);trainstep(r,tt[:,:-1],tt[:,1:],oo,.0003)
  assert all(torch.equal(a,r.state_dict()[k]) for k,a in m.state_dict().items())
  m.eval()
  with torch.no_grad():
   a=m(tt);u=tt.clone();u[:,4:]=u[:,4:]+1
   assert torch.equal(a[:,:4],m(u)[:,:4])
  stat[str(bit)]=dict(par=sum(x.numel() for x in m.parameters()),restore_next_update_exact=True)
 assert stat['True']['par']==stat['False']['par']
 return dict(signed_projection_integer_ref_max_abs=float((F.linear(q8(n,True)[0],wq)-exact).abs().max().detach()),uncentered_beta=3.,
   identity_ste=True,suffix_causality_eval=True,full_training_restore=stat)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--qa-cpu',action='store_true');ap.add_argument('--data',type=Path);a=ap.parse_args();a.out.mkdir(exist_ok=True)
 torch.set_num_threads(1)
 if a.qa_cpu:p.save(a.out/'cpu-qa.json',qa('cpu'));return
 assert a.data and torch.cuda.is_available()
 torch.cuda.set_per_process_memory_fraction(.55,0);torch.backends.cuda.matmul.allow_tf32=False
 start=time.monotonic();dev='cuda';raw=p.load_data(a.data)
 ix={k:pick(len(raw[k]),n) for k,n in [('validation',128),('test',256)]}
 ws=[raw[k][int(i):int(i)+T+1] for k,ii in ix.items() for i in ii]
 assert len(ws)==len(set(ws)) and not any(v in raw['train'] for v in ws)
 tape={k:torch.from_numpy(np.frombuffer(v,np.uint8).copy()).cuda() for k,v in raw.items()}
 p.save(a.out/'preflight.json',dict(qa=qa(dev),data=p.HASH,model=dict(width=D,layers=L,heads=H),torch=torch.__version__,
  offsets={k:p.digest(v.astype('<u8').tobytes()) for k,v in ix.items()},seed=SEEDS,steps=STEPS,peak_lr=PEAK,warm=WARM))
 frozen={}
 for j,seed in enumerate(SEEDS):
  train=np.random.Generator(np.random.PCG64(seed+57000)).integers(0,len(raw['train'])-T,size=(STEPS,B))
  for arm in ARMS if j==0 else ARMS[::-1]:
   torch.manual_seed(seed); m=Net(arm=='BitNet-v1').cuda();o=opt(m,dev)
   curve=[]; fit=0.;pk=0;p.sync(dev);t0=time.perf_counter()
   for u in range(STEPS+1):
    if u%EVAL==0:
     p.sync(dev)
     if u:fit+=time.perf_counter()-t0;pk=max(pk,torch.cuda.max_memory_allocated())
     e=statistics.mean(evaluate(m,tape['validation'],ix['validation']))
     row=dict(step=u,fit_s=fit,dev_bpb=e,targets=u*B*T)
     curve.append(row);p.save(a.out/f'{arm}-{seed}-curve.json',curve);print(arm,seed,row,flush=True)
     m.train();p.sync(dev);torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    if u==STEPS:break
    x,y=p.pull(tape['train'],train[u]);r=trainstep(m,x,y,o,cfg(u))
    if u%EVAL==EVAL-1:assert bool(torch.isfinite(r))
   ck=a.out/f'{arm}-{seed}.pt';p.atomic(ck,lambda f:torch.save(dict(net=m.state_dict(),optimizer_retained=False),f))
   state=torch.load(ck,map_location='cpu',weights_only=True)
   assert all(torch.equal(v.detach().cpu(),state['net'][k]) for k,v in m.state_dict().items())
   actual=bytes_state(m,o); x,_=p.pull(tape['validation'],ix['validation'][:2])
   packed=packed_export(m,a.out/f'{arm}-{seed}-inference.pt',x)
   frozen[(arm,seed)]=dict(arm=arm,seed=seed,checkpoint_sha=p.digest(ck.read_bytes()),params=sum(t.numel() for t in m.parameters()),
      cuda_fit_peak_B=pk,curve=curve,real_training_state=actual,inference_payload=packed,train_offsets_sha=p.digest(train.astype('<u8').tobytes()))
   del r,o,m,state;torch.cuda.empty_cache()
 p.save(a.out/'FIT-FROZEN.json',list(frozen.values()))
 for (arm,seed),r in frozen.items():
  ck=a.out/f'{arm}-{seed}.pt';assert p.digest(ck.read_bytes())==r['checkpoint_sha']
  torch.manual_seed(seed);m=Net(arm=='BitNet-v1').cuda()
  initial=statistics.mean(evaluate(m,tape['test'],ix['test']))
  m.load_state_dict(torch.load(ck,map_location=dev,weights_only=True)['net'],strict=True)
  final=evaluate(m,tape['test'],ix['test'])
  p.save(a.out/f'{arm}-{seed}.json',dict(r,test_bpb=statistics.mean(final),test_bpb_by_window=final,initial_bpb=initial,**timed(m,tape['test'],ix['test'])))
 p.save(a.out/'DONE.json',dict(arms=ARMS,seeds=SEEDS,steps=STEPS,elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':main()
