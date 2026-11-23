"""Small clean-room 2025 BitNet-b1.58 schedule and architecture study."""
import argparse,math,statistics,time,json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import run as p
from model import Net,Linear,ternary,act8,D,H,L,T,M,H_KV
B=16;SEEDS=(1733,1741);STEPS=2400;WARM=45;PEAK=.0024;EVAL=300; MID=1200
ARMS=('FP32','BitNet-b1.58')

def opt(m,dev):
 return torch.optim.AdamW(m.parameters(),lr=PEAK,betas=(.9,.98),eps=1.e-8,weight_decay=.01,fused=(dev=='cuda'))
def loss(m,x,y):return F.cross_entropy(m(x).transpose(1,2),y)
def trainstep(m,x,y,o,lr):
 for g in o.param_groups:
  g['lr']=lr[0]
  g['weight_decay']=lr[1]
 o.zero_grad(set_to_none=True); l=loss(m,x,y);l.backward();o.step();o.zero_grad(set_to_none=True)
 return l.detach()
def cfg(u):
 if u<WARM:return (PEAK*(u+1)/WARM,.1)
 if u<MID:
  k=(u-WARM)/(MID-1-WARM);v=.5*(1.+math.cos(math.pi*k))
  return (PEAK*(.25+.75*v),.01+.09*v)
 k=(u-MID)/(STEPS-1-MID);v=.5*(1.+math.cos(math.pi*k))
 return (PEAK*(.02+.08*v),0.)

def train_split(raw,seed):
 rng=np.random.Generator(np.random.PCG64(seed+79022))
 data=np.frombuffer(raw,np.uint8); n=len(data)//(T+1);rows=data[:n*(T+1)].reshape(n,T+1)
 letter=((rows>=65)&(rows<=90))|((rows>=97)&(rows<=122))
 prose=(letter.sum(-1)>=72)&((rows==10).sum(-1)<=3)
 general=rng.choice(n,MID*B,replace=False)
 prose[general]=False
 eligible=np.flatnonzero(prose)
 assert len(eligible)>=MID*B
 focused=rng.choice(eligible,(STEPS-MID)*B,replace=False)
 assert np.intersect1d(general,focused).size==0
 return np.r_[general,focused].reshape(STEPS,B)*(T+1),int(len(eligible))

def bytes_state(m,o):
 return dict(master_B=sum(x.untyped_storage().nbytes() for x in m.state_dict().values()),
 opt_B=sum(v.untyped_storage().nbytes() for s in o.state.values() for v in s.values() if isinstance(v,torch.Tensor)))
def pick_holdout(raw,name,count,seen):
 data=raw[name];piv=p.offsets(len(data),count)
 old=np.concatenate([np.r_[piv[:-1]+f*(T+1),piv[-1]-f*(T+1)] for f in (0,2,4,6,8,10,12)])
 nom=np.r_[piv[:-1]+14*(T+1),piv[-1]-14*(T+1)]
 st=[]; rejects={'outside_or_prior_overlap':0,'duplicate_within_eval':0,'exact_train_match':0}
 # Frozen rule: nominal, then forward/backward in one whole-window strides.
 scan=[0]+[v for r in range(1,33) for v in (r,-r)]
 for center in nom:
  for o in (int(center)+d*(T+1) for d in scan):
   if not (0<=o<len(data)-T) or np.any(np.abs(old-o)<=T) or any(abs(k-o)<=T for k in st):
    rejects['outside_or_prior_overlap']+=1;continue
   part=data[o:o+T+1]
   if part in seen:rejects['duplicate_within_eval']+=1;continue
   if part in raw['train']:rejects['exact_train_match']+=1;continue
   st.append(o);seen.add(part);break
  else:raise AssertionError((name,'no eligible window',int(center)))
 return np.sort(np.array(st,dtype=np.int64)),rejects
@torch.inference_mode()
def evaluate(m,tape,i):
 m.eval();rows=[]
 for ii in np.array_split(i,len(i)//B):
  x,y=p.pull(tape,ii);z=m(x)
  rows.extend((F.cross_entropy(z.transpose(1,2),y,reduction='none').mean(-1).double()/math.log(2)).cpu().tolist())
 return rows
def pack2(c):
 u=(c.flatten().to(torch.int16)+1).view(-1,4)
 return (u<<(2*torch.arange(4,device=u.device))).sum(-1).byte()
def open2(z):
 return ((z.short()[:,None]>>(2*torch.arange(4,device=z.device)))&3).flatten().float()-1.
@torch.inference_mode()
def packed_export(m,path,x):
 m.eval()
 st=m.state_dict();names={k for k,v in m.named_modules() if isinstance(v,Linear)}
 remove={k+'.weight' for k in names} if m.bit else set()
 non={k:v.cpu() for k,v in st.items() if k not in remove}
 if not m.bit:return dict(param_bytes=sum(v.numel()*v.element_size() for v in non.values()),method='FP32 tensors')
 codes={};sc={}
 for k,v in m.named_modules():
  if k not in names:continue
  _,c,b=ternary(v.weight)
  codes[k]=pack2(c).cpu();sc[k]=b.reciprocal().reshape(1).cpu()
 p.atomic(path,lambda f:torch.save(dict(shape_by_tensor={k:list(st[k+'.weight'].shape) for k in codes},
       ternary_pack=codes,inv_scale=sc,non_quantized=non),f))
 r=torch.load(path,map_location='cpu',weights_only=True)
 before=m(x)
 for k,v in m.named_modules():
  if k not in names:continue
  c=open2(r['ternary_pack'][k]).to(x.device)
  assert bool(((c>=-1)&(c<=1)).all())
  v.frozen=c.view_as(v.weight)/r['inv_scale'][k].to(x.device)
 assert torch.equal(before,m(x))
 for v in m.modules():
  if isinstance(v,Linear):v.frozen=None
 return dict(method='clean-room 2-bit code ternary projections plus FP32 scales and remaining tensors; not official Microsoft file',
  param_bytes=sum(v.numel()*v.element_size() for v in list(non.values())+list(sc.values())+list(codes.values())),
  code_B=sum(w.numel() for w in codes.values()),scale_B=sum(w.numel()*w.element_size() for w in sc.values()),
  fp32_remainder_B=sum(w.numel()*w.element_size() for w in non.values()),file_B=path.stat().st_size,sha=p.digest(path.read_bytes()))

def qa(dev):
 torch.manual_seed(714)
 w=torch.tensor([[-2.,0.,2.,4.],[0.,-4.,2.,2.]],requires_grad=True,device=dev)
 wq,s,b=ternary(w)
 assert b.item()==2. and torch.equal(s,torch.tensor([[-1.,0.,1.,1.],[0.,-1.,1.,1.]],device=dev))
 assert torch.equal(open2(pack2(s)),s.flatten())
 x=torch.tensor([[2.,-1.,3.,-2.],[1.,-1.,1.,-1.]],requires_grad=True,device=dev)
 a,q,h=act8(x)
 exact=(q.int()[:,None,:]*s.int()[None,:,:]).sum(-1).float()/h/b.reciprocal()
 ee=float((F.linear(a,wq)-exact).abs().max().detach())
 assert torch.allclose(F.linear(a,wq),exact,atol=2e-6)
 gw,gx=torch.autograd.grad((wq.sum()+a.sum()),(w,x))
 assert torch.equal(gw,torch.ones_like(w)) and torch.equal(gx,torch.ones_like(x))
 stat={}
 for bit in (False,True):
  torch.manual_seed(754);m=Net(bit).to(dev);o=opt(m,dev)
  tt=torch.tensor([[1,7,1,3,44,18,12,9],[5,7,3,8,2,9,6,90]],device=dev)
  m.train()
  trainstep(m,tt[:,:-1],tt[:,1:],o,(.0003,.01))
  from io import BytesIO
  f=BytesIO();torch.save(dict(net=m.state_dict(),opt=o.state_dict()),f);f.seek(0)
  ch=torch.load(f,map_location=dev,weights_only=True)
  r=Net(bit).to(dev);r.load_state_dict(ch['net'],strict=True);oo=opt(r,dev);oo.load_state_dict(ch['opt'])
  trainstep(m,tt[:,:-1],tt[:,1:],o,(.0003,.01));trainstep(r,tt[:,:-1],tt[:,1:],oo,(.0003,.01))
  assert all(torch.equal(a,r.state_dict()[k]) for k,a in m.state_dict().items())
  m.eval()
  with torch.no_grad():
   a=m(tt);u=tt.clone();u[:,4:]=u[:,4:]+1
   assert torch.equal(a[:,:4],m(u)[:,:4])
  stat[str(bit)]=dict(par=sum(x.numel() for x in m.parameters()),restore_next_update_exact=True)
 assert stat['True']['par']==stat['False']['par']
 return dict(ternary_projection_int_reference_max_abs=ee,absmean_beta=2.,
   identity_ste=True,suffix_causality_eval=True,full_training_restore=stat)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--qa-cpu',action='store_true');ap.add_argument('--data',type=Path);a=ap.parse_args();a.out.mkdir(exist_ok=True)
 torch.set_num_threads(1)
 if a.qa_cpu:p.save(a.out/'cpu-qa.json',qa('cpu'));return
 assert a.data and torch.cuda.is_available()
 torch.cuda.set_per_process_memory_fraction(.55,0);torch.backends.cuda.matmul.allow_tf32=False
 start=time.monotonic();dev='cuda';raw=p.load_data(a.data)
 seen=set(); picked={k:pick_holdout(raw,k,n,seen) for k,n in [('validation',128),('test',256)]}
 ix={k:r[0] for k,r in picked.items()}
 ws=[raw[k][int(i):int(i)+T+1] for k,ii in ix.items() for i in ii]
 assert len(ws)==len(set(ws)) and not any(v in raw['train'] for v in ws)
 tape={k:torch.from_numpy(np.frombuffer(v,np.uint8).copy()).cuda() for k,v in raw.items()}
 p.save(a.out/'preflight.json',dict(qa=qa(dev),data=p.HASH,model=dict(width=D,layers=L,q_heads=H,kv_heads=H_KV,ff_width=M),torch=torch.__version__,
  prescore_holdout_exclusions={k:r[1] for k,r in picked.items()},offsets={k:p.digest(v.astype('<u8').tobytes()) for k,v in ix.items()},seed=SEEDS,steps=STEPS,peak_lr=PEAK,warm=WARM,transition=cfg(MID),before_transition=cfg(MID-1)))
 frozen={}
 for j,seed in enumerate(SEEDS):
  train,prose_pool=train_split(raw['train'],seed)
  for arm in ARMS if j==0 else ARMS[::-1]:
   torch.manual_seed(seed); m=Net(arm=='BitNet-b1.58').cuda();o=opt(m,dev)
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
      cuda_fit_peak_B=pk,curve=curve,real_training_state=actual,inference_payload=packed,stage2_proxy_eligible_windows=prose_pool,train_offsets_sha=p.digest(train.astype('<u8').tobytes()))
   del r,o,m,state;torch.cuda.empty_cache()
 p.save(a.out/'FIT-FROZEN.json',list(frozen.values()))
 for (arm,seed),r in frozen.items():
  ck=a.out/f'{arm}-{seed}.pt';assert p.digest(ck.read_bytes())==r['checkpoint_sha']
  torch.manual_seed(seed);m=Net(arm=='BitNet-b1.58').cuda()
  initial=statistics.mean(evaluate(m,tape['test'],ix['test']))
  m.load_state_dict(torch.load(ck,map_location=dev,weights_only=True)['net'],strict=True)
  final=evaluate(m,tape['test'],ix['test'])
  p.save(a.out/f'{arm}-{seed}.json',dict(r,test_bpb=statistics.mean(final),test_bpb_by_window=final,initial_bpb=initial,native_inference_speed_measured=False))
 p.save(a.out/'DONE.json',dict(arms=ARMS,seeds=SEEDS,steps=STEPS,elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':main()
