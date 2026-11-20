from pathlib import Path
import sys,math,time,json,resource,os
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'sakana_long_text_20261120'))
from run import *
from torch.utils.checkpoint import checkpoint
S=400; C=48
KINDS=('full','exact_recompute','detach_prefix')
DOM=('wiki','shakespeare','marlowe')
def emb(m,x,a=0):return m.emb(x)+m.pos[a:a+x.shape[1]]
def head(m,x):return m.head(m.norm(x))
@torch.no_grad()
def freeze_kv(m,x):
 h=emb(m,x);f=[]
 for b in m.blocks:
  q,k,v=b.project(h)
  f.append((k.detach().contiguous(),v.detach().contiguous()))
  h=b.residual(h,F.scaled_dot_product_attention(q,k,v,is_causal=True))
 return f
def exact_reg(m,x):
 h=emb(m,x)
 for b in m.blocks:h=checkpoint(b,h,use_reentrant=False,preserve_rng_state=False)
 return head(m,h)
def chunk(m,x,a,b,cache):
 h=emb(m,x[:,a:b],a)
 mask=torch.arange(b)[None,:]<=torch.arange(a,b)[:,None]
 for mod,(ko,vo) in zip(m.blocks,cache):
  q,k,v=mod.project(h)
  k=torch.cat((ko[:,:,:a],k),dim=2);v=torch.cat((vo[:,:,:a],v),dim=2)
  z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
  h=mod.residual(h,z)
 return head(m,h)
def grad(m,x,y,k,probe=False):
 static_bytes=sum(p.numel()*p.element_size() for p in m.parameters())
 pkeys={p.untyped_storage().data_ptr() for p in m.parameters()}
 byte=[];sumloss=0.;resident=0
 def one(f,y,scale=1.):
  seen={}
  def pack(t):
   r=t.untyped_storage();key=r.data_ptr()
   if key not in pkeys:seen[key]=max(seen.get(key,0),r.nbytes())
   return t
  def unpack(t):return t
  if probe:
   with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
    a=f();loss=F.cross_entropy(a.transpose(1,2),y)*scale
   byte.append(sum(seen.values()))
  else:a=f();loss=F.cross_entropy(a.transpose(1,2),y)*scale
  loss.backward()
  return loss.item()
 if k in ('full','exact_recompute'):
  sumloss=one((lambda:m(x)) if k=='full' else (lambda:exact_reg(m,x)),y)
 else:
  c=freeze_kv(m,x)
  resident=sum(u.numel()*u.element_size() for z in c for u in z)
  for a in range(0,x.shape[1],C):
   b=min(a+C,x.shape[1])
   sumloss+=one(lambda:chunk(m,x,a,b,c),y[:,a:b],(b-a)/x.shape[1])
 return sumloss,{'peak_saved_nonparameter_tensor_storage_bytes':max(byte,default=0),'resident_detached_kv_bytes':resident,'proxy_saved_plus_cache_bytes':max(byte,default=0)+resident,'parameter_bytes_excluded':static_bytes}
def checks(m,x,y):
 m.zero_grad(set_to_none=True)
 cache=freeze_kv(m,x)
 arr=torch.cat([chunk(m,x,a,min(a+C,x.shape[1]),cache) for a in range(0,x.shape[1],C)],1)
 clean=m(x);diff=float((arr-clean).abs().max().detach());torch.testing.assert_close(clean,arr,rtol=4e-5,atol=4e-6)
 xp=x.clone();xp[:,77:]=(xp[:,77:]+1)%256
 ac=freeze_kv(m,xp);arrp=torch.cat([chunk(m,xp,a,min(a+C,x.shape[1]),ac) for a in range(0,x.shape[1],C)],1)
 torch.testing.assert_close(arr[:,:77],arrp[:,:77],rtol=4e-5,atol=4e-6)
 values={}
 for k in KINDS:
  m.zero_grad(set_to_none=True);l,_=grad(m,x,y,k)
  values[k]=(l,torch.cat([p.grad.flatten() for p in m.parameters()]))
 m.zero_grad(set_to_none=True)
 assert abs(values['exact_recompute'][0]-values['full'][0])<1e-6
 torch.testing.assert_close(values['exact_recompute'][1],values['full'][1],atol=1e-7,rtol=1e-6)
 assert abs(values['detach_prefix'][0]-values['full'][0])<2e-6
 assert not torch.allclose(values['detach_prefix'][1],values['full'][1])
 return {'initial_forward_max_logit_diff_detach_vs_full':diff,'exact_recompute_gradient_max_difference':float((values['exact_recompute'][1]-values['full'][1]).abs().max()),'detach_vs_full_gradient_norm_ratio':float(values['detach_prefix'][1].norm()/values['full'][1].norm()),'detach_gradient_relative_difference_l2':float((values['detach_prefix'][1]-values['full'][1]).norm()/values['full'][1].norm()),'future_mask_pass':True}
def train(k,seed,tr,dv,ts):
 torch.manual_seed(seed);m=Model('ar'); opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
 rng=np.random.default_rng(seed)
 idx=rng.integers(0,len(tr)-N,size=(S,B))
 x,y=grab(tr,idx[0]); admission=checks(m,x,y)
 t=time.perf_counter();timing=0.;dev=[];mem={}
 for j,ix in enumerate(idx):
  x,y=grab(tr,ix);opt.zero_grad(set_to_none=True)
  for group in opt.param_groups:group['lr']=.002*min(1.,(j+1)/30)*(.2+.8*(1+math.cos(math.pi*j/S))/2)
  s=time.perf_counter();loss,p=grad(m,x,y,k,probe=j==3)
  nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();timing+=time.perf_counter()-s
  if j==3:mem=p
  if (j+1)%100==0:
   with torch.no_grad():row=measure(m,dv,222)
   dev.append(dict(step=j+1,bpb=row['bpb']));print(k,seed,j+1,round(row['bpb'],4),round(time.perf_counter()-t,2),flush=True)
 ck=OUT/f'{k}_{seed}.pt';torch.save(m.state_dict(),ck)
 re=Model('ar');re.load_state_dict(torch.load(ck,weights_only=True));re.eval()
 with torch.no_grad():
  assert torch.equal(m(dv[0][:2]),re(dv[0][:2]))
 assert_causal(re,dv[0][:2])
 return dict(kind=k,seed=seed,steps=S,batch=B,context=N,targets=S*B*N,common_input_sha256=hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest(),params=sum(p.numel() for p in m.parameters()),checkpoint=ck.name,sha256=sha(ck),admission=admission,measured_train_step_time_s=timing,peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,activation_proxy=mem,dev=dev,test={q:measure(re,z,3371) for q,z in ts.items()},checkpoint_readback_bitexact=True)
if __name__=='__main__':
 k=sys.argv[1];assert k in KINDS
 tr=read('train_524288.bin');dv=evalsets(read('dev.bin'));ts={p:evalsets(read(p+'.bin')) for p in DOM}
 ar=[]
 for seed in SEEDS:
  ar.append(train(k,seed,tr,dv,ts));(OUT/f'{k}.json').write_text(json.dumps(ar,indent=2)+'\n')
