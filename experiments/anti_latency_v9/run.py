"""Exact inference optimization; no weights or architecture changed."""
from experiment import *

class Optimized(Fast):
 def __init__(self,kind,table=True):
  super().__init__(kind,True);self.table=table
 def rope(self,x,offset):
  if not self.table:return super().rope(x,offset)
  length=offset+x.shape[-2];key=(x.device,x.dtype)
  if getattr(self,'_table_key',None)!=key or self._cos.shape[0]<length:
   # Persistent constants must be ordinary tensors even if first built during evaluation.
   with torch.inference_mode(False), torch.no_grad():
    capacity=max(4096,2**(length-1).bit_length());k=x.shape[-1]
    f=10000.**(-torch.arange(0,k,2,device=x.device,dtype=x.dtype)/k)
    ang=torch.arange(capacity,device=x.device,dtype=x.dtype)[:,None]*f
    self._cos,self._sin=ang.cos(),ang.sin();self._table_key=key
  c,s=self._cos[offset:length],self._sin[offset:length];a,b=x[...,::2],x[...,1::2]
  return torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-2)
 def forward(self,x,state=None,cache=False):
  b,t,d=x.shape;q,k,v=[a.reshape(b,t,self.h,d//self.h).transpose(1,2) for a in self.qkv(x).chunk(3,-1)]
  offset=0 if state is None else state[0].shape[-2];q,k=self.rope(q,offset),self.rope(k,offset);newv=v
  if state is not None:k=torch.cat((state[0],k),-2);v=torch.cat((state[1],v),-2)
  if offset==0:a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
  elif t==1:a=F.scaled_dot_product_attention(q,k,v)
  else:
   mask=torch.arange(k.shape[-2],device=x.device)[None,:]<=torch.arange(offset,offset+t,device=x.device)[:,None]
   a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
  total=None
  if self.kind=='attention':z=a
  else:
   total=newv if t==1 else newv.cumsum(-2)
   if state is not None:total=total+state[2]
   if t==1:z=torch.add(a,total,alpha=-1/(offset+1))
   else:
    invn=1/torch.arange(offset+1,offset+t+1,device=x.device,dtype=x.dtype)[None,None,:,None]
    z=torch.addcmul(a,total,invn,value=-1)
  ns=None
  if cache:ns=(k,v) if self.kind=='attention' else (k,v,total if (t==1 and offset>0) else total[:,:,-1:,:].clone())
  return self.out(z.transpose(1,2).reshape(b,t,d)),ns

def build(kind,opt=True,table=True):
 m=make(kind,'fast')
 if opt:
  for block in m.blocks:
   old=block.mix;new=Optimized(kind,table);new.load_state_dict(old.state_dict())
   if kind=='signed':
    with torch.no_grad():new.out.weight.neg_()
   block.mix=new
 return m

def check():
 out=[]
 for kind in ['attention','signed']:
  a=make(kind,'original').double();b=build(kind).double();torch.manual_seed(82);x=torch.randint(256,(2,31));y=a(x);z=b(x)
  err=(y-z).abs().max().item();(y.square().mean()).backward();(z.square().mean()).backward();ge=0
  for (name,p),(n,q) in zip(a.named_parameters(),b.named_parameters()):
   g=-q.grad if kind=='signed' and name.endswith('mix.out.weight') else q.grad
   ge=max(ge,(p.grad-g).abs().max().item())
  assert err<1e-10 and ge<1e-10
  out.append(dict(kind=kind,fp64_logits=err,fp64_gradients=ge))
 with torch.inference_mode():
  for kind in ['attention','signed']:
   a=make(kind,'fast');b=build(kind)
   for n in [128,512,2048,4096]:
    torch.manual_seed(82);x=torch.randint(256,(1,n+8));expected=a(x);_,s=b(x[:,:n],cache=True);outs=[]
    for i in range(n,n+8):y,s=b(x[:,i:i+1],s,cache=True);outs.append(y)
    got=torch.cat(outs,1);e=(got-expected[:,n:]).abs().max().item();assert e<1e-4
    # Non-mutating request states: replay identical input from same state.
    y1,s1=b(x[:,-1:],s,cache=True);y2,s2=b(x[:,-1:],s,cache=True);assert torch.equal(y1,y2)
    out.append(dict(kind=kind,context=n,fp32_cache_logits=e))
 return out

@torch.inference_mode()
def bench():
 ms={'attention_v3':build('attention',False),'signed_v3':build('signed',False),'attention_table':build('attention'),'signed_axpy':build('signed',True,False),'signed_table_axpy':build('signed')};out={};rng=random.Random(810)
 for n in [128,512,2048,4096]:
  torch.manual_seed(3);x=torch.randint(256,(1,n+32));ts={k:[] for k in ms}
  def run(k):
   m=ms[k];t0=time.perf_counter();_,s=m(x[:,:n],cache=True);t1=time.perf_counter()
   for i in range(n,n+32):_,s=m(x[:,i:i+1],s,cache=True)
   t2=time.perf_counter();return [t1-t0,t2-t1,t2-t0]
  for k in ms:
   for _ in range(2):run(k)
  for rep in range(31):
   keys=list(ms);rng.shuffle(keys)
   for k in keys:ts[k].append(run(k))
  out[n]={'raw':ts,'median':{k:[statistics.median(v[i] for v in vs) for i in range(3)] for k,vs in ts.items()},'units':['prefill_s','decode32_s','paired_total_s']};print(n,out[n]['median'],flush=True)
 return out

if __name__=='__main__':
 checks=check();(P/'checks_v4.json').write_text(json.dumps(checks,indent=2));print('checks pass',flush=True)
 result={'checks':checks,'benchmark':bench(),'runtime':{'torch':torch.__version__,'threads':torch.get_num_threads(),'cuda_available':torch.cuda.is_available(),'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
 (P/'results_v4.json').write_text(json.dumps(result,indent=2))
