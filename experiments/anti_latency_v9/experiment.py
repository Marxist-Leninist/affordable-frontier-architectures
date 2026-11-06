import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import sys,copy,time,json,random,statistics,math,resource
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
P=Path(__file__).resolve().parent;SRC=P/'checkpoints'
import baseline as previous
from base_models import Attention,LM

torch.set_num_threads(2)
class Fast(Attention):
 def __init__(self,kind='attention',fused=False):
  super().__init__();self.kind=kind;self.fused=fused
 def rope(self,x,offset):
  key=(offset,x.shape[-2],x.device,x.dtype)
  if getattr(self,'_key',None)!=key:
   t,k=x.shape[-2:];freqs=10000.**(-torch.arange(0,k,2,device=x.device,dtype=x.dtype)/k)
   ang=torch.arange(offset,offset+t,device=x.device,dtype=x.dtype)[:,None]*freqs
   self._cos,self._sin=ang.cos(),ang.sin();self._key=key
  a,b=x[...,::2],x[...,1::2]
  return torch.stack((a*self._cos-b*self._sin,a*self._sin+b*self._cos),-1).flatten(-2)
 def forward(self,x,state=None,cache=False):
  b,t,d=x.shape;rawq,rawk,rawv=self.qkv(x).chunk(3,-1)
  q,k,v=[a.reshape(b,t,self.h,d//self.h).transpose(1,2) for a in (rawq,rawk,rawv)]
  offset=0 if state is None else state[0].shape[-2]
  q,k=self.rope(q,offset),self.rope(k,offset)
  newv=v
  if state is not None:k=torch.cat((state[0],k),-2);v=torch.cat((state[1],v),-2)
  if offset==0:a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
  elif t==1:a=F.scaled_dot_product_attention(q,k,v,is_causal=False)
  else:
   mask=torch.arange(k.shape[-2],device=x.device)[None,:]<=torch.arange(offset,offset+t,device=x.device)[:,None]
   a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
  total=None
  if self.kind=='attention':z=a
  else:
   total=newv if t==1 else newv.cumsum(-2)
   if state is not None:total=total+state[2]
   if t==1:
    z=a-total/(offset+1) if self.fused else total/(offset+1)-a
   else:
    key=(offset,t,x.device,x.dtype)
    if getattr(self,'_nkey',None)!=key:
     self._invn=1/torch.arange(offset+1,offset+t+1,device=x.device,dtype=x.dtype)[None,None,:,None];self._nkey=key
    if self.fused:z=torch.addcmul(a,total,self._invn,value=-1) # A-U, negated output projection
    else:z=total*self._invn-a
  ns=None
  if cache:ns=(k,v) if self.kind=='attention' else (k,v,total if (t==1 and offset>0) else total[:,:,-1:,:].clone())
  return self.out(z.transpose(1,2).reshape(b,t,d)),ns if cache else None

def make(kind,impl='fast',seed=1):
 m=previous.model(kind);m.load_state_dict(torch.load(SRC/f'{kind}_{seed}_0.006.pt',weights_only=True));m.eval()
 if impl=='original':return m
 for block in m.blocks:
  old=block.mix;new=Fast(kind,impl=='fused');new.load_state_dict(old.state_dict())
  if impl=='fused' and kind=='signed':
   with torch.no_grad():new.out.weight.neg_()
  block.mix=new
 return m

def parity():
 tests=[]
 for kind in ['attention','signed']:
  for impl in ['fast','fused']:
   torch.manual_seed(47);a=make(kind,'original').double();b=make(kind,impl).double();x=torch.randint(256,(2,19));y=a(x);z=b(x);error=float((y-z).abs().max().detach());assert error<1e-10
   tgt=torch.randn_like(y);(y*tgt).sum().backward();(z*tgt).sum().backward();ge=0
   for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):
    expected=-q.grad if (kind=='signed' and impl=='fused' and name.endswith('mix.out.weight')) else q.grad
    ge=max(ge,float((p.grad-expected).abs().max()))
   assert ge<1e-8
   with torch.no_grad():
    pref,state=b(x[:,:7],cache=True);outs=[pref]
    for i in range(7,19):
     zz,state=b(x[:,i:i+1],state,cache=True);outs.append(zz)
    ce=float((torch.cat(outs,1)-z).abs().max());assert ce<1e-9
    z1,state=b(x[:,:7],cache=True);z2,state=b(x[:,7:],state,cache=True);chunk=float((torch.cat([z1,z2],1)-z).abs().max());assert chunk<1e-9
   tests.append(dict(kind=kind,implementation=impl,fp64_max_error=error,parameter_gradient_max_error=ge,cached_decode_max_error=ce,chunked_cache_error=chunk))
 return tests

@torch.inference_mode()
def quality():
 # Same pinned fresh heldout used in tuning; no selection based on this test.
 _,held=previous.prepare();out=[]
 for seed in [1,2,3]:
  for kind in ['attention','signed']:
   a=make(kind,'original',seed);b=make(kind,'fused',seed);err=0
   for data in held['test'].split(8):err=max(err,float((a(data[:,:-1])-b(data[:,:-1])).abs().max()))
   qa=previous.evaluate(a,held['test']);qb=previous.evaluate(b,held['test']);assert abs(qa-qb)<1e-5;assert err<1e-4
   out.append(dict(seed=seed,kind=kind,original_bpb=qa,optimized_bpb=qb,logit_max_error=err))
 return out

@torch.inference_mode()
def timing():
 models={k:make(kind,impl) for k,kind,impl in [('attention_original','attention','original'),('attention_fast','attention','fast'),('signed_original','signed','original'),('signed_fast','signed','fast'),('signed_fused','signed','fused')]};rng=random.Random(88);result={}
 for n in [128,512]:
  x=torch.randint(256,(8,n));ts={k:[] for k in models}
  for m in models.values():
   for _ in range(8):m(x)
  for rep in range(100):
   keys=list(models);rng.shuffle(keys)
   for k in keys:
    t=time.perf_counter();models[k](x);ts[k].append(time.perf_counter()-t)
  result[f'prefill_{n}']={'raw':ts,'median':{k:statistics.median(v) for k,v in ts.items()}}
 # One generated-token call after already cached prefix; state allocation included, prefill excluded.
 for n in [128,512]:
  models2={k:models[k] for k in ['attention_fast','signed_fast','signed_fused']};x=torch.randint(256,(1,n));token=torch.randint(256,(1,1));states={k:m(x,cache=True)[1] for k,m in models2.items()};ts={k:[] for k in models2}
  for k,m in models2.items():
   for _ in range(10):m(token,states[k],cache=True)
  for rep in range(200):
   keys=list(models2);rng.shuffle(keys)
   for k in keys:
    t=time.perf_counter();models2[k](token,states[k],cache=True);ts[k].append(time.perf_counter()-t)
  result[f'decode_{n}']={'raw':ts,'median':{k:statistics.median(v) for k,v in ts.items()},'state_bytes':{k:sum(v.numel()*v.element_size() for state in ss for v in state) for k,ss in states.items()}}
 return result

def main():
 t=time.perf_counter();out={'parity':parity(),'quality':quality()};(P/'validation.json').write_text(json.dumps(out,indent=2));print('VALIDATION_DONE',flush=True)
 out['timing']=timing();out['runtime']={'seconds':time.perf_counter()-t,'torch':torch.__version__,'threads':2,'max_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 (P/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v['median'] for k,v in out['timing'].items()}),flush=True)
if __name__=='__main__':main()
