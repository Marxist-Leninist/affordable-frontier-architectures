import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,time,math
import torch
from torch.nn import functional as F
from models import *
torch.set_num_threads(1);torch.manual_seed(99)
m=Recurrent();x=torch.randint(0,256,(2,64));noise=torch.randn(2,64,D);y=torch.randint(0,256,(2,64))
m.eval();a=m.predict(x,noise);xx=x.clone();xx[:,33:]=(xx[:,33:]+1)%256;b=m.predict(xx,noise)
causal=float((a[:,:33]-b[:,:33]).abs().max());assert causal<1e-6
p=m.predict(x[:1,:33],noise[:1,:33]);prefix=float((a[:1,:33]-p).abs().max());assert prefix<2e-5
z=torch.randn(4,48);target=torch.randn(4,48);s=2.;sn=.1
zn=z+(sn-s)/s*(z-target);err=float((zn-target-(z-target)*sn/s).abs().max());assert err<1e-6
zg=z+(0-s)/s*(z-target);assert torch.allclose(zg,target,atol=1e-6)
out={'causal_max_abs':causal,'prefix_max_abs':prefix,'oracle_euler_max_abs':err,'parameters':{'recurrent':count(m),'transformer':count(Transformer())},'timings':{}}
for kind in ['bptt','diffusion','transformer']:
 model=Transformer() if kind=='transformer' else Recurrent();opt=torch.optim.AdamW(model.parameters(),lr=.003);times=[]
 for i in range(15):
  t=time.perf_counter();opt.zero_grad(set_to_none=True)
  if kind=='transformer':logits=model(x)
  elif kind=='bptt':logits=model.predict(x,noise)
  else:logits=model.train_denoise(x,y,noise,torch.full((2,1,1),.3))
  loss=F.cross_entropy(logits.flatten(0,1),y.flatten());loss.backward();opt.step();times.append(time.perf_counter()-t)
 assert all(torch.isfinite(p).all() for p in model.parameters())
 out['timings'][kind]=sum(times[5:])/len(times[5:])
print(json.dumps(out,indent=2));open(os.path.join(os.path.dirname(__file__),'checks.json'),'w').write(json.dumps(out,indent=2))
