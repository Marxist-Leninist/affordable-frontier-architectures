"""Bounded isolated step accounting, separate from quality experiment."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,json,time,statistics
from pathlib import Path
from torch.nn import functional as F
from models import *
torch.set_num_threads(1);torch.manual_seed(1717);P=Path(__file__).parent
x=torch.randint(256,(16,64));y=torch.randint(256,(16,64));noise=torch.randn(16,64,D);s=torch.ones(16,1,1)*.3
rows=[]
for kind in ['bptt','diffusion_weighted','transformer']:
 m=Transformer() if kind=='transformer' else Recurrent();opt=torch.optim.AdamW(m.parameters(),lr=.002)
 parameters={p.untyped_storage().data_ptr() for p in m.parameters()};saved={}
 def pack(t):
  sto=t.untyped_storage();ptr=sto.data_ptr()
  if ptr not in parameters:saved[ptr]=sto.nbytes()
  return t
 def lossfn():
  if kind=='transformer':z=m(x)
  elif kind=='bptt':z=m.predict(x,noise)
  else:z=m.train_denoise(x,y,noise,s)
  loss=F.cross_entropy(z.flatten(0,1),y.flatten())
  if kind=='diffusion_weighted':loss=loss*(.3**2+.25)/(.5*.3)**2/500.
  return loss
 with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):loss=lossfn()
 loss.backward();opt.step();saved_bytes=sum(saved.values())
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
  opt.zero_grad(set_to_none=True);lossfn().backward();opt.step()
 times=[]
 for i in range(25):
  t=time.process_time();opt.zero_grad(set_to_none=True);lossfn().backward();opt.step();times.append(time.process_time()-t)
 rows.append(dict(kind=kind,parameters=count(m),optimizer_state_bytes=sum(t.numel()*t.element_size() for st in opt.state.values() for t in st.values() if isinstance(t,torch.Tensor)),unique_saved_nonparameter_storage_bytes=saved_bytes,profiler_counted_flops=sum(e.flops for e in prof.key_averages()),median_step_cpu_seconds=statistics.median(times[5:]),forward_transformer_blocks=5 if kind=='bptt' else 2,denoiser_calls=4 if kind=='bptt' else 1 if kind=='diffusion_weighted' else 0))
out=dict(rows=rows,notes=['Profiler FLOPs cover recognized matmuls, exclude fused SDPA and many elementwise operations; not complete FLOPs.','Saved nonparameter storage is graph-retained activation proxy, not measured peak resident memory.','Same-parameter recurrent variants have identical parameter and optimizer memory. Recurrent single-pass training reduces activation/compute, not weights by number of recurrences.','CPU timings from shared machine; no GPU or production extrapolation.'])
(P/'profile.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
