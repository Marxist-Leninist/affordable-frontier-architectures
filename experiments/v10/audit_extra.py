import torch,json,numpy as np
from models import *
from data import *
from pathlib import Path
torch.set_num_threads(1);torch.manual_seed(55)
m=RNN(hidden()).double();x=torch.randint(0,41,(2,17));m(x).square().sum().backward();dirs=[torch.randn_like(p) for p in m.parameters()];a=sum((p.grad*d).sum().item() for p,d in zip(m.parameters(),dirs));e=1e-6
with torch.no_grad():
 for p,d in zip(m.parameters(),dirs):p.add_(d,alpha=e)
 plus=m(x).square().sum().item()
 for p,d in zip(m.parameters(),dirs):p.add_(d,alpha=-2*e)
 minus=m(x).square().sum().item()
n=(plus-minus)/(2*e);err=abs(a-n)/max(1,abs(a));assert err<1e-6
p=Path(__file__).parent;h=json.loads((p/'paired_training_batches.json').read_text())['sha256'];checks={}
for l in [512,2048,4096]:
 for k in [4,8,16]:
  xx,yy,pos=generate(720000+l*19+k,256,l,k,True)
  # Query rank among chronologically presented keys must vary, not always a special slot.
  ranks=[list(a[:-2][a[:-2]<16]).index(a[-1]) for a in xx]
  assert len(set(ranks))==k
  checks[f'L{l}_K{k}']={'query_rank_counts':np.bincount(ranks,minlength=k).tolist(),'query_target_joint_counts':np.bincount(xx[:,-1]*16+yy,minlength=256).reshape(16,16).tolist(),'full_dataset_not_train_batch_hash':digest(xx,yy) not in h}
r={'rnn_directional_gradient_relative_error':err,'tests':checks};(p/'audit_extra.json').write_text(json.dumps(r,indent=2));print('rnn gradient',err)
