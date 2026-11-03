"""Separate fixed-batch optimization sanity check; never used to select v10 results."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import json,torch
from pathlib import Path
from models import make
from data import generate,digest

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
x,y,_=generate(202,8,64,4);x=torch.from_numpy(x);y=torch.from_numpy(y)
report={'purpose':'Separate fixed-batch overfit diagnostic, not held-out evaluation or retuning','seed':202,'data_seed':202,'length':64,'facts':4,'batch_size':8,'steps':100,'learning_rate':0.003,'batch_sha256':digest(x.numpy(),y.numpy()),'models':{}}
for arch in ('transformer','rnn'):
 torch.manual_seed(202);m=make(arch);opt=torch.optim.Adam(m.parameters(),lr=.003);trace=[]
 with torch.no_grad():initial=float(torch.nn.functional.cross_entropy(m(x),y))
 for step in range(1,101):
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if step%20==0:trace.append({'step':step,'pre_update_ce':float(loss.detach())})
 with torch.no_grad():z=m(x);final=float(torch.nn.functional.cross_entropy(z,y));accuracy=float((z.argmax(-1)==y).float().mean())
 report['models'][arch]={'initial_ce':initial,'final_ce':final,'final_training_accuracy':accuracy,'relative_ce_reduction':1-final/initial,'trace':trace}
print(json.dumps(report,indent=2));Path(__file__).with_name('independent_training_sanity.json').write_text(json.dumps(report,indent=2)+'\n')
