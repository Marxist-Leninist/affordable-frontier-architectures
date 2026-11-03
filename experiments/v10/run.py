import os,json,time,hashlib,random
os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import numpy as np,torch
from models import *
from data import *
from pathlib import Path
P=Path(__file__).parent; torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
def savej(name,obj): (P/name).write_text(json.dumps(obj,indent=2))
def statehash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()): h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()
def checkpoint(model,name,config,probe):
 path=P/'checkpoints'/f'{name}.pt'; path.parent.mkdir(exist_ok=True)
 state=model.state_dict(); sh=statehash(state); torch.save({'state_dict':state,'config':config,'ordered_state_sha256':sh},path)
 loaded=torch.load(path,weights_only=True); other=make(config['architecture']);other.load_state_dict(loaded['state_dict']);other.eval();model.eval()
 with torch.no_grad(): diff=(model(probe)-other(probe)).abs().max().item()
 assert diff==0 and statehash(loaded['state_dict'])==sh
 return {'path':str(path),'state_sha256':sh,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'reload_max_logit_difference':diff}
def evaluate(model,lengths,n,base):
 model.eval();out={};preds={}
 with torch.no_grad():
  for l in lengths:
   for k in [4,8,16]:
    x,y,_=generate(base+l*19+k,n,l,k,True); correct=[]; losses=[]; pp=[]
    for i in range(0,n,8):
     z=model(torch.from_numpy(x[i:i+8])); pp.extend(z.argmax(-1).tolist());losses.extend(nn.functional.cross_entropy(z,torch.from_numpy(y[i:i+8]),reduction='none').tolist())
    out[f'L{l}_K{k}']={'accuracy':float(np.mean(np.array(pp)==y)),'ce':float(np.mean(losses)),'n':n,'dataset_sha256':digest(x,y)};preds[f'L{l}_K{k}']=pp
 return out,preds

def tests():
 torch.manual_seed(8); m=Transformer().double(); x=torch.randint(0,41,(2,13)); a=m(x); b=m(x,reference=True); delta=(a-b).abs().max().item(); assert delta<1e-12
 m.zero_grad(); a.square().sum().backward(); ga=[p.grad.clone() for p in m.parameters()];m.zero_grad();b.square().sum().backward();gd=max((p.grad-g).abs().max().item() for p,g in zip(m.parameters(),ga));assert gd<1e-10
 z=torch.randn(2,13,32,dtype=torch.double); z2=z.clone();z2[:,7:]+=10
 for local in [True,False]:
  layer=m.local if local else m.global_; ya=layer(z,local=local,reference=True);yb=layer(z2,local=local,reference=True);assert torch.equal(ya[:,:7],yb[:,:7])
 # Independent directional finite difference gradient of exact efficient computation.
 m.zero_grad(); loss=m(x).square().sum();loss.backward(); dirs=[torch.randn_like(p) for p in m.parameters()];analytic=sum((p.grad*d).sum().item() for p,d in zip(m.parameters(),dirs)); eps=1e-6
 with torch.no_grad():
  for p,d in zip(m.parameters(),dirs):p.add_(d,alpha=eps)
  plus=m(x).square().sum().item()
  for p,d in zip(m.parameters(),dirs):p.add_(d,alpha=-2*eps)
  minus=m(x).square().sum().item()
  for p,d in zip(m.parameters(),dirs):p.add_(d,alpha=eps)
 numeric=(plus-minus)/(2*eps);err=abs(numeric-analytic)/max(1,abs(analytic));assert err<1e-6
 controls={}
 for l in [128,256,512,2048,4096]:
  for k in [4,8,16]:
   xx,yy,pos=generate(17001+l+k,256,l,k,True);assert all(oracle(a)==b for a,b in zip(xx,yy));assert np.all(np.bincount(yy,minlength=16)==16);assert np.all(np.bincount(xx[:,-1],minlength=16)==16)
   assert all(sum((a[:-2]<16))==k for a in xx)
   controls[f'L{l}_K{k}']={'oracle_accuracy':1.,'constant_guess_accuracy':float(np.mean(yy==0)),'query_fact_position_min':min(pos),'query_fact_position_max':max(pos),'sha256':digest(xx,yy)}
 result={'reference_logit_max_difference':delta,'reference_parameter_gradient_max_difference':gd,'directional_gradient_relative_error':err,'causality_pass':True,'controls':controls};savej('tests.json',result);print('TESTS',json.dumps(result),flush=True)

def main():
 tests(); rng=np.random.default_rng(444);schedule=[(int(rng.choice([128,256,512])),int(rng.choice([4,8,16]))) for _ in range(600)]
 protocol={'steps':600,'batch_size':8,'schedule':schedule,'total_training_tokens':sum(l*8 for l,k in schedule),'lengths_train':[128,256,512],'facts_train':[4,8,16],'tuning_seed':101,'learning_rates':[.001,.003,.01],'selection':'lowest pooled validation cross entropy; then lower learning rate','validation_cases_per_cell':128,'test_cases_per_cell':256,'test_lengths':[512,2048,4096],'final_seeds':[1,2,3],'optimizer':'Adam default beta/eps, no weight decay','gradient_clip_norm':1.,'threads':1,'dtype':'float32','deterministic':True,'parameters':{a:count(make(a)) for a in ['transformer','rnn']},'rnn_hidden':hidden(),'memory_bits':[16,32,64],'architecture':'pre-LN local causal width2 block, global causal block with exact final-query-only computation; no positions; width32 heads4 FF64','no_test_tuning':True,'data_stream':'same fixed600 minibatches for every model, LR, and seed; seed900000+step; independent structural and value RNG streams','validation_seed_base':310000,'test_seed_base':720000,'source_sha256':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['models.py','data.py','run.py']}}
 savej('protocol.json',protocol); results={'protocol':protocol,'tuning':{},'final':{},'selection':{}}; hashlist=[]
 for i,(l,k) in enumerate(schedule):
  x,y,_=generate(900000+i,8,l,k);hashlist.append(digest(x,y))
 savej('paired_training_batches.json',{'sha256':hashlist,'schedule':schedule})
 def train(arch,lr,seed,stage):
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);m=make(arch);opt=torch.optim.Adam(m.parameters(),lr=lr);start=time.monotonic();losses=[];m.train()
  for i,(l,k) in enumerate(schedule):
   x,y,_=generate(900000+i,8,l,k);assert digest(x,y)==hashlist[i];opt.zero_grad();loss=nn.functional.cross_entropy(m(torch.from_numpy(x)),torch.from_numpy(y));loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(loss.item())
   if (i+1)%100==0: print(stage,arch,lr,seed,i+1,'loss',np.mean(losses[-100:]),'seconds',time.monotonic()-start,flush=True)
  config={'architecture':arch,'learning_rate':lr,'seed':seed,'stage':stage,'steps':600,'rnn_hidden':hidden() if arch=='rnn' else 0};ck=checkpoint(m,f'{stage}_{arch}_lr{lr}_seed{seed}',config,torch.from_numpy(x));return m,{'checkpoint':ck,'training_seconds':time.monotonic()-start,'last100_training_ce':float(np.mean(losses[-100:])),'batch_hash_sha256':hashlib.sha256(''.join(hashlist).encode()).hexdigest()}
 for arch in ['transformer','rnn']:
  results['tuning'][arch]=[]
  for lr in [.001,.003,.01]:
   m,info=train(arch,lr,101,'tune');val,_=evaluate(m,[128,256,512],128,310000);info.update(lr=lr,validation=val,mean_validation_ce=float(np.mean([v['ce'] for v in val.values()])));results['tuning'][arch].append(info);savej('results_partial.json',results)
  selected=min(results['tuning'][arch],key=lambda d:(d['mean_validation_ce'],d['lr']))['lr'];results['selection'][arch]=selected;savej('selection.json',results['selection'])
 # Selection for both frozen before any test evaluation.
 for arch in ['transformer','rnn']:
  results['final'][arch]=[]
  for seed in [1,2,3]:
   m,info=train(arch,results['selection'][arch],seed,'final');metrics,preds=evaluate(m,[512,2048,4096],256,720000);info.update(seed=seed,test=metrics,predictions=preds);results['final'][arch].append(info);savej('results_partial.json',results)
 savej('results.json',results);print('DONE',flush=True)
if __name__=='__main__':main()
