"""Frozen pilot: paired initialization/data seeds, target-free heldout decoder."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time,hashlib,math,resource,sys
import numpy as np
import torch
from torch.nn import functional as F
from models import *
P=Path(__file__).parent;STEPS=600;BS=16;SEEDS=(1,2,3);KINDS=('bptt','diffusion_weighted','diffusion_ce','transformer')
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(f,v):(P/f).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def source_data():
 manifest=json.loads((P/'source_data_manifest.json').read_text());out={}
 for split in ['train','validation','test']:
  p=P/'data'/f'{split}.txt';assert sha(p)==manifest['splits'][split]['text_sha256'];out[split]=torch.from_numpy(np.frombuffer(p.read_bytes(),dtype=np.uint8).copy()).long()
 return out
def freeze():
 data=source_data();write('protocol.json',dict(experiment='Independent recurrent-depth denoising pilot; not a Huginn/DiffusionBlocks paper replication',seeds=SEEDS,methods=KINDS,steps_equal_data=STEPS,batch=BS,context=64,train_token_exposures_equal_data=STEPS*BS*64,train_source='all WikiText2 train bytes; uniform random windows with replacement',evaluation='256 disjoint evenly spaced windows from official test;128 validation; final32 of64 shifted targets scored',decoder='Arithmetic mean of final softmax over four fixed independent target-free Gaussian particles; next-byte log-score, NOT exact diffusion/greedy sampler likelihood',optimization='AdamW lr.002,betas(.9,.95),weight_decay.01,gradclip1,warmup30,constant thereafter; no hyperparameter selection',noise='log sigma Normal(-1.2,1.2), clamped [.002,80]; codebook learned with unit L2 rows; EDM sigma_data.5',weighted='Primary diffusion: CE weighted (sigma^2+.25)/(sigma*.5)^2, divided by FIXED 500 for numerical loss scale; same relative weights. Secondary unweighted CE predeclared.',inference='4 Karras rho7 levels80 to.002, Euler with negative dt, posterior mean normalized embedding, last logits',baseline='Exact same recurrent architecture/inference graph, full 4-step BPTT from noise; not original Huginn32 with8 truncation',transformer='2-layer causal Transformer same width/context, secondary comparator, fewer noise-conditioning parameters',matching='Same seed/init shared recurrent models; same first600 batches; additionally continue each cheaper method until baseline600 cumulative CPU training time, cap3000. Timing excludes data extraction and validation/test, includes optimizer/loss',compute='Single CPU thread shared scratch; actual process/wall training seconds; no paid resources',selection='No test-based selection; all seeds, methods, equal-data and equal-CPU endpoints reported. Validation diagnostic only.',limits='1200 process CPU seconds entire training process; final artifact contains all checkpoints',files={f:sha(P/f) for f in ['models.py','run.py','checks.py']},data_manifest_sha256=sha(P/'source_data_manifest.json'),torch=torch.__version__))
 print('PROTOCOL FROZEN',flush=True)
def verify():
 p=json.loads((P/'protocol.json').read_text())
 for f,h in p['files'].items():assert sha(P/f)==h,(f,'changed after freeze')
def batch(data,g):
 starts=torch.randint(0,len(data)-65,(BS,),generator=g);z=data[starts[:,None]+torch.arange(65)[None,:]];return z[:,:-1],z[:,1:]
@torch.inference_mode()
def evaluate(m,kind,data,n=256):
 m.eval();starts=torch.linspace(0,len(data)-65,n).long();assert torch.diff(starts).min()>65
 values=[];t0=time.process_time();g=torch.Generator().manual_seed(128847)
 for ss in starts.split(16):
  z=data[ss[:,None]+torch.arange(65)[None,:]];x,y=z[:,:-1],z[:,1:]
  if kind=='transformer':probs=m(x).double().softmax(-1)
  else:
   probs=0
   for particle in range(4):
    noise=torch.randn(len(ss),64,D,generator=g);probs=probs+m.predict(x,noise).double().softmax(-1)/4
  values.extend((-probs[:,-32:].gather(-1,y[:,-32:,None]).squeeze(-1).log()/math.log(2)).mean(1).tolist())
 m.train();return dict(bpb=float(np.mean(values)),window_bpb=values,targets=n*32,eval_cpu=time.process_time()-t0)
def save_eval(m,kind,seed,endpoint,steps,cpu,wall,loss,data):
 torch.save(m.state_dict(),P/'checkpoints'/f'{kind}_{seed}_{endpoint}.pt')
 r=dict(kind=kind,seed=seed,endpoint=endpoint,steps=steps,token_exposures=steps*BS*64,training_cpu_seconds=cpu,training_wall_seconds=wall,parameters=count(m),last_training_objective=loss,validation=evaluate(m,kind,data['validation'],128),test=evaluate(m,kind,data['test']))
 write(f'{kind}_{seed}_{endpoint}.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ['validation','test']}|{'test_bpb':r['test']['bpb']}),flush=True);return r
def train(kind,seed,data,budget=None):
 torch.manual_seed(seed);m=Transformer() if kind=='transformer' else Recurrent();opt=torch.optim.AdamW(m.parameters(),lr=.002,betas=(.9,.95),weight_decay=.01)
 g=torch.Generator().manual_seed(seed+1000);ng=torch.Generator().manual_seed(seed+2000);cpu=wall=0.;results=[]
 for step in range(1,3001):
  x,y=batch(data['train'],g);noise=torch.randn(BS,64,D,generator=ng);s=torch.exp(torch.randn(BS,1,1,generator=ng)*1.2-1.2).clamp(.002,80.)
  t0=time.process_time();w0=time.perf_counter();opt.zero_grad(set_to_none=True);opt.param_groups[0]['lr']=.002*min(1,step/30)
  if kind=='transformer':logits=m(x)
  elif kind=='bptt':logits=m.predict(x,noise)
  else:logits=m.train_denoise(x,y,noise,s)
  losses=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape(BS,64)
  if kind=='diffusion_weighted':losses=losses*((s.square()+.25)/(.5*s).square()).reshape(BS,1)/500.
  loss=losses.mean();assert torch.isfinite(loss),kind;loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();cpu+=time.process_time()-t0;wall+=time.perf_counter()-w0
  if step==STEPS:results.append(save_eval(m,kind,seed,'equal_data',step,cpu,wall,float(loss),data))
  if kind=='bptt' and step==STEPS:return results,cpu
  if step>=STEPS and cpu>=budget:
   results.append(save_eval(m,kind,seed,'equal_cpu',step,cpu,wall,float(loss),data));return results,cpu
  if step%300==0:print(kind,seed,step,'cpu',round(cpu,2),flush=True)
 results.append(save_eval(m,kind,seed,'capped_cpu',step,cpu,wall,float(loss),data));return results,cpu
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(1200,1205));verify();data=source_data();out=[];start=time.process_time()
 for seed in SEEDS:
  rows,budget=train('bptt',seed,data);out.extend(rows)
  for kind in KINDS[1:]:rows,_=train(kind,seed,data,budget);out.extend(rows)
  write('results_partial.json',out)
 write('results.json',dict(complete=True,results=out,total_cpu_seconds=time.process_time()-start))
if __name__=='__main__':freeze() if '--freeze' in sys.argv else main()
