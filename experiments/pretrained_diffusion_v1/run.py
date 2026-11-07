"""Frozen small-data protocol. Each process handles one method/seed only."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import json,time,hashlib,math,resource,sys
import numpy as np
import torch
from torch.nn import functional as F
from models import *
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
STEPS=96;BS=2;CONTEXT=32;SEEDS=(1,2,3);PARTICLES=4
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(f,v):(P/f).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def freeze():
    write('protocol.json',dict(experiment='Pretrained Pythia14M to three independent two-layer diffusion blocks: feasibility pilot',seeds=SEEDS,equal_data_steps=STEPS,batch=BS,context=CONTEXT,training_token_exposures_equal_data=STEPS*BS*CONTEXT,methods=['untouched','normal_finetune','pretrained_diffusion'],frozen_shared_parameters='Original input embedding and output head frozen in both fine-tuned methods; all six original layers trainable. Diffusion adds per-block final norms and zero-output time-conditioning MLPs. No parameters shared between trainable blocks.',original_model='EleutherAI/pythia-14m at cf967c0a9a04383db6f7b1108d86b2962634b4ac, Apache-2.0',data='Official WikiText2 raw train/validation/test, each LF-preserved line independently BPE tokenized, no special tokens. Heldout from adaptation only, pretraining contamination unknown.',training='AdamW lr3e-4 betas(.9,.95),weight_decay.01,clip1,warmup12,constant thereafter. Same random train windows per paired seed. Diffusion cycles blocks0,1,2, inverse-CDF truncated-lognormal sigma within equal-probability interval; perexample weightedCE ((sigma²+1/128)/(sigma²/128))/512.',conversion='Fresh start from original pretrained weights for each run. No distillation, no cross-block gradients, clean raw embeddings plus scaled/preconditioned unit-L2 noisy target embeddings; concat clean/noisy streams and strict clean-past-only attention.',inference='Two log-spaced Euler steps in each of3 noise ranges followed by final block at sigma=.002:7 denoiser calls,14 layer applications,2x sequence. Start noise at80. Posterior-mean unit-L2 target embedding from softmax; negative Euler dt.',evaluation='64 disjoint uniformly spaced test windows,32 validation,32 tokens/window,score final16 targets. Converted decoder log score from arithmetic4-particle mixture of final softmax probabilities, fixed target-free particles. This is finite-decoder NLL, NOT exact implicit-diffusion perplexity or ELBO. Normal model exact truncated-context nexttoken NLL.',matching='Paired96 updates equal observed data (6144tokens); additionally save converted first step reaching normal96 cumulative measuredCPU, potentially below96 or after96,cap384. One-step timing overshoot disclosed. Warmup2 forwardbackward passes WITHOUT optimizer update excluded from budget, evaluation excluded.',compute='Scratch CPU one thread; no rentals or production changes. Each method process capped600CPU seconds.',selection='No tuning or selecting using test; all3 seeds and both endpoints reported. Validation diagnostic only. No large-model claims.',files={f:sha(P/f) for f in ['models.py','run.py','checks.py','prepare.py']},model_manifest_sha256=sha(P/'model_manifest.json'),source_manifest_sha256=sha(P/'source_data_manifest.json'),token_manifest_sha256=sha(P/'token_manifest.json')))
def data():
    manifest=json.loads((P/'token_manifest.json').read_text());out={}
    for s in ['train','validation','test']:
        f=P/'data'/f'{s}.npy';assert sha(f)==manifest[s]['sha256'];out[s]=torch.from_numpy(np.load(f))
    return out
def batch(d,g):
    starts=torch.randint(0,len(d)-CONTEXT-1,(BS,),generator=g);z=d[starts[:,None]+torch.arange(CONTEXT+1)[None,:]]
    return z[:,:-1],z[:,1:]
@torch.inference_mode()
def evaluate(m,kind,d,n):
    m.eval();starts=torch.linspace(0,len(d)-CONTEXT-1,n).long();assert torch.diff(starts).min()>CONTEXT+1
    losses=[];g=torch.Generator().manual_seed(449874);t=time.process_time()
    for ss in starts.split(BS):
        z=d[ss[:,None]+torch.arange(CONTEXT+1)[None,:]];x,y=z[:,:-1],z[:,1:]
        if kind!='pretrained_diffusion':
            ll=m(x,use_cache=False).logits.log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)
        else:
            vals=[]
            for particle in range(PARTICLES):
                noise=torch.randn(len(ss),CONTEXT,D,generator=g)
                vals.append(m.predict(x,noise).log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1))
            ll=torch.logsumexp(torch.stack(vals),0)-math.log(PARTICLES)
        losses.extend((-ll[:,-16:]).mean(1).tolist())
    m.train();return dict(nll_nats=float(np.mean(losses)),window_nll_nats=losses,scored_targets=n*16,evaluation_cpu_seconds=time.process_time()-t)
def save(m,kind,seed,endpoint,steps,cpu,wall,last,d,budget):
    (P/'checkpoints').mkdir(exist_ok=True)
    path=P/'checkpoints'/f'{kind}_{seed}_{endpoint}.pt'
    # Save only trainable tensors; no upstream or frozen weights in research artifact.
    tensors={n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
    torch.save(tensors,path)
    r=dict(kind=kind,seed=seed,endpoint=endpoint,steps=steps,token_exposures=steps*BS*CONTEXT,trainable_parameters=trainable_count(m),training_cpu_seconds=cpu,training_wall_seconds=wall,last_training_loss=last,cpu_target=budget,cpu_ratio=None if budget is None else cpu/budget,validation=evaluate(m,kind,d['validation'],32),test=evaluate(m,kind,d['test'],64),checkpoint_sha256=sha(path))
    write(f'{kind}_{seed}_{endpoint}.json',r)
    print(json.dumps({k:v for k,v in r.items() if k not in ['validation','test']}|{'test_nll':r['test']['nll_nats']}),flush=True)
def train(kind,seed):
    p=json.loads((P/'protocol.json').read_text())
    for f,h in p['files'].items():assert sha(P/f)==h,(f,'changed after freeze')
    d=data();torch.manual_seed(seed);original=load_original()
    m=DiffusionBlocks(original) if kind=='pretrained_diffusion' else frozen_maps(original)
    if kind=='untouched':
        r=dict(kind=kind,validation=evaluate(m,kind,d['validation'],32),test=evaluate(m,kind,d['test'],64));write('untouched.json',r);print(r['test']['nll_nats']);return
    m.train();params=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=.0003,betas=(.9,.95),weight_decay=.01)
    g=torch.Generator().manual_seed(seed+100);ng=torch.Generator().manual_seed(seed+200)
    budget=None if kind=='normal_finetune' else json.loads((P/f'normal_finetune_{seed}_equal_data.json').read_text())['training_cpu_seconds']
    # Warm up dispatch and allocator, without modifying any parameter or optimizer.
    wx,wy=batch(d['train'],torch.Generator().manual_seed(980))
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        logits=m.training_logits(wx,wy,torch.zeros(BS,CONTEXT,D),torch.ones(BS,1,1),0) if kind=='pretrained_diffusion' else m(wx,use_cache=False).logits
        F.cross_entropy(logits.flatten(0,1),wy.flatten()).backward()
    del logits;opt.zero_grad(set_to_none=True);cpu=wall=0.;equal_cpu=False
    for step in range(1,385):
        x,y=batch(d['train'],g);b=(step-1)%3
        if kind=='pretrained_diffusion':noise=torch.randn(BS,CONTEXT,D,generator=ng);s=sample_sigma(b,BS,ng)
        t=time.process_time();w=time.perf_counter();opt.zero_grad(set_to_none=True)
        opt.param_groups[0]['lr']=.0003*min(1,step/12)
        if kind=='pretrained_diffusion':
            logits=m.training_logits(x,y,noise,s,b)
            ce=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape(BS,CONTEXT)
            weight=((s.square()+1./D)/(s.square()/D)/512.).reshape(BS,1)
            assert weight.shape==(BS,1) and ce.shape==(BS,CONTEXT)
            loss=(ce*weight).mean()
        else:loss=F.cross_entropy(m(x,use_cache=False).logits.flatten(0,1),y.flatten())
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        cpu+=time.process_time()-t;wall+=time.perf_counter()-w;last=float(loss)
        if kind=='pretrained_diffusion' and not equal_cpu and cpu>=budget:
            save(m,kind,seed,'equal_cpu',step,cpu,wall,last,d,budget);equal_cpu=True
        if step==STEPS:
            save(m,kind,seed,'equal_data',step,cpu,wall,last,d,budget)
            if kind=='normal_finetune':return
        if step>=STEPS and equal_cpu:return
        if step%24==0:print(kind,seed,step,round(cpu,3),last,flush=True)
    if not equal_cpu:save(m,kind,seed,'capped_cpu',step,cpu,wall,last,d,budget)
if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(600,605))
    if '--freeze' in sys.argv:freeze()
    else:train(sys.argv[1],int(sys.argv[2]))
