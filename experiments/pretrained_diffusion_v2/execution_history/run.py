"""Paired direct-input conversion study. Frozen before any v2 test outcome."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import json,time,hashlib,math,resource,sys
import numpy as np
import torch
from torch.nn import functional as F
from models import *
from setup import STEPS,MAX_STEPS,BS,T,SEEDS,train_windows
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(f,d):(P/f).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def freeze():
    write('protocol.json',dict(version=2,purpose='Controlled causal direct-input retry, not a 512B trial or replication of unpublished paper LM code',model='Pinned pretrained Pythia-14M, same Apache-2.0 safe weights as v1',seeds=SEEDS,methods=['untouched','normal_ce','normal_kd','direct_ce','direct_kd'],steps=STEPS,maximum_steps=MAX_STEPS,batch=BS,context=T,token_exposures=STEPS*BS*T,
    changes='Clean last-past-token query anchor, zero-initialized context/noise/time adapters, clean-only K/V two-stream attention, unweighted denoising CE, offline coarse teacher KL ablation, eightfold longer equal-data training, normalized transport and one Euler step per block.',
    structure='Independent three groups of two original layers. Per-block context/noise/time adapters and norm; frozen shared original input/output vocabulary maps. Block inputs constructed directly from clean past tokens and noisy target codes. No earlier block forward or online teacher during selected-block updates.',
    preservation='A separately checked full-depth neutral adapter model must preserve original logits. Splitting it into independent two-layer denoisers does not preserve that original function. No claim that the block partition is function-preserving.',
    attention='Clean and noisy query streams, both indexed at the last past-token position; each query reads only clean keys at positions<=its index, no noisy keys. Noisy target embedding remains in query residual/MLP. Original clean streams receive no noisy-target information. This is a causal adaptation, not claimed identical to unpublished paper LM code.',
    optimization='AdamW lr1e-4 betas(.9,.95),weight_decay.01,clip1,warmup48. All compared models start independently from original weights. Same train windows/noise schedule per seed; direct groups cycle0,1,2.',
    kd='CE + lambda*KL, lambda=sigma²/(sigma²+1/128), same per-example schedule in normal_kd and direct_kd. KL aggregates teacher top32 token categories plus remaining mass. Targets cached offline from untouched teacher on training data only; full preprocessing CPU/bytes reported and charged to each KD recipe, but physically generated once per seed.',
    noise='Equal-probability truncated lognormal bands, log sigma Normal(-1.2,1.2),range[.002,80],unit-L2 frozen target codes. No EDM CE weighting in v2.',
    inference='Primary:4 fixed independent target-free noise particles; normalized posterior-mean code transport,1 Euler step in each of3 bands then final low-noise group2 readout. Fixed-prefix clean KV computed once per distinct block, reused across particles/noise levels:6clean layer passes plus8query layer passes per particle. Negative Euler delta. High-noise1pass and unnormalized-rollout diagnostics use validation only. No claim of autoregressive cache continuation support.',
    evaluation='New disjoint32validation/64test windows of33tokens excluding all v1 scored contexts. Final16targets scored,512validation/1024test. Converted score is explicit finite4particle decoder NLL, NOT exact implicit diffusion likelihood/perplexity. Ordinary nexttoken NLL exact at context<=32.',
    matching='Equal768updates/data and additional direct endpoint at first update reaching paired ordinary training processCPU, cap1536. CPU budgets include training forward/backward/optimizer, exclude warmup/data construction/model load/eval. KD cache preprocessing is identical shared fixed extra cost for both paired KD methods. Wall time, core matrixFLOPs and file I/O separately measured, not claimed simultaneously matched.',
    limits='Scratch only, one CPU thread, per-run600CPU sec, no production changes or paid resources. Original pretraining data overlap unknown. New evaluation windows exclude previous scored contexts but come from same familiar benchmark.',
    selection='No test-based tuning or checkpoint selection. Fixed hyperparameters,all seeds/endpoints reported. Supplemental decoder diagnostics are validation-only and cannot replace primary test result.',
    source_sha256={f:sha(P/f) for f in ['models.py','run.py','setup.py','checks.py','prepare.py']},model_manifest_sha256=sha(P/'model_manifest.json'),source_data_manifest_sha256=sha(P/'source_data_manifest.json'),token_manifest_sha256=sha(P/'token_manifest.json'),window_manifest_sha256=sha(P/'window_manifest.json')))
def data():
    manifest=json.loads((P/'token_manifest.json').read_text());out={}
    for split in ['train','validation','test']:
        f=P/'data'/f'{split}.npy';assert sha(f)==manifest[split]['sha256'];out[split]=torch.from_numpy(np.load(f))
    return out
def getbatch(d,starts):
    z=d[starts[:,None]+torch.arange(T+1)[None,:]];return z[:,:-1],z[:,1:]
@torch.inference_mode()
def evaluate(m,kind,d,split,decoder='normalized_rollout',particles=4):
    m.eval();starts=torch.tensor(json.loads((P/'window_manifest.json').read_text())['splits'][split]['starts']);g=torch.Generator().manual_seed(44801)
    losses=[];t=time.process_time();w=time.perf_counter()
    for ss in starts.split(BS):
        x,y=getbatch(d,ss)
        if not kind.startswith('direct'):
            ll=m(x,use_cache=False).logits.log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)
        else:
            ll=[];context=m.prepare_context(x,0 if decoder=='high_noise' else None)
            for _ in range(particles):
                z=torch.randn(len(x),T,D,generator=g);lp=m.predict(x,z,decoder,context).log_softmax(-1)
                ll.append(lp.gather(-1,y[:,:,None]).squeeze(-1))
            ll=torch.logsumexp(torch.stack(ll),0)-math.log(particles)
        losses.extend((-ll[:,-16:]).mean(1).tolist())
    m.train();return dict(nll_nats=float(np.mean(losses)),window_nll_nats=losses,scored_targets=len(starts)*16,cpu_seconds=time.process_time()-t,wall_seconds=time.perf_counter()-w,decoder=decoder if kind.startswith('direct') else 'ordinary_next_token',particles=particles if kind.startswith('direct') else None)
def save(m,kind,seed,endpoint,step,cpu,wall,last,d,budget,cache):
    (P/'checkpoints').mkdir(exist_ok=True);f=P/'checkpoints'/f'{kind}_{seed}_{endpoint}.pt'
    torch.save({n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad},f)
    overhead=cache['cpu_seconds'] if cache else 0.
    r=dict(kind=kind,seed=seed,endpoint=endpoint,steps=step,token_exposures=step*BS*T,trainable_parameters=count(m),training_cpu_seconds=cpu,training_wall_seconds=wall,teacher_preprocessing_cpu_seconds=overhead,teacher_preprocessing_wall_seconds=cache['wall_seconds'] if cache else 0.,recipe_cpu_including_full_teacher_cache=cpu+overhead,teacher_cache_bytes=cache['bytes'] if cache else 0,teacher_target_logical_bytes_consumed=step*BS*T*32*8 if cache else 0,training_cpu_target=budget,cpu_ratio=cpu/budget if budget else None,last_training_loss=last,validation=evaluate(m,kind,d['validation'],'validation'),test=evaluate(m,kind,d['test'],'test'),checkpoint_sha256=sha(f))
    write(f'{kind}_{seed}_{endpoint}.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ['validation','test']}|{'test_nll':r['test']['nll_nats']}),flush=True)
def train(kind,seed):
    protocol=json.loads((P/'protocol.json').read_text())
    for f,h in protocol['source_sha256'].items():assert sha(P/f)==h,(f,'changed after freeze')
    d=data();torch.manual_seed(seed);original=load_original();direct=kind.startswith('direct');kd=kind.endswith('kd')
    m=DirectBlocks(original) if direct else freeze_maps(original)
    if kind=='untouched':
        write('untouched.json',dict(kind=kind,validation=evaluate(m,kind,d['validation'],'validation'),test=evaluate(m,kind,d['test'],'test')));return
    params=[p for p in m.parameters() if p.requires_grad];m.train()
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),weight_decay=.01);starts=train_windows(seed);ng=torch.Generator().manual_seed(seed+200)
    cache=None
    if kd:
        cache=json.loads((P/f'teacher_{seed}_manifest.json').read_text())
        for fn,h in cache['files'].items():assert sha(P/'teacher_cache'/fn)==h
        assert hashlib.sha256(starts.numpy().tobytes()).hexdigest()==cache['train_window_sha256']
        indices=np.load(P/'teacher_cache'/f'indices_{seed}.npy',mmap_mode='r');probs=np.load(P/'teacher_cache'/f'probs_{seed}.npy',mmap_mode='r')
    control='normal_kd' if kd else 'normal_ce'
    budget=json.loads((P/f'{control}_{seed}_equal_data.json').read_text())['training_cpu_seconds'] if direct else None
    wx,wy=getbatch(d['train'],starts[0])
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        logits=m.training_logits(wx,wy,torch.zeros(BS,T,D),torch.ones(BS,1,1),0) if direct else m(wx,use_cache=False).logits
        F.cross_entropy(logits.flatten(0,1),wy.flatten()).backward()
    del logits;opt.zero_grad(set_to_none=True);cpu=wall=0.;matched=False
    for step in range(1,MAX_STEPS+1):
        x,y=getbatch(d['train'],starts[step-1]);b=(step-1)%3;noise=torch.randn(BS,T,D,generator=ng);s=sample_sigma(b,BS,ng)
        if kd:ix=torch.from_numpy(indices[step-1].copy());q=torch.from_numpy(probs[step-1].copy())
        t=time.process_time();w=time.perf_counter();opt.zero_grad(set_to_none=True);opt.param_groups[0]['lr']=1e-4*min(1,step/48)
        logits=m.training_logits(x,y,noise,s,b) if direct else m(x,use_cache=False).logits
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        if kd:loss=loss+(coarse_kl(logits,ix,q)*(s.square()/(s.square()+1/D)).reshape(BS,1)).mean()
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();cpu+=time.process_time()-t;wall+=time.perf_counter()-w;last=float(loss)
        if direct and not matched and cpu>=budget:
            save(m,kind,seed,'equal_cpu',step,cpu,wall,last,d,budget,cache);matched=True
        if step==STEPS:
            save(m,kind,seed,'equal_data',step,cpu,wall,last,d,budget,cache)
            if not direct:return
        if step>=STEPS and matched:return
        if step%192==0:print(kind,seed,step,'cpu',round(cpu,3),'loss',last,flush=True)
    if not matched:save(m,kind,seed,'capped_cpu',step,cpu,wall,last,d,budget,cache)
if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(600,605))
    if '--freeze' in sys.argv:freeze()
    else:train(sys.argv[1],int(sys.argv[2]))
