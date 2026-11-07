"""Freeze disjoint evaluation windows; cache public-training-text teacher targets."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import hashlib,json,time,sys
import numpy as np
import torch
from models import *
torch.set_num_threads(1)
STEPS=768;MAX_STEPS=1536;BS=2;T=32;SEEDS=[11,22,33]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(P/n).write_text(json.dumps(d,indent=2)+'\n')
def windows():
    old=P.parent/'pretrained_diffusion_v1'
    result={'description':'New disjoint evaluation windows exclude every v1 scored context. Official split separation retained; pretraining overlap unknown.','context':T,'splits':{}}
    for split,n,prior_n in [('validation',32,32),('test',64,64)]:
        d=np.load(P/'data'/f'{split}.npy');prior=np.linspace(0,len(d)-33,prior_n).astype(np.int64)
        candidates=np.random.default_rng(64913 if split=='validation' else 99027).permutation(np.arange(0,len(d)-33,64));chosen=[]
        for i in candidates:
            if all(abs(int(i)-int(j))>33 for j in prior):chosen.append(int(i))
            if len(chosen)==n:break
        assert len(chosen)==n;assert all(abs(a-b)>33 for i,a in enumerate(chosen) for b in chosen[i+1:])
        result['splits'][split]={'starts':chosen,'excluded_v1_starts':prior.tolist()}
    write('window_manifest.json',result)
def train_windows(seed):
    n=len(np.load(P/'data'/'train.npy',mmap_mode='r'));g=torch.Generator().manual_seed(seed+100)
    return torch.randint(0,n-T-1,(MAX_STEPS,BS),generator=g)
@torch.inference_mode()
def cache(seed):
    d=torch.from_numpy(np.load(P/'data'/'train.npy'));model=load_original();starts=train_windows(seed)
    (P/'teacher_cache').mkdir(exist_ok=True);indices=np.empty((MAX_STEPS,BS,T,32),np.int32);probs=np.empty_like(indices,dtype=np.float32)
    t=time.process_time();w=time.perf_counter()
    for step,ss in enumerate(starts):
        x=d[ss[:,None]+torch.arange(T)[None,:]];q,ix=model(x,use_cache=False).logits.softmax(-1).topk(32,-1)
        indices[step]=ix.numpy();probs[step]=q.numpy()
        if (step+1)%384==0:print('teacher',seed,step+1,flush=True)
    cpu=time.process_time()-t;wall=time.perf_counter()-w
    a=P/'teacher_cache'/f'indices_{seed}.npy';b=P/'teacher_cache'/f'probs_{seed}.npy';np.save(a,indices);np.save(b,probs)
    write(f'teacher_{seed}_manifest.json',dict(seed=seed,steps=MAX_STEPS,top_k=32,cpu_seconds=cpu,wall_seconds=wall,initial_equal_data_fraction=STEPS/MAX_STEPS,files={a.name:sha(a),b.name:sha(b)},train_window_sha256=hashlib.sha256(starts.numpy().tobytes()).hexdigest(),bytes=a.stat().st_size+b.stat().st_size,teacher='Pinned untouched Pythia-14M, offline targets from training split only, no teacher computation required during selected-block updates'))
    print('teacher done',seed,cpu,flush=True)
if __name__=='__main__':windows() if '--windows' in sys.argv else cache(int(sys.argv[1]))
