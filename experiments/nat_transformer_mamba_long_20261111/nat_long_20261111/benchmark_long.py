"""Standalone paired inference/training throughput after all training finishes."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import copy,json,time,resource,fcntl
import numpy as np
import torch
from torch.nn import functional as F
from models_long import NAT
from data_long import ROOT,TrainingCorpus
from train_long import optimizer,atomic_json

def main():
    phase_lock=(ROOT/'.evaluation_timing.lock').open('a')
    fcntl.flock(phase_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'standalone_timings.json').exists():raise RuntimeError('Refusing duplicate standalone timing')
    assert json.loads((ROOT/'pipeline_status.json').read_text())['status']=='training_complete'
    assert (ROOT/'evaluation_results.json').exists(), 'Complete held-out evaluation before standalone timing.'
    os.sched_setaffinity(0,{1});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    selected=json.loads((ROOT/'selection_long.json').read_text())['selected_lrs'];models={};opts={}
    for kind in ['transformer','mamba']:
        model=NAT(kind);obj=torch.load(ROOT/'runs'/f'final_{kind}_s1_lr{selected[kind]}'/'final.pt',weights_only=True);model.load_state_dict(obj['model']);models[kind]=model;opts[kind]=optimizer(model,selected[kind])
    rng=np.random.default_rng(7731);corpus=TrainingCorpus();p,y=corpus.batch(rng,8);inference=[];training=[]
    # Both training paths use their own bounded throwaway copy; saved models untouched.
    for kind,model in models.items():
        for _ in range(5):
            opt=opts[kind];opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(p).flatten(0,1),y.flatten());loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    for trial in range(40):
        for kind in rng.permutation(list(models)):
            model,opt=models[kind],opts[kind];st=time.perf_counter();ct=time.process_time();opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(p).flatten(0,1),y.flatten());loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            training.append({'kind':kind,'trial':trial,'wall_ms':1000*(time.perf_counter()-st),'cpu_ms':1000*(time.process_time()-ct)})
    for batch in [1,8]:
        pp=p[:batch]
        with torch.inference_mode():
            for model in models.values():
                model.eval()
                for _ in range(5):model(pp).argmax(-1)
            for trial in range(60):
                for kind in rng.permutation(list(models)):
                    st=time.perf_counter();models[kind](pp).argmax(-1);elapsed=time.perf_counter()-st
                    inference.append({'kind':kind,'batch':batch,'trial':trial,'ms':elapsed*1000})
    out={'host':'AMD EPYC9V74 shared cloud CPU, 1 pinned thread','scope':'Standalone serial paired order after training. Training uses throwaway model copies initialized from seed1 final checkpoint, five optimizer warmup steps, no saved checkpoint altered. Inference follows those bounded throwaway steps, so it is a shape/runtime benchmark rather than quality evaluation.','training':training,'inference':inference,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    for kind in models:
        ts=[x['wall_ms'] for x in training if x['kind']==kind]
        out[kind]={'train_step_p50_ms':float(np.median(ts)),'train_step_p95_ms':float(np.percentile(ts,95)),'supervised_byte_targets_per_second':256/(np.median(ts)/1000)}
        for batch in [1,8]:
            ts=[x['ms'] for x in inference if x['kind']==kind and x['batch']==batch]
            out[kind][f'infer_batch{batch}']={'p50_ms':float(np.median(ts)),'p95_ms':float(np.percentile(ts,95))}
    atomic_json(ROOT/'standalone_timings.json',out);print(json.dumps({k:out[k] for k in models}),flush=True)

if __name__=='__main__':main()
