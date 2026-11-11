"""Post-training whole-model scan speed comparison, no saved weights modified."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,fcntl,hashlib,json,math,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ACTIVE_TARGET=None
def atomic_result(path,obj):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def main():
    global ACTIVE_TARGET
    ap=argparse.ArgumentParser();ap.add_argument('--go-after-postprocess',action='store_true');args=ap.parse_args()
    assert args.go_after_postprocess, 'Explicit launch after main postprocessing is required'
    assert json.loads((ROOT/'postprocess_status.json').read_text())['status']=='completed'
    lock=(ROOT/'.evaluation_timing.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    dory_lock=(ROOT.parent/'dory_loop_compare_20261111'/'.execution.lock').open('a')
    fcntl.flock(dory_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    target=ROOT/'scan_upgrade_whole_model.json';assert not target.exists(), 'Refuse duplicate timing'
    ACTIVE_TARGET=target
    for name,digest in json.loads((ROOT/'frozen_sources.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    def timed_out(*unused):raise TimeoutError('Whole-model upgrade benchmark exceeded 180-second cap')
    def cancelled(*unused):raise InterruptedError('Whole-model upgrade benchmark cancelled')
    signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled)
    signal.signal(signal.SIGALRM,timed_out);signal.alarm(180)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from models_long import NAT,MambaDirection,scan_parallel
    from train_long import optimizer
    from data_long import TrainingCorpus
    os.sched_setaffinity(0,{1});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    selection=json.loads((ROOT/'selection_long.json').read_text())['selected_lrs']['mamba']
    path=ROOT/'runs'/f'final_mamba_s1_lr{selection}'/'final.pt'
    receipt=json.loads(path.with_name('result.json').read_text())
    checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt['status']=='completed' and checkpoint_hash==receipt['final_checkpoint_sha256'], 'Checkpoint receipt mismatch'
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    assert checkpoint['step']==receipt['steps']
    assert all(v.dtype==torch.float32 for v in checkpoint['model'].values()), 'Expected FP32 checkpoint'
    models={};opts={}
    for variant in ('native_linear','torch_parallel_reference'):
        model=NAT('mamba');model.load_state_dict(checkpoint['model'])
        if variant=='torch_parallel_reference':
            for module in model.modules():
                if isinstance(module,MambaDirection):module.scan=scan_parallel
        models[variant]=model;opts[variant]=optimizer(model,selection)
    rng=np.random.default_rng(773188);p,y=TrainingCorpus().batch(rng,8)
    def step(variant,inspect=False):
        model,opt=models[variant],opts[variant];ct=time.process_time();wt=time.perf_counter()
        opt.zero_grad(set_to_none=True);z=model(p);loss=F.cross_entropy(z.flatten(0,1),y.flatten())
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss: '+variant)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        gradients={n:q.grad.detach().clone() for n,q in model.named_parameters()} if inspect else None
        opt.step();wall=time.perf_counter()-wt;cpu=time.process_time()-ct
        if inspect:return z.detach().clone(),gradients,{n:q.detach().clone() for n,q in model.named_parameters()}
        return {'variant':variant,'wall_seconds':wall,'cpu_seconds':cpu}
    assert all(torch.equal(v,models['torch_parallel_reference'].state_dict()[k]) for k,v in models['native_linear'].state_dict().items()), 'Initial weights differ'
    assert all(not opt.state for opt in opts.values()), 'Expected fresh optimizer states'
    input_hash=hashlib.sha256(p.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
    # The checked first step is warm-up, never included in timed measurements.
    fast=step('native_linear',True);ref=step('torch_parallel_reference',True)
    tolerances={'logits':{'atol':2e-5,'rtol':2e-4},'gradients':{'atol':3e-5,'rtol':1e-3},'updated_parameters':{'atol':2e-5,'rtol':2e-4}}
    torch.testing.assert_close(fast[0],ref[0],**tolerances['logits'])
    errors={'logits':float((fast[0]-ref[0]).abs().max())}
    for index,key in ((1,'gradients'),(2,'updated_parameters')):
        errors[key]=max(float((fast[index][name]-ref[index][name]).abs().max()) for name in fast[index])
        for name in fast[index]:torch.testing.assert_close(fast[index][name],ref[index][name],**tolerances[key],msg=lambda msg:name+': '+msg)
    del fast,ref
    for variant in models:step(variant)
    rows=[]
    for trial in range(7):
        for variant in rng.permutation(list(models)):
            row=step(str(variant));row['trial']=trial;rows.append(row)
    summary={}
    for variant in models:
        vals=np.array([r['wall_seconds'] for r in rows if r['variant']==variant]);summary[variant]={'median_seconds':float(np.median(vals)),'p25_seconds':float(np.quantile(vals,.25)),'p75_seconds':float(np.quantile(vals,.75))}
    assert hashlib.sha256(path.read_bytes()).hexdigest()==checkpoint_hash, 'Saved checkpoint changed during benchmark'
    assert hashlib.sha256(p.numpy().tobytes()+y.numpy().tobytes()).hexdigest()==input_hash, 'Input batch changed'
    out={'input_batch_sha256':input_hash,'gradient_comparison':'After shared norm1 clipping, before optimizer update','optimizer_state':'Fresh identical AdamW groups and empty states; checkpoint optimizer state intentionally not restored','timed_trials_exclude_inspection_copies':True,'status':'completed','scope':'Same complete 765184-parameter, three-layer bidirectional Mamba NAT model, B8/L128, CPU FP32, one pinned thread. Both start at seed1 final weights with fresh identical AdamW optimizers. Two warm-up steps then seven randomized paired training steps. Only the affine scan implementation changes. All models are throwaway copies; no saved checkpoint modified. Forward/backward/AdamW checked within FP32 tolerances, not bitwise identical to associative reduction. No GPU or universal hardware claim.', 'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parity_max_absolute_error':errors,'parity_tolerances':tolerances,'timing':rows,'summary':summary,'whole_model_training_speedup':summary['torch_parallel_reference']['median_seconds']/summary['native_linear']['median_seconds']}
    atomic_result(target,out);signal.alarm(0);print(json.dumps(out['summary']));print('Speedup',out['whole_model_training_speedup'])
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        signal.alarm(0)
        if ACTIVE_TARGET is not None and not ACTIVE_TARGET.exists():
            atomic_result(ACTIVE_TARGET,{'status':'failed_or_cancelled','error':str(exc),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'timing_claim_valid':False})
        raise
