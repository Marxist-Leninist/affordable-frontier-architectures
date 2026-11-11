"""Launch only after parent go-ahead. One CPU core, bounded wall clock."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,math,signal,time,resource
import control
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import LoopedNAT,count
from data_loop import TrainingCorpus,text_sets,fineweb,sha
ROOT=Path(__file__).resolve().parent
STOP=False
def stop(*args):
    global STOP
    STOP=True

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

@torch.inference_mode()
def evaluate(model,p,y,groups):
    losses=[];acc=[];exact=[]
    for i in range(0,len(p),8):
        if STOP: raise TimeoutError('evaluation stopped at run cap or signal')
        z=model(p[i:i+8]); target=y[i:i+8]
        if not torch.isfinite(z).all():raise FloatingPointError('nonfinite evaluation logits')
        losses.extend(F.cross_entropy(z.flatten(0,1),target.flatten(),reduction='none').view(-1,32).mean(1).tolist())
        correct=z.argmax(-1).eq(target);acc.extend(correct.float().mean(1).tolist());exact.extend(correct.all(1).tolist())
    if not np.isfinite(losses).all():raise FloatingPointError('nonfinite evaluation loss')
    return dict(loss_nats=float(np.mean(losses)),bpb=float(np.mean(losses)/math.log(2)),accuracy=float(np.mean(acc)),exact_match=float(np.mean(exact)),per_window_loss=losses,groups=groups.tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--go-after-nat-complete',action='store_true');ap.add_argument('--evaluate-only',action='store_true');ap.add_argument('--kind',choices=['split','whole'],required=True);ap.add_argument('--seed',type=int,choices=[1,2,3],required=True);a=ap.parse_args()
    if not a.go_after_nat_complete: ap.error('Explicit parent launch gate required')
    control.ready()
    cfg=json.loads((ROOT/'protocol.json').read_text())
    expected=json.loads((ROOT/'source_hashes.json').read_text())
    for name,digest in expected.items(): assert sha((ROOT/name).read_bytes())==digest,name
    os.sched_setaffinity(0,{cfg['core']});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(cfg['run_wall_cap_seconds'])
    out=ROOT/'runs'/f'{a.kind}_s{a.seed}'
    if a.evaluate_only:
        evaluate_saved(a,cfg,out);return
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.manual_seed(a.seed);model=LoopedNAT(a.kind)
    assert count(model)==cfg['expected_parameters']
    initial=sha(b''.join(t.detach().numpy().tobytes() for t in model.parameters()))
    opt=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.01},{'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0}],lr=cfg['lr'])
    corpus=TrainingCorpus();rng=np.random.default_rng(90000+a.seed);datahash=hashlib.sha256();cpu=wall=0.;step=0;pending=list(cfg['time_checkpoints_cpu_seconds']);checkpoints=[]
    base=dict(kind=a.kind,seed=a.seed,parameters=count(model),initial_hash=initial,source_hashes=expected,schedule=model.schedule(),input_blocks=len(model.input_blocks),output_blocks=len(model.output_blocks),applied_layers=16+len(model.schedule()),recurrent_applications=len(model.schedule()),protocol=cfg)
    write(out/'progress.json',{**base,'status':'training'})
    while not STOP and (step<cfg['steps'] or pending):
        p,y=corpus.batch(rng,cfg['batch']);datahash.update(p.numpy().tobytes());datahash.update(y.numpy().tobytes());step+=1
        for group in opt.param_groups:group['lr']=cfg['lr']*min(step/100,1.)
        ct=time.process_time();wt=time.perf_counter();opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(p).flatten(0,1),y.flatten());assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();cpu+=time.process_time()-ct;wall+=time.perf_counter()-wt
        tags=[]
        if step==cfg['steps']: tags.append('matched_step')
        for threshold in list(pending):
            if cpu>=threshold: tags.append(f'cpu_{threshold}');pending.remove(threshold)
        for tag in tags:
            meta=dict(tag=tag,step=step,cpu=cpu,wall=wall,batch_sha256=datahash.hexdigest())
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),meta=meta),out/f'{tag}.tmp');(out/f'{tag}.tmp').replace(out/f'{tag}.pt');meta['checkpoint_sha256']=sha((out/f'{tag}.pt').read_bytes());checkpoints.append(meta)
        if step%100==0: write(out/'progress.json',{**base,'status':'training','step':step,'cpu':cpu,'wall':wall,'loss':float(loss.detach())})
    status='training_complete' if step>=cfg['steps'] and not pending else 'capped_or_stopped'
    write(out/'training.json',{**base,'status':status,'steps':step,'cpu':cpu,'wall':wall,'checkpoints':checkpoints,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    write(out/'result.json',{**base,'status':status,'steps':step,'cpu':cpu,'wall':wall})
    if status!='training_complete':raise RuntimeError('Training capped or stopped; no test evaluation allowed')

def evaluate_saved(a,cfg,out):
    # All six completed fits and checkpoint identities precede any held-out reads.
    for seed in cfg['seeds']:
        for kind in cfg['topologies']:
            d=ROOT/'runs'/f'{kind}_s{seed}'
            r=json.loads((d/'training.json').read_text())
            assert r['status']=='training_complete' and r['steps']>=cfg['steps']
            assert r['source_hashes']==json.loads((ROOT/'source_hashes.json').read_text())
            assert {x['tag'] for x in r['checkpoints']}=={'matched_step','cpu_300','cpu_900'}
            for ck in r['checkpoints']:assert sha((d/(ck['tag']+'.pt')).read_bytes())==ck['checkpoint_sha256']
    if (out/'evaluation.json').exists():raise RuntimeError('Refusing duplicate/partial evaluation')
    training=json.loads((out/'training.json').read_text());checkpoints=training['checkpoints'];model=LoopedNAT(a.kind).eval();started=time.monotonic()
    sets={'validation':text_sets('validation',cfg['validation_windows']),'test':text_sets('test'),'fineweb':fineweb()}
    results=[]
    try:
        for meta in checkpoints:
            ck=torch.load(out/f"{meta['tag']}.pt",weights_only=False,map_location='cpu');model.load_state_dict(ck['model'])
            result={**meta,'datasets':{}}
            for name,(p,y,g,manifest) in sets.items():result['datasets'][name]={'metrics':evaluate(model,p,y,g),'manifest':manifest}
            results.append(result);write(out/'evaluation.json',results)
    except Exception as e:
        write(out/'result.json',{**training,'status':'evaluation_failed_or_capped','error':str(e),'evaluated_checkpoints':len(results)});raise
    write(out/'result.json',{**training,'status':'completed','evaluation_wall_seconds':time.monotonic()-started,'evaluated_checkpoints':len(results)})
if __name__=='__main__':main()
