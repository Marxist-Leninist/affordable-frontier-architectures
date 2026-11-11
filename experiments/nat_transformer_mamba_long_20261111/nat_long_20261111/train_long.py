"""Bounded text-only NAT training, followed by a separate capability adaptation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json,math,signal,time,resource
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models_long import NAT,count
from fast_scan import load_library
from data_long import ROOT,TrainingCorpus,text_sets,toy,PREFIX,OUTPUT,sha

STOP=False
def signal_stop(signum,frame):
    global STOP;STOP=True

def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def optimizer(model,lr):
    decay=[];nodecay=[]
    for name,p in model.named_parameters():
        (decay if p.ndim>=2 and not name.endswith('A_log') else nodecay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':.01},{'params':nodecay,'weight_decay':0.}],lr=lr,betas=(.9,.999),eps=1e-8)

@torch.inference_mode()
def evaluate(model,p,y,batch=16,return_rows=False):
    model.eval();losses=[];accuracies=[];exact=[]
    for st in range(0,len(p),batch):
        logits=model(p[st:st+batch],out_len=y.shape[1]);target=y[st:st+batch]
        loss=F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='none').view(-1,y.shape[1])
        pred=logits.argmax(-1);correct=pred.eq(target)
        losses.extend(loss.mean(-1).tolist());accuracies.extend(correct.float().mean(-1).tolist());exact.extend(correct.all(-1).float().tolist())
    model.train()
    out={'n':len(p),'loss_nats':float(np.mean(losses)),'bpb':float(np.mean(losses)/math.log(2)),'byte_accuracy':float(np.mean(accuracies)),'exact_match':float(np.mean(exact))}
    if return_rows:out.update({'per_example_loss':losses,'per_example_accuracy':accuracies,'per_example_exact':exact})
    return out

def save(path,model,opt,step,rng,extra):
    obj={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'numpy_rng_state':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'extra':extra}
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['transformer','mamba'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--lr',type=float,required=True);ap.add_argument('--steps',type=int,required=True);ap.add_argument('--mode',choices=['tune','final'],required=True);ap.add_argument('--core',type=int,required=True);a=ap.parse_args()
    os.sched_setaffinity(0,{a.core});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    protocol=json.loads((ROOT/'protocol_long.json').read_text());expected=json.loads((ROOT/'frozen_sources.json').read_text())
    for name,digest in expected.items():assert sha((ROOT/name).read_bytes())==digest, f'Frozen source changed: {name}'
    tag=f'{a.mode}_{a.kind}_s{a.seed}_lr{a.lr}';out=ROOT/'runs'/tag;out.mkdir(parents=True,exist_ok=False)
    wall_start=time.monotonic();compile_start=time.perf_counter();load_library();compile_seconds=time.perf_counter()-compile_start
    torch.manual_seed(a.seed);model=NAT(a.kind);opt=optimizer(model,a.lr)
    common_sha={name:sha(t.detach().numpy().tobytes()) for name,t in [('embedding',model.embed.weight),('output',model.output.weight)]}
    initial_sha=sha(b''.join(p.detach().numpy().tobytes() for p in model.parameters()))
    corpus=TrainingCorpus();vp,vy,vg,vm=text_sets('validation',limit=1024)
    rng=np.random.default_rng(90000+a.seed);running_hash=hashlib.sha256();curves=[];timing=[];cpu=wall=0.;budgets=list(protocol['matched_training_cpu_seconds']) if a.mode=='final' else []
    base={'kind':a.kind,'seed':a.seed,'lr':a.lr,'mode':a.mode,'core':a.core,'parameters':count(model),'initial_common_hashes':common_sha,'initial_full_hash':initial_sha,'compiler_load_seconds':compile_seconds,'planned_steps':a.steps,'batch_size':protocol['batch_size'],'output_bytes':OUTPUT,'prefix_bytes':PREFIX,'validation':vm,'source_hashes':expected}
    done_step=0;status='running'
    atomic_json(out/'progress.json',{**base,'status':status,'step':0})
    for step in range(1,a.steps+1):
        if STOP or time.monotonic()-wall_start>protocol['run_wall_cap_seconds']:
            status='stopped_at_resource_cap_or_signal';break
        p,y=corpus.batch(rng,protocol['batch_size']);running_hash.update(p.numpy().tobytes());running_hash.update(y.numpy().tobytes())
        # All trials use the same long-run schedule; tuning is its first 1000 updates.
        warm=min(1.,step/100);decay=.1+.9*.5*(1+math.cos(math.pi*min(step,protocol['final_steps'])/protocol['final_steps']))
        for group in opt.param_groups:group['lr']=a.lr*warm*decay
        st=time.perf_counter();ct=time.process_time();opt.zero_grad(set_to_none=True)
        logits=model(p);loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss at {step}')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        elapsed=time.perf_counter()-st;cput=time.process_time()-ct;wall+=elapsed;cpu+=cput;timing.append({'step':step,'wall_seconds':elapsed,'cpu_seconds':cput,'loss_nats':float(loss.detach()),'gradient_norm_before_clip':float(norm)})
        done_step=step
        crossed=[b for b in budgets if cpu>=b]
        for budget in crossed:
            extra={'training_cpu_seconds':cpu,'training_wall_seconds':wall,'threshold_seconds':budget,'batch_sha256':running_hash.hexdigest()}
            save(out/f'time_{budget}s.pt',model,opt,step,rng,extra);budgets.remove(budget)
        evaluate_now=(step==a.steps or (a.mode=='final' and step%2000==0) or (a.mode=='tune' and step%500==0))
        if evaluate_now:
            metric=evaluate(model,vp,vy)
            curves.append({'step':step,'training_cpu_seconds':cpu,'training_wall_seconds':wall,'learning_rate':opt.param_groups[0]['lr'],'validation':metric,'batch_sha256':running_hash.hexdigest()})
            save(out/f'step_{step:06d}.pt',model,opt,step,rng,{'training_cpu_seconds':cpu,'training_wall_seconds':wall,'batch_sha256':running_hash.hexdigest()})
        if step%100==0 or evaluate_now:
            atomic_json(out/'progress.json',{**base,'status':'running','step':step,'training_cpu_seconds':cpu,'training_wall_seconds':wall,'end_to_end_wall_seconds':time.monotonic()-wall_start,'latest_validation':curves[-1] if curves else None})
            print(json.dumps({'tag':tag,'step':step,'cpu':round(cpu,2),'wall':round(wall,2),'loss':round(float(loss.detach()),4),'validation':curves[-1]['validation']['bpb'] if evaluate_now else None}),flush=True)
    else:status='completed'
    save(out/'final.pt',model,opt,done_step,rng,{'training_cpu_seconds':cpu,'training_wall_seconds':wall,'batch_sha256':running_hash.hexdigest(),'status':status})
    result={**base,'status':status,'steps':done_step,'supervised_target_presentations':done_step*protocol['batch_size']*OUTPUT,'training_cpu_seconds':cpu,'training_wall_seconds':wall,'end_to_end_wall_seconds':time.monotonic()-wall_start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'batch_sha256':running_hash.hexdigest(),'validation_curves':curves,'timing':timing,'final_checkpoint_sha256':sha((out/'final.pt').read_bytes())}
    atomic_json(out/'result.json',result);atomic_json(out/'progress.json',{**base,'status':status,'step':done_step,'training_cpu_seconds':cpu,'training_wall_seconds':wall})
    if status!='completed':return
    # Separate capability fine-tune, preserving the text-only final model above.
    if a.mode=='final':
        rng=np.random.default_rng(200000+a.seed);opt=optimizer(model,protocol['task_lr']);task_hash=hashlib.sha256();task_cpu=task_wall=0.;task_steps=0;task_curves=[]
        tval={name:toy(np.random.default_rng(88000+i),256,i)[:2] for i,name in enumerate(['copy','reverse','sort','prefixsum_mod10'])}
        for step in range(1,protocol['task_steps']+1):
            if STOP or time.monotonic()-wall_start>protocol['run_wall_cap_seconds']:status='task_stopped_at_resource_cap_or_signal';break
            p,y,_=toy(rng,protocol['batch_size']);task_hash.update(p.numpy().tobytes());task_hash.update(y.numpy().tobytes())
            st=time.perf_counter();ct=time.process_time();opt.zero_grad(set_to_none=True);logits=model(p);loss=F.cross_entropy(logits.flatten(0,1),y.flatten());loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();task_wall+=time.perf_counter()-st;task_cpu+=time.process_time()-ct;task_steps=step
            if step%500==0 or step==protocol['task_steps']:
                metrics={name:evaluate(model,*pair) for name,pair in tval.items()};task_curves.append({'step':step,'training_cpu_seconds':task_cpu,'training_wall_seconds':task_wall,'validation':metrics});print(json.dumps({'tag':tag,'task_step':step,'task_cpu':round(task_cpu,2)}),flush=True)
                atomic_json(out/'progress.json',{**base,'status':'capability_finetune','text_steps':done_step,'task_step':step,'task_cpu_seconds':task_cpu})
        save(out/'tasks_final.pt',model,opt,task_steps,rng,{'task_cpu_seconds':task_cpu,'task_wall_seconds':task_wall,'task_batch_sha256':task_hash.hexdigest()})
        atomic_json(out/'task_result.json',{'status':status,'steps':task_steps,'lr':protocol['task_lr'],'training_cpu_seconds':task_cpu,'training_wall_seconds':task_wall,'batch_sha256':task_hash.hexdigest(),'validation_curves':task_curves,'checkpoint_sha256':sha((out/'tasks_final.pt').read_bytes())})
        atomic_json(out/'progress.json',{**base,'status':status,'text_steps':done_step,'task_steps':task_steps,'end_to_end_wall_seconds':time.monotonic()-wall_start})

if __name__=='__main__':main()
