"""Frozen 0..100 inference extrapolation. No training or external publication."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,time,signal,hashlib,math
import control
from runtime_plan import COUNTS as CALIBRATION_COUNTS,choose,validate_protocol
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import LoopedNAT,count
from data_loop import text_sets,fineweb,sha
ROOT=Path(__file__).resolve().parent
STOP=False
def stop(*args):
    global STOP
    STOP=True

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def cohort(pair,n):
    p,y,g,meta=pair
    idx=np.linspace(0,len(p)-1,min(n,len(p)),dtype=np.int64)
    p,y,g=p[idx],y[idx],g[idx]
    assert len(set(g.tolist()))>=4, 'Cohort must cover at least four documents'
    return p,y,g,dict(source=meta,indices=idx.tolist(),selection_sha256=sha(torch.cat((p,y),1).numpy().tobytes()),windows=len(idx))

@torch.inference_mode()
def evaluate(model,p,y,g,batch,deadline):
    cpu=wall=0.;losses=[];accuracies=[];exact=[]
    for i in range(0,len(p),batch):
        if STOP or time.monotonic()>=deadline: return {'status':'capped','evaluated_windows':len(losses)}
        c=time.process_time();w=time.perf_counter();z=model(p[i:i+batch]);wall+=time.perf_counter()-w;cpu+=time.process_time()-c
        if not torch.isfinite(z).all():return {'status':'nonfinite_logits','evaluated_windows':len(losses),'forward_cpu_seconds':cpu,'forward_wall_seconds':wall}
        target=y[i:i+batch];ls=F.cross_entropy(z.flatten(0,1),target.flatten(),reduction='none').view(-1,32).mean(1)
        if not torch.isfinite(ls).all():return {'status':'nonfinite_loss','evaluated_windows':len(losses)}
        losses.extend(ls.tolist());corr=z.argmax(-1).eq(target);accuracies.extend(corr.float().mean(1).tolist());exact.extend(corr.all(1).tolist())
    return dict(status='completed',evaluated_windows=len(losses),suffix_bpb=float(np.mean(losses)/math.log(2)),byte_accuracy=float(np.mean(accuracies)),exact_32_byte_match=float(np.mean(exact)),per_window_loss_nats=losses,groups=g.tolist(),forward_cpu_seconds=cpu,forward_wall_seconds=wall)

class CalibrationBlocked(RuntimeError):pass

def calibrate(jobs,out,started,global_deadline):
    """Synthetic only; freeze receipts before any held-out loader call."""
    begin=time.monotonic();deadline=min(global_deadline,begin+180);samples=[];warmups=[]
    def alarm(*args):raise CalibrationBlocked('Calibration wall cap reached')
    old=signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    def receipt():
        write(out/'calibration_raw.json',dict(synthetic_input='arange(8*96) modulo256',synthetic_target='arange(8*32) modulo256',batch=8,counts=list(CALIBRATION_COUNTS),repetitions=2,warmup_loop_count=2,samples=samples,warmups=warmups,elapsed_seconds=time.monotonic()-begin))
    try:
        p=torch.arange(8*96,dtype=torch.long).reshape(8,96)%256
        y=torch.arange(8*32,dtype=torch.long).reshape(8,32)%256
        g=np.arange(8)
        for job in jobs:
            if STOP:raise CalibrationBlocked('Cancelled before calibration')
            job['model'].loops=2
            result=evaluate(job['model'],p,y,g,8,deadline)
            warmups.append(dict(job=f"{job['kind']}_{job['seed']}",status=result['status']))
            if result['status']!='completed':raise CalibrationBlocked('Incomplete/nonfinite warmup')
            receipt()
        for ordinal,loops in enumerate(CALIBRATION_COUNTS):
            order=jobs[ordinal%6:]+jobs[:ordinal%6]
            for job in order:
                for repeat in range(2):
                    if STOP or time.monotonic()>=deadline:raise CalibrationBlocked('Calibration cancelled/capped')
                    job['model'].loops=loops;tick=time.monotonic()
                    result=evaluate(job['model'],p,y,g,8,deadline)
                    samples.append(dict(job=f"{job['kind']}_{job['seed']}",loops=loops,repeat=repeat,seconds=time.monotonic()-tick,status=result['status']))
                    receipt()
                    if result['status']!='completed':raise CalibrationBlocked('Incomplete/nonfinite calibration')
        selection=choose(samples,[f"{j['kind']}_{j['seed']}" for j in jobs],time.monotonic()-started,time.monotonic()-begin)
        selection['checkpoint_hashes']={f"{j['kind']}_{j['seed']}":j['checkpoint_sha256'] for j in jobs}
        write(out/'cohort_selection.json',selection)
        if selection['status']!='selected':
            write(out/'blocker.json',{'status':'blocked_no_fit','heldout_loaded':False,'detail':'No candidate satisfies prespecified runtime budgets'})
            return None
        return selection
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        receipt();write(out/'blocker.json',{'status':'blocked_calibration','heldout_loaded':False,'reason':str(exc)})
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        signal.setitimer(signal.ITIMER_REAL,max(.001,global_deadline-time.monotonic()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--go-after-nat-complete',action='store_true');a=ap.parse_args()
    if not a.go_after_nat_complete:ap.error('Explicit parent go-ahead required after NAT and standalone timing')
    lock=control.ownership()
    cfg=json.loads((ROOT/'sweep_protocol.json').read_text());hashes=json.loads((ROOT/'source_hashes.json').read_text())
    validate_protocol(cfg)
    for name,digest in hashes.items():assert sha((ROOT/name).read_bytes())==digest,name
    os.sched_setaffinity(0,{cfg['core']});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(cfg['global_wall_cap_seconds'])
    out=ROOT/'sweep_results';out.mkdir(exist_ok=False)
    started=time.monotonic();global_deadline=started+cfg['global_wall_cap_seconds']
    # Verify all six completed fits and trained topology before held-out access.
    for seed in cfg['seeds']:
        for kind in cfg['topologies']:
            d=ROOT/'runs'/f'{kind}_s{seed}';r=json.loads((d/'training.json').read_text())
            assert r['status']=='training_complete' and r['source_hashes']==hashes
            assert r['protocol']['loops']==cfg['trained_loop_count']==2
            assert r['parameters']==764304 and r['protocol']['hidden']==117
            assert r['protocol']['input_blocks']==r['protocol']['output_blocks']==8
            assert r['protocol']['recurrent_unique_blocks']==20
            assert r['schedule']==(list(range(10))*2+list(range(10,20))*2 if kind=='split' else list(range(20))*2)
            receipt=next(x for x in r['checkpoints'] if x['tag']=='matched_step')
            assert receipt['step']==r['protocol']['steps']
            assert sha((d/cfg['checkpoint']).read_bytes())==receipt['checkpoint_sha256']
    jobs=[]
    for seed in cfg['seeds']:
        kinds=cfg['topologies'] if seed%2 else cfg['topologies'][::-1]
        for kind in kinds:
            ckpath=ROOT/'runs'/f'{kind}_s{seed}'/cfg['checkpoint']
            records=[dict(phase=phase,loops=r,trained_loops=2,training_loop_distribution='in_distribution' if r==2 else 'extrapolation',unique_parameters=764304,invoked_module_parameters=353424 if r==0 else 764304,applied_layers=16+20*r,recurrent_applications=20*r,datasets={name:{'status':'not_run'} for name in cfg['domains']}) for phase,counts in [('all101',cfg['counts']),('selected',cfg['selected_counts'])] for r in counts]
            job=dict(seed=seed,kind=kind,rows=records,elapsed=0.,checkpoint_sha256=None,checkpoint_meta=None,model=None)
            if ckpath.exists():
                ck=torch.load(ckpath,weights_only=False,map_location='cpu');model=LoopedNAT(kind);model.load_state_dict(ck['model']);model.eval()
                job.update(model=model,checkpoint_sha256=sha(ckpath.read_bytes()),checkpoint_meta=ck['meta']);del ck
            else:
                for row in records:
                    for name in cfg['domains']:row['datasets'][name]={'status':'missing_checkpoint'}
            jobs.append(job)
    selection=calibrate(jobs,out,started,global_deadline)
    if selection is None:return
    if STOP or time.monotonic()>=global_deadline:
        write(out/'blocker.json',{'status':'blocked_cancelled_or_capped','heldout_loaded':False})
        return
    # Selection and raw timing receipts exist before any held-out data is loaded.
    sets={'test':text_sets('test'),'fineweb':fineweb()}
    cohorts={phase:{k:cohort(v,n) for k,v in sets.items()} for phase,n in [('all101',selection['windows_per_domain']),('selected',cfg['selected_windows_per_domain'])]}
    write(out/'manifest.json',{'protocol':cfg,'source_hashes':hashes,'selection':selection,'cohorts':{phase:{name:v[3] for name,v in pairs.items()} for phase,pairs in cohorts.items()},'timing_note':'Independent full forward calls; no cross-count latent reuse. Synthetic calibration only determines cohort size; held-out runtime is descriptive.'})
    def persist(job):
        write(out/f"{job['kind']}_s{job['seed']}.json",{k:v for k,v in job.items() if k!='model'})
    for job in jobs:persist(job)
    # Cover every count across every checkpoint before optional larger evaluation.
    # Rotate starting job every round; every job visits every rank in six rounds.
    ordinal=0
    for phase,counts in [('all101',cfg['counts']),('selected',cfg['selected_counts'])]:
        for loops in counts:
            order=jobs[ordinal%len(jobs):]+jobs[:ordinal%len(jobs)]
            ordinal+=1
            for job in order:
                row=next(r for r in job['rows'] if r['phase']==phase and r['loops']==loops)
                if job['model'] is None:continue
                remaining=cfg['per_checkpoint_wall_cap_seconds']-job['elapsed']
                if STOP or time.monotonic()>=global_deadline or remaining<=0:
                    for name in cfg['domains']:row['datasets'][name]={'status':'missing_resource_cap'}
                    persist(job);continue
                begin=time.monotonic();deadline=min(global_deadline,begin+remaining);model=job['model'];model.loops=loops
                assert len(model.schedule())==20*loops
                for domain,(p,y,g,meta) in cohorts[phase].items():row['datasets'][domain]=evaluate(model,p,y,g,cfg['batch'],deadline)
                job['elapsed']+=time.monotonic()-begin;persist(job)
    statuses=[]
    for job in jobs:
        vals=[v['status'] for row in job['rows'] for v in row['datasets'].values()]
        statuses.append(dict(seed=job['seed'],kind=job['kind'],completed_cells=vals.count('completed'),nonfinite_cells=sum(v.startswith('nonfinite') for v in vals),missing_or_capped_cells=sum(v!='completed' and not v.startswith('nonfinite') for v in vals),total_cells=len(vals)))
    write(out/'status.json',dict(runs=statuses,end_to_end_wall_seconds=time.monotonic()-started,global_capped=STOP or time.monotonic()>=global_deadline))
if __name__=='__main__':main()
