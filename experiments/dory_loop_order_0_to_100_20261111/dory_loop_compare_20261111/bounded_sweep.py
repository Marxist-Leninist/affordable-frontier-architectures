"""Frozen 0..100 inference extrapolation. No training or external publication."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,time,signal,hashlib,math
PROCESS_STARTED=time.monotonic()
import control
from bounded_runtime_plan import validate_protocol,verify_calibration,plan,sha as file_sha
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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--go-after-nat-complete',action='store_true');ap.add_argument('--calibration-dir',type=Path,default=ROOT/'sweep_results');ap.add_argument('--calibration-receipts',type=Path,required=True);a=ap.parse_args()
    if not a.go_after_nat_complete:ap.error('Explicit authorization required')
    started=PROCESS_STARTED;lock=control.ownership()
    cfg=json.loads((ROOT/'bounded_sweep_protocol.json').read_text());validate_protocol(cfg)
    original_hashes=json.loads((ROOT/'source_hashes.json').read_text());new_hashes=json.loads((ROOT/'bounded_source_hashes.json').read_text())
    for names in (original_hashes,new_hashes):
        for name,digest in names.items():assert file_sha(ROOT/name)==digest,name
    receipt_manifest=json.loads(a.calibration_receipts.read_text())
    original=verify_calibration(a.calibration_dir,receipt_manifest,file_sha(ROOT/'source_hashes.json'))
    prior=original['elapsed_debited_seconds'];global_deadline=started+cfg['global_wall_cap_seconds']-prior
    os.sched_setaffinity(0,{cfg['core']});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,max(.001,global_deadline-time.monotonic()))
    out=ROOT/'bounded_sweep_results';out.mkdir(exist_ok=False)
    # All six completed standard evaluations and all18 checkpoint identities,
    # not just the six sweep checkpoints, precede held-out loading.
    verified=[];paired={}
    for seed in cfg['seeds']:
        for kind in cfg['topologies']:
            d=ROOT/'runs'/f'{kind}_s{seed}';training=json.loads((d/'training.json').read_text());result=json.loads((d/'result.json').read_text());ev=json.loads((d/'evaluation.json').read_text())
            assert training['status']=='training_complete' and result['status']=='completed' and result['evaluated_checkpoints']==3
            assert training['source_hashes']==result['source_hashes']==original_hashes
            assert training['parameters']==764304 and training['protocol']['hidden']==117 and training['protocol']['loops']==2
            assert training['protocol']['input_blocks']==training['protocol']['output_blocks']==8 and training['protocol']['recurrent_unique_blocks']==20
            assert training['schedule']==(list(range(10))*2+list(range(10,20))*2 if kind=='split' else list(range(20))*2)
            tags={'matched_step','cpu_300','cpu_900'}
            assert len(ev)==len(training['checkpoints'])==3 and {v['tag'] for v in ev}=={v['tag'] for v in training['checkpoints']}==tags
            for ck in training['checkpoints']:
                assert file_sha(d/(ck['tag']+'.pt'))==ck['checkpoint_sha256']
                evaluated=next(e for e in ev if e['tag']==ck['tag'])
                assert evaluated['checkpoint_sha256']==ck['checkpoint_sha256']
                assert set(evaluated['datasets'])=={'validation','test','fineweb'}
                for entry in evaluated['datasets'].values():
                    metrics=entry['metrics'];assert all(math.isfinite(metrics[k]) for k in ['loss_nats','bpb','accuracy','exact_match'])
            matched=next(c for c in training['checkpoints'] if c['tag']=='matched_step')
            assert matched['step']==training['protocol']['steps']==2000
            assert matched['checkpoint_sha256']==original['checkpoint_hashes'][f'{kind}_{seed}']
            paired[(seed,kind)]=(training['initial_hash'],matched['batch_sha256'])
            verified.append({'kind':kind,'seed':seed,'training_receipt_sha256':file_sha(d/'training.json'),'standard_result_sha256':file_sha(d/'result.json'),'standard_evaluation_sha256':file_sha(d/'evaluation.json')})
    for seed in cfg['seeds']:assert paired[(seed,'split')]==paired[(seed,'whole')]
    jobs=[]
    for seed in cfg['seeds']:
        for kind in (cfg['topologies'] if seed%2 else cfg['topologies'][::-1]):
            ckpath=ROOT/'runs'/f'{kind}_s{seed}'/cfg['checkpoint'];ck=torch.load(ckpath,weights_only=False,map_location='cpu');model=LoopedNAT(kind);model.load_state_dict(ck['model']);model.eval()
            rows=[dict(phase=phase,loops=r,trained_loops=2,training_loop_distribution='in_distribution' if r==2 else 'extrapolation',unique_parameters=764304,invoked_module_parameters=353424 if r==0 else 764304,applied_layers=16+20*r,recurrent_applications=20*r,datasets={d:{'status':'not_run'} for d in cfg['domains']}) for phase,counts in [('all101',cfg['counts']),('selected',cfg['selected_counts'])] for r in counts]
            jobs.append(dict(kind=kind,seed=seed,model=model,rows=rows,elapsed=0.,checkpoint_sha256=file_sha(ckpath),checkpoint_meta=ck['meta']));del ck
    selection=plan(original,time.monotonic()-started)
    write(out/'amendment_receipt.json',dict(protocol=cfg,original_source_hashes=original_hashes,amended_source_hashes=new_hashes,calibration_receipt_manifest=receipt_manifest,standard_evaluations_verified=verified,heldout_loaded=False,reason='Synthetic runtime only: original scope exceeded caps; preserve all101 counts/all6models, reduce extra cohort to64windows at0/2/100'))
    write(out/'cohort_selection.json',selection)
    if selection['status']!='selected' or STOP or time.monotonic()>=global_deadline:
        write(out/'blocker.json',{'status':'blocked_no_fit_or_cancelled','heldout_loaded':False});return
    sets={'test':text_sets('test'),'fineweb':fineweb()}
    cohorts={phase:{k:cohort(v,n) for k,v in sets.items()} for phase,n in [('all101',8),('selected',64)]}
    write(out/'manifest.json',dict(protocol=cfg,source_hashes=original_hashes,amended_source_hashes=new_hashes,selection=selection,cohorts={p:{d:v[3] for d,v in pairs.items()} for p,pairs in cohorts.items()}))
    def persist(job):write(out/f"{job['kind']}_s{job['seed']}.json",{k:v for k,v in job.items() if k!='model'})
    for job in jobs:persist(job)
    ordinal=0
    for phase,counts in [('all101',cfg['counts']),('selected',cfg['selected_counts'])]:
        for loops in counts:
            order=jobs[ordinal%6:]+jobs[:ordinal%6];ordinal+=1
            for job in order:
                row=next(r for r in job['rows'] if r['phase']==phase and r['loops']==loops);remaining=1200-job['elapsed']
                if STOP or time.monotonic()>=global_deadline or remaining<=0:
                    for d in cfg['domains']:row['datasets'][d]={'status':'missing_resource_cap'}
                    persist(job);continue
                begin=time.monotonic();deadline=min(global_deadline,begin+remaining);job['model'].loops=loops
                assert len(job['model'].schedule())==20*loops
                for domain,(p,y,g,meta) in cohorts[phase].items():row['datasets'][domain]=evaluate(job['model'],p,y,g,8,deadline)
                job['elapsed']+=time.monotonic()-begin;persist(job)
    statuses=[]
    for job in jobs:
        vals=[v['status'] for r in job['rows'] for v in r['datasets'].values()]
        statuses.append(dict(seed=job['seed'],kind=job['kind'],completed_cells=vals.count('completed'),nonfinite_cells=sum(v.startswith('nonfinite') for v in vals),missing_or_capped_cells=sum(v!='completed' and not v.startswith('nonfinite') for v in vals),total_cells=len(vals)))
    write(out/'status.json',dict(runs=statuses,prior_elapsed_debited_seconds=prior,new_elapsed_seconds=time.monotonic()-started,combined_elapsed_seconds=prior+time.monotonic()-started,global_capped=STOP or time.monotonic()>=global_deadline))
if __name__=='__main__':main()
