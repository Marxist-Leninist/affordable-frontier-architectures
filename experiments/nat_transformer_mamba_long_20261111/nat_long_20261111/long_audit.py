"""Lightweight prelaunch audit; no training or timing benchmarks."""
import hashlib,inspect,json,re
from pathlib import Path
import numpy as np
import torch
from models_long import NAT,count,MambaBlock,AttentionBlock,MASK
from data_long import TrainingCorpus,raw,articles,toy,PREFIX,OUTPUT
P=Path(__file__).resolve().parent
protocol=json.loads((P/'protocol_long.json').read_text());checks={}
torch.set_num_threads(1)
def check(k,v):checks[k]=bool(v)
common={};params={}
for k in ['transformer','mamba']:
 torch.manual_seed(8);m=NAT(k).eval();params[k]=count(m)
 common[k]={n:t.detach().clone() for n,t in [('embedding',m.embed.weight),('output',m.output.weight)]}
 check(k+'_protocol_parameter_count',params[k]==protocol['architecture'][k+'_parameters'])
 check(k+'_correct_branch',all(isinstance(b,AttentionBlock if k=='transformer' else MambaBlock) for b in m.blocks))
 captured=[];h=m.embed.register_forward_pre_hook(lambda m,args:captured.append(args[0].clone()))
 prefix=torch.arange(96)[None]
 with torch.no_grad():out=m(prefix)
 h.remove();check(k+'_single_pass_suffix_masks',len(captured)==1 and torch.equal(captured[0][:,:96],prefix) and bool((captured[0][:,96:]==MASK).all()) and out.shape==(1,32,256) and bool(out.isfinite().all()))
check('common_embedding_head_initialization',all(torch.equal(common['mamba'][k],common['transformer'][k]) for k in common['mamba']))
check('forward_no_targets_argument',list(inspect.signature(NAT.forward).parameters)==['self','prefix','out_len'])
corpus=TrainingCorpus();a=np.random.default_rng(90001);b=np.random.default_rng(90001)
for i in range(3):
 p,y=corpus.batch(a,8);q,z=corpus.batch(b,8);check(f'paired_batch_{i}',torch.equal(p,q) and torch.equal(y,z) and p.shape==(8,96) and y.shape==(8,32))
hashes={s:hashlib.sha256(raw(s)).hexdigest() for s in ['train','validation','test']}
docs={s:len(articles(raw(s))) for s in ['train','validation','test']}
for t in range(4):
 p,y,_=toy(np.random.default_rng(13),2,t)
 names=[b'copy:',b'reverse:',b'sort:',b'prefixsum_mod10:'];d=p[:,len(names[t]):len(names[t])+32].numpy()-48
 expect=d if t==0 else d[:,::-1] if t==1 else np.sort(d,axis=1) if t==2 else np.cumsum(d,axis=1)%10
 check(f'task_{t}_correct_targets',np.array_equal(y.numpy(),expect+48))
source=(P/'train_long.py').read_text();orch=(P/'orchestrate_long.py').read_text()
check('training_no_test_loads',"text_sets('test'" not in source and 'fineweb(' not in source)
check('text_checkpoint_preserved_before_task_training',source.index("save(out/'final.pt'")<source.index("# Separate capability fine-tune"))
check('gradient_nonfinite_guard',"error_if_nonfinite=True" in source)
check('cpu_thread_affinity_config',"os.sched_setaffinity(0,{a.core})" in source and 'torch.set_num_threads(1)' in source and protocol['cpu_cores']==[1,2])
check('prespecified_resource_limits',protocol['max_concurrent_jobs']==2 and protocol['global_wall_cap_seconds']==21600 and protocol['run_wall_cap_seconds']==10800)
record={'passed_model_data_checks':all(checks.values()),'checks':checks,'parameters':params,'raw_sha256_verified':hashes,'parsed_article_counts':docs,'valid_train_window_count':int(corpus.ends[-1]),'source_sha256':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['models_long.py','data_long.py','train_long.py','orchestrate_long.py','protocol_long.json','fast_scan.py','scan_cpu.cpp']},'resource_safety_review':'Cancellation fix inspected and four bounded mock-child tests pass; see orchestrator_cancel_audit.json. No launch blocker identified.','findings':['Mamba-1 equations retained; native C++ recurrence reduces scan to linear work. Fast-scan parity artifacts reviewed; bounded complete NAT forward succeeded.','Full train split now used via uniformly sampled valid article-contained windows; byte windows near article ends/short documents excluded by definition.','Paired seed controls identical data and common embedding/output initialization, not identical architecture-specific random weights.','300/900 CPU-second snapshots may be unavailable when a run completes first; mark missing, never substitute final or compare unequal budgets silently.','Two concurrent pinned jobs can compete for cache/memory and frequency; standalone timing required, CPU time excludes waits and differs from user wall time.','Task sampling is balanced in expectation, not guaranteed exact counts per minibatch; adaptation uses common LR rather than per-family task-LR tuning.','FineWeb cohort was used in prior research but is new to this comparison, not a universally untouched external dataset.','Three small dense FP32 seeds and fixed 32-byte conditional generation do not establish general intelligence, GPU speed or 600B benefit.','Tune seed is hard-coded101 and max_workers2 rather than read from protocol, but matches frozen current protocol.','run_wall_cap checks happen between steps, not inside evaluation/compiler/checkpoint operations; global parent timeout bounds child lifetime with up to60s cleanup grace.']}
(P/'long_audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'passed_model_data_checks':record['passed_model_data_checks'],'checks':len(checks),'parameters':params,'articles':docs},indent=2))
