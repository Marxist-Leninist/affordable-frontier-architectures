"""Frozen-checkpoint test evaluation, only after all planned training completes."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,math,time,hashlib,fcntl
import numpy as np
import torch
from models_long import NAT
from data_long import ROOT,raw,text_sets,fineweb,toy,sha
from train_long import evaluate,atomic_json

def load(kind,path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=NAT(kind);model.load_state_dict(checkpoint['model']);model.eval()
    return model,checkpoint

def main():
    phase_lock=(ROOT/'.evaluation_timing.lock').open('a')
    fcntl.flock(phase_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'evaluation_results.json').exists():raise RuntimeError('Refusing duplicate held-out evaluation')
    status=json.loads((ROOT/'pipeline_status.json').read_text())
    assert status['status']=='training_complete', 'Do not evaluate test metrics before all planned fits complete.'
    os.sched_setaffinity(0,{1});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    (ROOT/'evaluation').mkdir(exist_ok=True)
    selection=json.loads((ROOT/'selection_long.json').read_text())['selected_lrs']
    protocol=json.loads((ROOT/'protocol_long.json').read_text())
    # Fail closed on every required fit before opening any held-out test cohort.
    for seed in protocol['seeds']:
        for kind in ['transformer','mamba']:
            d=ROOT/'runs'/f'final_{kind}_s{seed}_lr{selection[kind]}'
            r=json.loads((d/'result.json').read_text());t=json.loads((d/'task_result.json').read_text())
            assert r['status']=='completed' and r['steps']==protocol['final_steps']
            assert t['status']=='completed' and t['steps']==protocol['task_steps']
            assert sha((d/'final.pt').read_bytes())==r['final_checkpoint_sha256']
            assert sha((d/'tasks_final.pt').read_bytes())==t['checkpoint_sha256']
    test={};manifests={}
    for name,prepared in [('wikitext',text_sets('test')),('fineweb',fineweb())]:
        p,y,g,m=prepared;test[name]=(p,y,g);manifests[name]={**m,'group_ids':g.tolist()}
    atomic_json(ROOT/'evaluation'/'test_manifest.json',manifests)
    data=np.frombuffer(raw('train'),dtype=np.uint8);counts=np.bincount(data,minlength=256).astype(float)+.5;logp=np.log(counts/counts.sum())
    unigram={name:float(-logp[y.numpy()].mean()/math.log(2)) for name,(p,y,g) in test.items()}
    atomic_json(ROOT/'evaluation'/'unigram_baseline.json',{'training_only_half_count_smoothing':True,'bpb':unigram})
    rows=[]
    for seed in protocol['seeds']:
        for kind in ['transformer','mamba']:
            tag=f'final_{kind}_s{seed}_lr{selection[kind]}';directory=ROOT/'runs'/tag
            result=json.loads((directory/'result.json').read_text());taskresult=json.loads((directory/'task_result.json').read_text())
            assert result['steps']==protocol['final_steps'] and taskresult['steps']==protocol['task_steps']
            for budget,filename in [('fixed_data','final.pt'),('cpu_900s','time_900s.pt')]:
                path=directory/filename
                if not path.exists():
                    if budget=='fixed_data':raise FileNotFoundError(path)
                    rows.append({'kind':kind,'seed':seed,'budget':budget,'status':'unavailable_no_threshold_crossing'});continue
                model,ckpt=load(kind,path);start=time.perf_counter()
                metrics={name:evaluate(model,p,y,batch=16,return_rows=True) for name,(p,y,g) in test.items()}
                row={'kind':kind,'seed':seed,'budget':budget,'checkpoint_step':ckpt['step'],'checkpoint_extra':ckpt['extra'],'checkpoint_sha256':sha(path.read_bytes()),'metrics':metrics,'evaluation_wall_seconds':time.perf_counter()-start,'status':'completed'}
                if budget=='fixed_data':
                    # Fixed diagnostic: same 2048 test targets, mismatched prefixes.
                    p,y,g=test['wikitext'];idx=np.linspace(0,len(p)-1,2048,dtype=np.int64);pp,yy=p[idx],y[idx]
                    row['conditioning_diagnostic']={'examples':2048,'real_prefix':evaluate(model,pp,yy),'shuffled_prefix':evaluate(model,pp.roll(1,0),yy),'selection':'2048 evenly spaced test windows; rotate prefixes by one, targets unchanged'}
                    with torch.inference_mode():
                        generated=model(p[:6]).argmax(-1)
                    row['examples']=[{'prefix':bytes(a.tolist()).decode('utf-8',errors='replace'),'output':bytes(b.tolist()).decode('utf-8',errors='replace')} for a,b in zip(p[:6],generated)]
                atomic_json(ROOT/'evaluation'/f'{kind}_s{seed}_{budget}.json',row);rows.append(row)
                print(json.dumps({'kind':kind,'seed':seed,'budget':budget,'text_bpb':metrics['wikitext']['bpb'],'fineweb_bpb':metrics['fineweb']['bpb']}),flush=True)
            taskmodel,ckpt=load(kind,directory/'tasks_final.pt');metrics={}
            for i,name in enumerate(['copy','reverse','sort','prefixsum_mod10']):
                p,y,_=toy(np.random.default_rng(99000+i),1024,i);metrics[name]=evaluate(taskmodel,p,y,batch=16,return_rows=True)
            row={'kind':kind,'seed':seed,'budget':'separate_task_adaptation','checkpoint_step':ckpt['step'],'metrics':metrics,'checkpoint_sha256':sha((directory/'tasks_final.pt').read_bytes()),'status':'completed'}
            atomic_json(ROOT/'evaluation'/f'{kind}_s{seed}_tasks.json',row);rows.append(row)
    atomic_json(ROOT/'evaluation_results.json',{'selection':selection,'protocol':protocol,'unigram_bpb':unigram,'rows':rows})
    print('ALL_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
