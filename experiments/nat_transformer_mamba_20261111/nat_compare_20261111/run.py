import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import NAT, count, PAD

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'frontier_research_v8'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
def write(name,obj): (HERE/name).write_text(json.dumps(obj,indent=2)+'\n')
def digest(b): return hashlib.sha256(b).hexdigest()
def corpus(split):
    b=(DATA/(split+'.txt')).read_bytes()
    manifest=json.loads((DATA/'data_manifest.json').read_text())
    assert digest(b)==manifest['splits'][split]['text_sha256']
    return b

def toy(rng,n,task=None):
    digits=rng.integers(0,10,(n,16),dtype=np.int64)+48
    tasks=rng.integers(0,3,n) if task is None else np.full(n,task)
    p=np.full((n,32),PAD,dtype=np.int64); y=np.empty((n,16),dtype=np.int64)
    for i,t in enumerate(tasks):
        name=[b'copy:',b'reverse:',b'sort:'][t]
        p[i,:len(name)]=np.frombuffer(name,dtype=np.uint8)
        p[i,len(name):len(name)+16]=digits[i]
        y[i]=digits[i] if t==0 else digits[i,::-1] if t==1 else np.sort(digits[i])
    return torch.from_numpy(p),torch.from_numpy(y),tasks

def text_eval(split,n=512):
    b=corpus(split); train=corpus('train')[:2_000_000]
    # Deterministic spread over official split, removing exact full-window matches.
    prior=set(); rows=[]; rejected=0
    for st in np.linspace(0,len(b)-48,n*2,dtype=np.int64):
        win=b[st:st+48]
        if win in train or win in prior: rejected+=1;continue
        prior.add(win); rows.append(list(win))
        if len(rows)==n:break
    assert len(rows)==n
    a=torch.tensor(rows,dtype=torch.long)
    return a[:,:32],a[:,32:],{'split':split,'n':n,'rejected_exact48_train_or_within':rejected,'sha256':digest(a.numpy().tobytes()),'selection':'first 512 acceptable from 1024 evenly spaced starts'}

def eval_sets(split):
    p,y,meta=text_eval(split)
    result={'text':(p,y)}
    for task in range(3):
        p,y,_=toy(np.random.default_rng((22000 if split=='validation' else 33000)+task),256,task)
        result[['copy','reverse','sort'][task]]=(p,y)
    return result,meta

@torch.inference_mode()
def evaluate(model,sets):
    model.eval();result={}
    for name,(p,y) in sets.items():
        losses=[];accs=[];exact=[]
        for st in range(0,len(p),32):
            logits=model(p[st:st+32]);target=y[st:st+32]
            loss=F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='none').view(-1,16)
            pred=logits.argmax(-1)
            losses.extend(loss.mean(-1).tolist());accs.extend((pred==target).float().mean(-1).tolist());exact.extend((pred==target).all(-1).float().tolist())
        result[name]={'loss_nats':float(np.mean(losses)),'bpb':float(np.mean(losses)/math.log(2)),'token_accuracy':float(np.mean(accs)),'exact_match':float(np.mean(exact)),'per_example_loss':losses,'per_example_accuracy':accs}
    model.train();return result

def score(metrics):
    return .5*metrics['text']['loss_nats']+sum(metrics[x]['loss_nats'] for x in ['copy','reverse','sort'])/6

def train(kind,seed,lr,steps,validation,tag,save_time=None):
    torch.manual_seed(seed); model=NAT(kind); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0)
    trainbytes=np.frombuffer(corpus('train')[:2_000_000],dtype=np.uint8)
    rng=np.random.default_rng(100000+seed)
    xs=[];ys=[]
    for step in range(steps):
        starts=rng.integers(0,len(trainbytes)-48,8)
        a=np.stack([trainbytes[s:s+48] for s in starts]).astype(np.int64)
        p,y,_=toy(rng,8)
        xs.append(torch.cat((torch.from_numpy(a[:,:32]),p)))
        ys.append(torch.cat((torch.from_numpy(a[:,32:]),y)))
    batchhash=digest(torch.stack(xs).numpy().tobytes()+torch.stack(ys).numpy().tobytes())
    curve=[];elapsed=0.;cpu=0.;times=[];timed_saved=False
    for step,(p,y) in enumerate(zip(xs,ys),1):
        st=time.perf_counter(); ct=time.process_time()
        opt.zero_grad(set_to_none=True); logits=model(p)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten());loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        elapsed+=time.perf_counter()-st;cpu+=time.process_time()-ct;times.append(time.perf_counter()-st)
        if save_time and not timed_saved and elapsed>=save_time:
            torch.save(model.state_dict(),HERE/'checkpoints'/f'{tag}_time.pt')
            timed_saved=True; curve.append({'step':step,'train_wall_seconds':elapsed,'train_cpu_seconds':cpu,'type':'time_budget','validation':evaluate(model,validation)})
        if step%100==0 or step==steps:
            metrics=evaluate(model,validation)
            curve.append({'step':step,'train_wall_seconds':elapsed,'train_cpu_seconds':cpu,'type':'fixed_steps','validation':metrics})
            print(json.dumps({'tag':tag,'step':step,'seconds':round(elapsed,2),'val':round(score(metrics),4)}),flush=True)
    path=HERE/'checkpoints'/f'{tag}.pt';torch.save(model.state_dict(),path)
    result={'kind':kind,'seed':seed,'lr':lr,'steps':steps,'parameters':count(model),'target_presentations':steps*16*16,'input_positions':steps*16*48,'batch_sha256':batchhash,'wall_seconds':elapsed,'cpu_seconds':cpu,'targets_per_second':steps*256/elapsed,'step_ms_p50':float(np.median(times[3:])*1000),'step_ms_p95':float(np.percentile(times[3:],95)*1000),'curve':curve,'checkpoint_sha256':digest(path.read_bytes())}
    write(tag+'.json',result);return result

@torch.inference_mode()
def timing(models,sets):
    # Random paired order, model forward plus all-16 argmax; no inter-step token loop.
    rng=np.random.default_rng(555); rows=[]
    for batch in [1,16]:
        p=sets['text'][0][:batch]
        for m in models.values():
            m.eval()
            for _ in range(5):m(p).argmax(-1)
        for trial in range(60):
            for name in rng.permutation(list(models)):
                st=time.perf_counter();models[name](p).argmax(-1);sec=time.perf_counter()-st
                rows.append({'name':name,'batch':batch,'trial':trial,'ms':sec*1000})
    return rows

def main():
    (HERE/'checkpoints').mkdir(exist_ok=True)
    validation,vm=eval_sets('validation')
    protocol={'created_before_training':True,'date':'2026-11-11','task':'single-pass full 16-byte suffix prediction from 32-byte prefix; no gold suffix input, no diffusion/refinement','mixture':'8 WikiText examples + 8 fresh random copy/reverse/sort examples per step','source':json.loads((DATA/'data_manifest.json').read_text()),'train_prefix_bytes':2000000,'validation':vm,'tuning':{'seed':101,'lrs':[.001,.003],'steps':300,'selection':'0.5 text loss + 0.5 mean of 3 toy task losses'},'final':{'seeds':[1,2,3],'steps':800,'matched_time_seconds':5},'params':{k:count(NAT(k)) for k in ['transformer','mamba']},'architecture':'2 layers, width 48, full precision; attention 4 heads + RoPE + squared ReLU FFN hidden266; Mamba1 independent forward/reverse dstate8 expand2 conv4, associative scan, no FFN; both RMSNorm and identical embeddings, sinusoidal positions, output heads','host':{'cpu':platform.processor(),'torch':torch.__version__,'threads':1,'cuda':False},'limitations':['tiny CPU reference, not fused-Mamba speed','full-precision dense architecture ablation, not exact quantized routed AGILLM or a 600B benchmark','fixed output length, not open-ended EOS/length generation','narrow capabilities, not intelligence','no equal-FLOP assumption']}
    write('protocol.json',protocol)
    tuning=[]
    for lr in [.001,.003]:
        for kind in ['transformer','mamba']:
            tuning.append(train(kind,101,lr,300,validation,f'tune_{kind}_{lr}'))
    selected={k:min([r for r in tuning if r['kind']==k],key=lambda r:score(r['curve'][-1]['validation']))['lr'] for k in ['transformer','mamba']}
    write('selection.json',selected)
    finals=[]
    for seed in [1,2,3]:
        # Counterbalance family ordering by seed.
        for kind in (['transformer','mamba'] if seed%2 else ['mamba','transformer']):
            finals.append(train(kind,seed,selected[kind],800,validation,f'final_{kind}_{seed}',save_time=5))
    # Test data/metrics only enter after all final fits and LR selection are frozen.
    test,tm=eval_sets('test');write('test_manifest.json',tm)
    models={}
    for result in finals:
        k,seed=result['kind'],result['seed'];tag=f'final_{k}_{seed}'
        model=NAT(k);model.load_state_dict(torch.load(HERE/'checkpoints'/f'{tag}.pt',weights_only=True))
        result['test_fixed_steps']=evaluate(model,test)
        models[f'{k}_{seed}']=model
        mtime=NAT(k);mtime.load_state_dict(torch.load(HERE/'checkpoints'/f'{tag}_time.pt',weights_only=True))
        result['test_fixed_time']=evaluate(mtime,test)
        result['time_budget_receipt']=next(x for x in result['curve'] if x['type']=='time_budget')
        with torch.inference_mode():
            examples=[]
            for p in test['text'][0][:6]:
                generated=model(p[None]).argmax(-1)[0]
                examples.append({'prefix':bytes(p.tolist()).decode('utf-8',errors='replace'),'output':bytes(generated.tolist()).decode('utf-8',errors='replace')})
        result['text_examples']=examples
        write(tag+'.json',result)
    write('results.json',{'selection':selected,'runs':finals,'inference_trials':timing(models,test)})
    print('DONE',flush=True)

if __name__=='__main__':main()
