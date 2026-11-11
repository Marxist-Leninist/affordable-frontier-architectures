import copy,hashlib,json,math,os,platform,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from models import LanguageModel,loss
from data import load,Corpus,selections,eval_arrays
P=Path(__file__).resolve().parent
MODES=['ordinary','reversible_graph','reversible_inverse'];SEEDS=[11,22,33];STEPS=500;BS=8;T=64

def sha(b):return hashlib.sha256(b).hexdigest()
def write(name,obj):(P/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def setup():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def freeze():
    docs=load();sel,meta=selections(docs,t=T)
    out={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'modes':MODES,'seeds':SEEDS,'updates':STEPS,'batch':BS,'context':T,'targets_per_equal_data_run':STEPS*BS*T,'eval_selections':sel,'eval_meta':meta,'corpus_article_counts':{k:len(v) for k,v in docs.items()},'train_bytes':sum(map(len,docs['train'])),'source_hashes':{p.name:sha(p.read_bytes()) for p in sorted(P.iterdir()) if p.suffix in ('.py','.md')},'source_manifests':{n:json.loads((P/n).read_text()) for n in ('wikitext_manifest.json','fineweb_manifest.json')},'runtime':{'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),'affinity':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads()},'quality_metric':'Byte cross-entropy nats and bits/byte over final32 targets of each65-byte evaluation window. Lower is better.','limitations':['Small CPU byte-language proxy, not general intelligence.','Ordinary and reversible have equal parameter count and forward module shapes but different internal state dimensions.','Equal update/data is not equal training compute; inverse algorithm performs recomputation.','Same benchmark family has been used in earlier user research; this is a fixed-protocol pilot, not a never-touched benchmark-family claim.']}
    write('protocol_frozen.json',out);print(json.dumps({'frozen':True,'eval_meta':meta}))

def evaluate(m,arrays):
    m.eval();out={}
    with torch.no_grad():
        for name,(x,y) in arrays.items():
            total=0.;n=0;per_window=[]
            for start in range(0,len(x),BS):
                logits=m(x[start:start+BS])[:,T//2:];target=y[start:start+BS,T//2:]
                ce=torch.nn.functional.cross_entropy(logits.reshape(-1,256),target.reshape(-1),reduction='none').reshape(-1,T//2)
                total+=float(ce.sum());n+=ce.numel();per_window.extend(ce.mean(1).tolist())
            out[name]={'nll_nats':total/n,'bits_per_byte':total/n/math.log(2),'scored_bytes':n,'per_window_nll':per_window}
    m.train();return out

def make(seed,mode):
    torch.manual_seed(seed);m=LanguageModel(mode);opt=torch.optim.AdamW(m.parameters(),lr=.002,betas=(.9,.95),weight_decay=.01)
    gen=torch.Generator().manual_seed(seed+123456);return m,opt,gen

def update(m,opt,gen,corpus,step):
    x,y=corpus.batch(gen,BS);opt.zero_grad(set_to_none=True)
    lr=.002*min(1.,(step+1)/30)
    for pg in opt.param_groups:pg['lr']=lr
    l=loss(m(x),y);l.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    if not torch.isfinite(l) or not torch.isfinite(norm):raise RuntimeError('Nonfinite loss or gradient')
    return float(l.detach()),float(norm),sha(x.numpy().tobytes())

def save_ckpt(path,m,opt,gen,step):torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'generator':gen.get_state(),'step':step},path)

def run():
    frozen=json.loads((P/'protocol_frozen.json').read_text());assert frozen['updates']==STEPS
    for name,h in frozen['source_hashes'].items():assert sha((P/name).read_bytes())==h,f'Frozen source changed: {name}'
    docs=load();corpus=Corpus(docs['train'],T);arrays={s:eval_arrays(docs,frozen['eval_selections'],s,T) for s in ('validation','test','fineweb')}
    output={'runs':[],'complete':False,'note':'Fixed last-step endpoints; no validation/test selection. Sampling and whole optimizer update included in training CPU/wall measurements.'}
    for si,seed in enumerate(SEEDS):
        rows=[];order=MODES[si:]+MODES[:si]
        for mode in order:
            m,opt,g=make(seed,mode);initial=evaluate(m,arrays);cpu=0.;wall=0.;trace=[];seqhash=hashlib.sha256()
            for step in range(STEPS):
                c=time.process_time();w=time.perf_counter();value,norm,h=update(m,opt,g,corpus,step);seqhash.update(h.encode());cpu+=time.process_time()-c;wall+=time.perf_counter()-w
                if step%50==0 or step==STEPS-1:trace.append({'step':step+1,'loss':value,'gradient_norm':norm,'train_cpu':cpu})
            path=P/f'checkpoint_s{seed}_{mode}_equal_data.pt';save_ckpt(path,m,opt,g,STEPS)
            row={'seed':seed,'mode':mode,'parameters':sum(p.numel() for p in m.parameters()),'initial':initial,'trace':trace,'equal_data':{'updates':STEPS,'targets':STEPS*BS*T,'train_cpu_seconds':cpu,'train_wall_seconds':wall,'data_sequence_sha256':seqhash.hexdigest(),'metrics':evaluate(m,arrays),'checkpoint':path.name,'checkpoint_sha256':sha(path.read_bytes())}}
            rows.append(row);output['runs'].append(row);write('results.json',output);print(json.dumps({'seed':seed,'mode':mode,'phase':'equal_data','cpu':cpu,'test_bpb':row['equal_data']['metrics']['test']['bits_per_byte']}),flush=True)
            del m,opt
        assert len({r['equal_data']['data_sequence_sha256'] for r in rows})==1
        budget=max(r['equal_data']['train_cpu_seconds'] for r in rows)
        for row in rows:
            mode=row['mode'];m,opt,g=make(seed,mode);state=torch.load(P/row['equal_data']['checkpoint'],map_location='cpu',weights_only=True);m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);g.set_state(state['generator']);step=STEPS
            cpu=row['equal_data']['train_cpu_seconds'];wall=row['equal_data']['train_wall_seconds'];extra_hash=hashlib.sha256()
            while cpu<budget and step<2000:
                c=time.process_time();w=time.perf_counter();value,norm,h=update(m,opt,g,corpus,step);extra_hash.update(h.encode());cpu+=time.process_time()-c;wall+=time.perf_counter()-w;step+=1
            path=P/f'checkpoint_s{seed}_{mode}_equal_cpu.pt';save_ckpt(path,m,opt,g,step)
            row['equal_cpu']={'updates':step,'targets':step*BS*T,'train_cpu_seconds':cpu,'train_wall_seconds':wall,'target_cpu_seconds':budget,'overshoot_cpu_seconds':max(0,cpu-budget),'update_cap_reached':step==2000 and cpu<budget,'additional_data_sha256':extra_hash.hexdigest(),'metrics':evaluate(m,arrays),'checkpoint':path.name,'checkpoint_sha256':sha(path.read_bytes())}
            write('results.json',output);print(json.dumps({'seed':seed,'mode':mode,'phase':'equal_cpu','steps':step,'test_bpb':row['equal_cpu']['metrics']['test']['bits_per_byte']}),flush=True);del m,opt
    output['complete']=True;write('results.json',output)
if __name__=='__main__':
    setup();{'freeze':freeze,'run':run}[sys.argv[1]]()
