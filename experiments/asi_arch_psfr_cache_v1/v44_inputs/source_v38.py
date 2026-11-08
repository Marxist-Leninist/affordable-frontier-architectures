"""Architecture-first GQA/MQA uptraining, matched full-attention controls."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import torch,numpy as np,json,math,time,hashlib,sys,random,resource,platform
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from base_models import LM,Attention,count
P=Path(__file__).resolve().parent;ROOT=P.parent;SEEDS=(1,2,3);HEADS=(4,2,1);STEPS=320
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def write(name,o):(P/name).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')

class GQA(Attention):
    def __init__(self,kvh):
        super().__init__();self.kvh=kvh;self.qkv=nn.Linear(64,64+32*kvh,bias=False)
    def rope(self,x,offset):
        length=offset+x.shape[-2]
        if not hasattr(self,'cs') or self.cs[0].dtype!=x.dtype or len(self.cs[0])<length:
            with torch.inference_mode(False),torch.no_grad():
                a=torch.arange(max(2048,length),dtype=x.dtype)[:,None]*10000.**(-torch.arange(0,16,2,dtype=x.dtype)/16)
                self.cs=(a.cos(),a.sin())
        c,s=(z[offset:length] for z in self.cs);u,v=x[...,::2],x[...,1::2]
        return torch.stack((u*c-v*s,u*s+v*c),-1).flatten(-2)
    def forward(self,x,state=None,cache=False):
        b,t,d=x.shape;q,k,v=self.qkv(x).split((64,16*self.kvh,16*self.kvh),-1)
        q=q.reshape(b,t,4,16).transpose(1,2);k=k.reshape(b,t,self.kvh,16).transpose(1,2);v=v.reshape(b,t,self.kvh,16).transpose(1,2)
        off=0 if state is None else state[0].shape[-2];q,k=self.rope(q,off),self.rope(k,off)
        if state is not None:k=torch.cat((state[0],k),-2);v=torch.cat((state[1],v),-2)
        # Repeat is transient compute, not compact persistent state. CPU SDPA still
        # computes all four query heads; no claimed attention-FLOP reduction.
        kk=k if self.kvh==4 else k.repeat_interleave(4//self.kvh,1)
        vv=v if self.kvh==4 else v.repeat_interleave(4//self.kvh,1)
        if not off:z=F.scaled_dot_product_attention(q,kk,vv,is_causal=True)
        elif t==1:z=F.scaled_dot_product_attention(q,kk,vv)
        else:
            mask=torch.arange(k.shape[-2])[None,:]<=torch.arange(off,off+t)[:,None]
            z=F.scaled_dot_product_attention(q,kk,vv,attn_mask=mask)
        ns=(k.clone(),v.clone()) if cache else None
        return self.out(z.transpose(1,2).reshape(b,t,d)),ns

def build(kvh,seed,trained=False):
    torch.manual_seed(seed);m=LM('transformer')
    source=torch.load(ROOT/'anti_latency_v7'/f'attention_{seed}_long.pt',weights_only=True)['model']
    m.load_state_dict(source)
    for b in m.blocks:
        old=b.mix;new=GQA(kvh)
        q,k,v=old.qkv.weight.detach().reshape(3,4,16,64)
        def pool(z):return z.reshape(kvh,4//kvh,16,64).mean(1).reshape(16*kvh,64)
        with torch.no_grad():new.qkv.weight.copy_(torch.cat((q.reshape(64,64),pool(k),pool(v))));new.out.weight.copy_(old.out.weight)
        b.mix=new
    if trained:m.load_state_dict(torch.load(P/'checkpoints'/f'h{kvh}_s{seed}.pt',weights_only=True))
    return m

def sources():
    manifest=json.loads((P/'source_data_manifest.json').read_text());out={};train_seen=set()
    for split in ('train','validation','test'):
        f=ROOT/'frontier_research_v8'/f'{split}.txt';assert sha(f)==manifest['splits'][split]['text_sha256'];seen=set();rows=[]
        for line in f.read_bytes().splitlines(keepends=True):
            k=line.strip()
            if k and (k in seen or split!='train' and k in train_seen):continue
            if k:seen.add(k)
            rows.append(line)
        if split=='train':train_seen=seen
        out[split]=b''.join(rows)
    return out

def prepare():
    ss=sources();train=ss['train'][:2000000];assert digest(train)=='77fc488d9c24735f44714b30f6c8507e61c16369d278810cffb0decb5df5f4af'
    arrays={};meta={};excluded=[];seen=set()
    wiki=[];wm=[]
    for start in range(360448,360448+128*512,512):
        raw=ss['test'][start:start+513];target=raw[-128:]
        if len(raw)!=513 or target in ss['train'] or target in ss['validation'] or target in seen:
            excluded.append({'dataset':'wiki','start':start});continue
        seen.add(target);wiki.append(list(raw));wm.append({'start':start,'target_sha256':digest(target),'window_sha256':digest(raw)})
    fm=json.loads((P/'fineweb_source_manifest.json').read_text());assert sha(P/'fineweb_rows.json')==fm['response_sha256'];fine=[];mm=[]
    for row in json.loads((P/'fineweb_rows.json').read_text())['rows']:
        raw=row['row']['text'].encode()[:513];target=raw[-128:]
        if len(raw)!=513 or target in ss['train'] or target in ss['validation'] or target in seen:
            excluded.append({'dataset':'fineweb','doc':row['row_idx']});continue
        seen.add(target);fine.append(list(raw));mm.append({'doc':row['row_idx'],'target_sha256':digest(target),'window_sha256':digest(raw)})
    arrays={'wiki':np.array(wiki,dtype=np.uint8),'fineweb':np.array(fine,dtype=np.uint8)};np.savez(P/'evaluation.npz',**arrays)
    write('ledger.json',{'train_bytes':len(train),'train_sha256':digest(train),'wiki':wm,'fineweb':mm,'excluded':excluded,'array_hashes':{k:digest(v.tobytes()) for k,v in arrays.items()}})
    write('protocol.json',{'version':38,'purpose':'Measure practical attention-head compression before economic calculation. Not a new invention.','seeds':SEEDS,'kv_heads':HEADS,'query_heads':4,'steps':STEPS,'batch':4,'context':512,'targets_per_run':STEPS*4*512,
        'initial':'Existing context-trained attention long checkpoints, same source per seed. Pool contiguous K/V projection heads by mean for2/1KVheads; unchanged4head control uses identical optimized implementation. Original source800updates at128 plus200retained-optimizer updates at512; previous validation-only LR search. No older weights retrained.',
        'optimizer':'Reset AdamW for EVERY arm,including control; lr.001*(.1+.9*(1+cos(pi*step/319))/2),betas(.9,.999),wd.01,clip1; identical sampled512target windows per seed. Equal targets/steps, not equal parameters/FLOPs/wall.',
        'evaluation':'WikiTexttest128fixed nonoverlapping candidates at byte360448; FineWebEdu first513bytes of each existing rows1500:1600. Exact128target overlap exclusions vs full dedup train/validation and current targets. FineWeb cohort previously evaluated with different models; exploratory reuse, no new-unseen-cohort claim. Last128targets scored at nested128 and512contexts.',
        'primary':'At512context,2compressed arms x2datasets =4 comparisons, seed-mean perwindow losses,5000paired window/doc bootstrap seed38,98.75% intervals(Bonferroni4). Practical tolerance upper bound<=.01bits/byte in BOTH datasets plus measured resource advantage; not frontier noninferiority certificate. Wiki windows not independent articles.',
        'timing':'After training/audits stop: batch1,context480prefill+32observed token updates,33complete256outcome distributions;8FineWebdocs,all3seeds,3randomized repeats/arm,warmup1. Native compact cache4/2/1KVheads; repeated transientK/V counted as limitation. Report p50/p95,paired ratios,actual unique state storage,weights and RoPE constants; not served generation or GPUcost.',
        'source_sha256':{f:sha(P/f) for f in ['run.py','base_models.py']},'input_checkpoints':{str(s):sha(ROOT/'anti_latency_v7'/f'attention_{s}_long.pt') for s in SEEDS},'ledger_sha256':sha(P/'ledger.json')})
    print('prepared',len(wiki),len(fine),'windows',flush=True)

def verify():
    p=json.loads((P/'protocol.json').read_text())
    for f,h in p['source_sha256'].items():assert sha(P/f)==h
    assert sha(P/'ledger.json')==p['ledger_sha256']
    for s,h in p['input_checkpoints'].items():assert sha(ROOT/'anti_latency_v7'/f'attention_{s}_long.pt')==h

def datasets():
    ledger=json.loads((P/'ledger.json').read_text());z=np.load(P/'evaluation.npz');out={}
    for k in ('wiki','fineweb'):assert digest(z[k].tobytes())==ledger['array_hashes'][k];out[k]=torch.from_numpy(z[k].astype(np.int64))
    return out

@torch.inference_mode()
def evaluate(m):
    m.eval();out={}
    for ds,z in datasets().items():
        for context in (128,512):
            losses=[]
            for a in z.split(4):
                x=a[:,-context-1:-1];y=a[:,-128:];logits=m(x)[:,-128:]
                loss=F.cross_entropy(logits.double().flatten(0,1),y.flatten(),reduction='none').reshape(-1,128)/math.log(2);losses.extend(loss.mean(1).tolist())
            out[f'{ds}_{context}']={'bpb':float(np.mean(losses)),'window_bpb':losses}
    return out

def train(seed,heads):
    verify();train=torch.tensor(list(sources()['train'][:2000000]));m=build(heads,seed);before=evaluate(m)
    starts=torch.randint(len(train)-513,(STEPS,4),generator=torch.Generator().manual_seed(38000+seed));opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);history=[];cpu=time.process_time();wall=time.perf_counter();m.train()
    for step,ss in enumerate(starts):
        x=train[ss[:,None]+torch.arange(513)[None,:]];opt.zero_grad(set_to_none=True);opt.param_groups[0]['lr']=.001*(.1+.9*(1+math.cos(math.pi*step/(STEPS-1)))/2)
        loss=F.cross_entropy(m(x[:,:-1]).flatten(0,1),x[:,1:].flatten());assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();history.append(float(loss.detach()))
    cpu=time.process_time()-cpu;wall=time.perf_counter()-wall;(P/'checkpoints').mkdir(exist_ok=True);f=P/'checkpoints'/f'h{heads}_s{seed}.pt';torch.save(m.state_dict(),f)
    r={'seed':seed,'kv_heads':heads,'parameters':count(m),'weight_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),'training_cpu_seconds':cpu,'training_wall_seconds':wall,'target_presentations':STEPS*4*512,'sample_starts_sha256':digest(starts.numpy().tobytes()),'checkpoint_sha256':sha(f),'before':before,'after':evaluate(m),'train_losses':history};write(f'h{heads}_s{seed}.json',r);print(seed,heads,cpu,{k:v['bpb'] for k,v in r['after'].items()},flush=True)

def summary():
    verify();rs=[json.loads((P/f'h{h}_s{s}.json').read_text()) for s in SEEDS for h in HEADS];out={'means':{},'primary':{},'training_cpu_seconds':sum(r['training_cpu_seconds'] for r in rs)}
    for h in HEADS:
        out['means'][str(h)]={k:float(np.mean([r['after'][k]['bpb'] for r in rs if r['kv_heads']==h])) for k in ('wiki_128','wiki_512','fineweb_128','fineweb_512')}
    for ds in ('wiki','fineweb'):
        metric=ds+'_512';base=np.mean([r['after'][metric]['window_bpb'] for r in rs if r['kv_heads']==4],0);n=len(base);draws=np.random.default_rng(38).integers(0,n,(5000,n))
        for h in (2,1):
            candidate=np.mean([r['after'][metric]['window_bpb'] for r in rs if r['kv_heads']==h],0);delta=candidate-base;ci=np.quantile(delta[draws].mean(1),[.00625,.99375]);out['primary'][ds+'_h'+str(h)]={'delta_bpb':float(delta.mean()),'ci98_75':ci.tolist(),'practical_gate':bool(ci[1]<=.01),'windows':n}
    write('summary.json',out);print(json.dumps(out),flush=True)

def state_bytes(states):
    tensors=[v for s in states for v in s];logical=sum(t.numel()*t.element_size() for t in tensors);stores={(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()):t.untyped_storage().nbytes() for t in tensors};return logical,sum(stores.values())

@torch.inference_mode()
def timing():
    verify();windows=datasets()['fineweb'][:8];rng=random.Random(380038);rows=[]
    for seed in SEEDS:
        models={h:build(h,seed,True).eval() for h in HEADS}
        for doc,z in enumerate(windows):
            def go(h):
                m=models[h];start=time.perf_counter();logits,states=m(z[:480][None,:],cache=True);logits[:,-1].softmax(-1);pre=time.perf_counter()-start;start=time.perf_counter()
                for token in z[480:512]:logits,states=m(token.reshape(1,1),states,cache=True);logits[:,-1].softmax(-1)
                dec=time.perf_counter()-start;logical,allocated=state_bytes(states)
                return dict(prefill_s=pre,decode_s=dec,total_s=pre+dec,logical_state_bytes=logical,allocated_state_bytes=allocated,weight_bytes=sum(p.numel()*p.element_size() for p in m.parameters()),rope_bytes=sum(sum(v.numel()*v.element_size() for v in b.mix.cs) for b in m.blocks))
            for h in HEADS:go(h)
            for rep in range(3):
                order=list(HEADS);rng.shuffle(order)
                for pos,h in enumerate(order):r=go(h);r.update(seed=seed,doc=doc,rep=rep,heads=h,order=pos);rows.append(r)
    write('timing.json',{'rows':rows,'batch':1,'threads':1,'prefill_tokens':480,'teacher_forced_updates':32,'output_distributions':33,'hardware':platform.processor()})
    out={}
    for h in HEADS:
        rr=[r for r in rows if r['heads']==h];out[str(h)]={k:float(np.median([r[k] for r in rr])) for k in ('prefill_s','decode_s','total_s','allocated_state_bytes','weight_bytes','rope_bytes')};out[str(h)]['total_p95_s']=float(np.quantile([r['total_s'] for r in rr],.95))
        if h!=4:
            ratios=[]
            for r in rr:
                b=next(v for v in rows if v['heads']==4 and all(v[k]==r[k] for k in ('seed','doc','rep')));ratios.append(b['total_s']/r['total_s'])
            out[str(h)]['paired_median_speed_ratio']=float(np.median(ratios));out[str(h)]['faster_pairs']=int(np.sum(np.array(ratios)>1));out[str(h)]['pairs']=len(ratios)
    write('timing_summary.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(600,610))
    mode=sys.argv[1]
    if mode=='train':train(int(sys.argv[2]),int(sys.argv[3]))
    else:{'prepare':prepare,'summary':summary,'timing':timing}[mode]()
