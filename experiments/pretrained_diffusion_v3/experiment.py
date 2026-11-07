import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from models import *
import numpy as np
import json,hashlib,time,sys,resource
P=Path(__file__).parent;BS=2;T=32;STEPS=648;MAX_STEPS=2592;SEEDS=[101,202,303]
KINDS=['normal','masked_hybrid_parent','masked_hybrid_all','masked_local_all','direct_hybrid_all','direct_local_all']
torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,obj):(P/n).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def data():
    out={}
    for s,m in json.loads((P/'token_manifest.json').read_text()).items():
        p=P/'data'/f'{s}.npy';assert sha(p)==m['sha256'];out[s]=torch.from_numpy(np.load(p));assert not (out[s]==MASK).any()
    return out
def starts(seed,n):return torch.randint(0,n-T-1,(MAX_STEPS,BS),generator=torch.Generator().manual_seed(seed+100))
def batch(d,ss):
    a=d[ss[:,None]+torch.arange(T+1)[None,:]];return a[:,:-1],a[:,1:]
def freeze():
    ds=data();windows={}
    v2=json.loads((P.parent/'pretrained_diffusion_v2'/'window_manifest.json').read_text())
    for split,count,oldn in [('validation',32,32),('test',64,64)]:
        old=np.linspace(0,len(ds[split])-33,oldn).astype(int).tolist()+v2['splits'][split]['starts'];chosen=[]
        for i in np.random.default_rng(77130 if split=='validation' else 815019).permutation(np.arange(0,len(ds[split])-33,64)):
            if all(abs(int(i)-j)>33 for j in old):chosen.append(int(i))
            if len(chosen)==count:break
        assert len(chosen)==count;windows[split]={'starts':chosen,'excluded_v1_v2_starts':old}
    write('window_manifest.json',windows)
    write('protocol.json',{'version':3,'model':'Pinned pretrained Pythia-14M; frozen vocabulary input/output maps for all methods','methods':KINDS,'seeds':SEEDS,'steps':STEPS,'max_steps':MAX_STEPS,'batch':BS,'context':T,'input_token_exposures':STEPS*BS*T,'inference':'Every method uses the unchanged native six-layer causal next-token inference graph. Local branches share core layers with the native graph; no diffusion rollout or teacher at inference.','matching':'Equal data/updates plus first update reaching same-seed ordinary training CPU. CPU includes forward, loss, backward, clipping, and optimizer, excludes preparation, loading, evaluation and no-update warmup. Not simultaneous equal predicted-target count or equal FLOPs.','optimizer':'AdamW lr1e-4,warmup48,betas(.9,.95),wd.01,clip1. All methods start from same original independently. Local losses above35 are skipped and counted; nonfinite loss aborts.','masked':'Source-matched local mechanism: every ninth update for hybrid, first native group only for parent analogue or all3 groups cycling; sigma uniform per descending probability band over[.05,.95], token selection probability sigma with >=1 selected; unused reserved ID50303; VP corruption sqrt(1-sigma^2)*embedding+sigma*standard_normal; zero affine conditioning x*(1+.1*tanh(scale))+shift; selected2layers with bidirectional attention; selected-token reconstruction CE. Uses Pythia LayerNorm rather than source RMS.','direct':'V2 clean-past-only/noisy-query denoising auxiliary, native shared layers, same frozen vocabulary codes and noise bands as v2. Every ninth update for hybrid, all updates for local. Causal native inference remains the common evaluation. Not a paper-faithful diffusion generator.','anchors':'Common ordinary causal CE anchors for all hybrids to isolate the local intervention. Deliberately does NOT reproduce source SAT-fixed, learned SAT-variable, NAT anchors, ternary weights, MoE routing, shared trainable table, writeback/checkpoint system or rotating update shard.','restricted_group_limit':'Live schedule uses oldest20/80 groups. Six-layer proxy has3 two-layer groups, so parent analogue uses first1/3 groups rather than exactly1/4. It retains the source parent-count noise-band recalibration; label as mechanism analogue, not exact AGILLM runtime reproduction.','evaluation':'New32validation/64test disjoint contexts excluding all scored v1/v2 contexts; final16 targets each. Standard causal exact next-token NLL at context<=32. Original pretraining contamination unknown.','decision':'Report all methods/endpoints. Practical screening gate: mean test gap<=.05nats versus ordinary paired control, with conditional window bootstrap uncertainty, and measured useful compute/I/O advantage. This small pilot cannot certify production noninferiority. No test selection for extra training or deployment.','safety':'Scratch CPU only; no production changes, no paid resources, no old checkpoint removal.','source_sha256':{f:sha(P/f) for f in ['direct.py','models.py','experiment.py','checks.py','prepare.py']},'data_manifest_sha256':sha(P/'token_manifest.json'),'window_manifest_sha256':sha(P/'window_manifest.json')})
@torch.inference_mode()
def evaluate(m,d,split):
    m.eval();losses=[];cpu=time.process_time();wall=time.perf_counter();ss=torch.tensor(json.loads((P/'window_manifest.json').read_text())[split]['starts'])
    for starts_ in ss.split(BS):
        x,y=batch(d,starts_);nll=-m(x).log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)
        losses.extend(nll[:,-16:].mean(1).tolist())
    m.train();return {'nll_nats':float(np.mean(losses)),'window_nll_nats':losses,'scored_tokens':len(ss)*16,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall}
def save(m,kind,seed,endpoint,step,cpu,wall,d,stats,budget):
    (P/'checkpoints').mkdir(exist_ok=True);f=P/'checkpoints'/f'{kind}_{seed}_{endpoint}.pt';torch.save({n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad},f)
    r={'kind':kind,'seed':seed,'endpoint':endpoint,'steps':step,'input_token_exposures':step*BS*T,'cpu_seconds':cpu,'wall_seconds':wall,'cpu_target':budget,'cpu_ratio':cpu/budget if budget else None,'trainable_parameters':count(m),'stats':stats.copy(),'checkpoint_sha256':sha(f),'validation':evaluate(m,d['validation'],'validation'),'test':evaluate(m,d['test'],'test')};write(f'{kind}_{seed}_{endpoint}.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ['validation','test','stats']}|{'test_nll':r['test']['nll_nats'],'stats':stats}),flush=True)
def train(kind,seed):
    p=json.loads((P/'protocol.json').read_text())
    for f,h in p['source_sha256'].items():assert sha(P/f)==h,(f,'changed after freeze')
    d=data();torch.manual_seed(seed);m=Treatment('normal' if kind=='untouched' else kind);m.train()
    if kind=='untouched':write('untouched.json',{'validation':evaluate(m,d['validation'],'validation'),'test':evaluate(m,d['test'],'test')});return
    params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),weight_decay=.01)
    windows=starts(seed,len(d['train']));budget=None if kind=='normal' else json.loads((P/f'normal_{seed}_equal_data.json').read_text())['cpu_seconds']
    # No-update warmup of the actual family; no optimizer/RNG/model changes.
    x,y=batch(d['train'],windows[0]);o=batch_objective(kind,9 if 'hybrid' in kind else 1,seed,x,y)
    for _ in range(2):
        opt.zero_grad(set_to_none=True);F.cross_entropy(objective_logits(m,x,y,o),o['labels']).backward()
    opt.zero_grad(set_to_none=True);cpu=wall=0.;matched=False;stats={'ar_updates':0,'local_updates':0,'local_skipped_above35':0,'predicted_targets':0,'group_updates':[0,0,0]}
    for step in range(1,MAX_STEPS+1):
        x,y=batch(d['train'],windows[step-1]);o=batch_objective(kind,step,seed,x,y)
        t=time.process_time();w=time.perf_counter();opt.zero_grad(set_to_none=True);opt.param_groups[0]['lr']=1e-4*min(1.,step/48)
        logits=objective_logits(m,x,y,o);loss=F.cross_entropy(logits,o['labels']);assert torch.isfinite(loss)
        if o['local'] and float(loss.detach())>35:stats['local_skipped_above35']+=1
        else:
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();stats['predicted_targets']+=o['selected_count'];stats['local_updates' if o['local'] else 'ar_updates']+=1
            if o['local']:stats['group_updates'][o['group']]+=1
        cpu+=time.process_time()-t;wall+=time.perf_counter()-w
        if budget and not matched and cpu>=budget:save(m,kind,seed,'equal_cpu',step,cpu,wall,d,stats,budget);matched=True
        if step==STEPS:
            save(m,kind,seed,'equal_data',step,cpu,wall,d,stats,budget)
            if kind=='normal':return
        if step>=STEPS and matched:return
        if step%216==0:print(kind,seed,step,'cpu',round(cpu,2),'loss',float(loss.detach()),flush=True)
    if not matched:save(m,kind,seed,'capped_cpu',step,cpu,wall,d,stats,budget)
if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(360,365))
    freeze() if '--freeze' in sys.argv else train(sys.argv[1],int(sys.argv[2]))
