"""Executed, isolated selected-file loading and local/full update accounting.

Main study holds weights in RAM. This proves the selected-file contract and
measures a separate real CE update, not uncached disk or production throughput.
"""
from experiment import *
from safetensors.torch import save_file,load as load_bytes
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
import io
Q=P/'io_probe'
def restore(kind,seed,endpoint):
    torch.manual_seed(seed);m=Treatment(kind);state=torch.load(P/'checkpoints'/f'{kind}_{seed}_{endpoint}.pt',weights_only=True)
    assert set(state)=={n for n,p in m.named_parameters() if p.requires_grad};r=m.load_state_dict(state,strict=False);assert not r.unexpected_keys;return m

def io_stat():return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
def delta(a,b):return {k:b[k]-a[k] for k in a}
def read_file(p):
    a=io_stat();t=time.perf_counter()
    with p.open('rb') as f:raw=f.read()
    s=load_bytes(raw);return s,{'file':p.name,'logical_bytes':len(raw),'wall_seconds':time.perf_counter()-t,'proc_io_delta':delta(a,io_stat())}
def fixed():
    x=torch.arange(BS*T).reshape(BS,T)+10;y=x+1;z=torch.sin(torch.arange(BS*T*D).float()).reshape(BS,T,D)
    selected=(torch.arange(T)[None,:].expand(BS,-1)%2==0);inputs=x.clone();inputs[selected]=MASK
    return x,y,z,selected,inputs

def local_output(m,family,group):
    x,y,z,selected,inputs=fixed()
    if family=='normal':return m(x).flatten(0,1),y.reshape(-1)
    if family=='masked':return m.masked(inputs,selected,.5,z,group),x[selected]
    return m.local.training_logits(x,y,z,torch.full((BS,1,1),.2),group).flatten(0,1),y.reshape(-1)
@torch.inference_mode()
def fingerprint(m,family,group):
    m.eval();return hashlib.sha256(local_output(m,family,group)[0].contiguous().numpy().tobytes()).hexdigest()
def export():
    Q.mkdir(exist_ok=True);meta={'sources':{},'scope':'One-time sharding reads full study checkpoints. Shared vocabulary files and selected bodies are separate. Local probe group chosen explicitly; no optimizer states included.'};t=time.process_time();w=time.perf_counter()
    for family,kind,group in [('normal','normal',0),('masked','masked_hybrid_parent',0),('direct','direct_hybrid_all',1)]:
        m=restore(kind,101,'equal_data');meta['sources'][family]={'kind':kind,'group':group,'reference_logits_sha256':fingerprint(m,family,group)}
        if family=='normal':
            shared={n:p.detach().contiguous() for n,p in m.named_parameters() if not p.requires_grad};save_file(shared,Q/'shared.safetensors')
            state={n:p.detach().contiguous() for n,p in m.named_parameters() if p.requires_grad}
        else:
            state={}
            for i in range(2):
                for n,p in m.base.gpt_neox.layers[group*2+i].named_parameters():state[f'base.gpt_neox.layers.{i}.{n}']=p.detach().contiguous()
            names=['conditions','norms'] if family=='masked' else ['norms','context_adapters','noise_adapters','times']
            parent=m if family=='masked' else m.local;prefix='' if family=='masked' else 'local.'
            for a in names:
                for n,p in getattr(parent,a)[group].named_parameters():state[f'{prefix}{a}.0.{n}']=p.detach().contiguous()
        save_file(state,Q/f'{family}_body.safetensors');del m,state
    meta['files']={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(Q.glob('*.safetensors'))};meta['export_cpu_seconds']=time.process_time()-t;meta['export_wall_seconds']=time.perf_counter()-w;write('io_manifest.json',meta)
def build(family):
    config=GPTNeoXConfig.from_json_file(str(P/'original'/'config.json'));config.num_hidden_layers=6 if family=='normal' else 2;config._attn_implementation='eager'
    shared,a=read_file(Q/'shared.safetensors');body,b=read_file(Q/f'{family}_body.safetensors')
    m=Treatment.__new__(Treatment);nn.Module.__init__(m);m.kind=family;m.base=freeze_maps(GPTNeoXForCausalLM(config).float())
    r=m.load_state_dict(shared,strict=False);assert not r.unexpected_keys
    if family=='masked':m.conditions=nn.ModuleList([nn.Linear(1,2*D)]);m.norms=nn.ModuleList([copy.deepcopy(m.base.gpt_neox.final_layer_norm)])
    elif family=='direct':m.local=DirectBlocks(m.base,full_depth=True)
    if family!='normal':m.base.gpt_neox.final_layer_norm.requires_grad_(False)
    assert set(body)=={n for n,p in m.named_parameters() if p.requires_grad},(set(body),{n for n,p in m.named_parameters() if p.requires_grad})
    r=m.load_state_dict(body,strict=False);assert not r.unexpected_keys;return m,[a,b]
def probe(family):
    torch.manual_seed(49);m,reads=build(family);fp=fingerprint(m,family,0);assert fp==json.loads((P/'io_manifest.json').read_text())['sources'][family]['reference_logits_sha256']
    params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),weight_decay=.01);m.train()
    def step():
        opt.zero_grad(set_to_none=True);logits,labels=local_output(m,family,0);F.cross_entropy(logits,labels).backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
    for _ in range(2):step()
    persistent={p.untyped_storage().data_ptr() for p in list(m.parameters())+list(m.buffers())};seen={}
    def pack(t):
        s=t.untyped_storage()
        if s.data_ptr() not in persistent:seen[s.data_ptr()]=s.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):step()
    t=time.process_time();w=time.perf_counter()
    for _ in range(8):step()
    cpu=(time.process_time()-t)/8;wall=(time.perf_counter()-w)/8
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:step()
    warm=[]
    for _ in range(3):s,r=read_file(Q/f'{family}_body.safetensors');warm.append(r);del s
    buf=io.BytesIO();torch.save({n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad},buf);raw=buf.getvalue();a=io_stat();w=time.perf_counter()
    with (Q/f'probe_updated_{family}.pt').open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    write(f'accounting_{family}.json',{'family':family,'loaded_files':[r['file'] for r in reads],'subset_logits_exact':True,'initial_reads':reads,'warm_body_reads':warm,'weight_write':{'logical_bytes':len(raw),'wall_seconds_excluding_serialization':time.perf_counter()-w,'proc_io_delta':delta(a,io_stat())},'saved_nonparameter_storage_bytes':sum(seen.values()),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'warmed_step_cpu_seconds':cpu,'warmed_step_wall_seconds':wall,'matrix_operator_flops':sum(e.flops for e in prof.key_averages()),'trainable_parameters':count(m),'scope':'Separate process; actual selected file reads and real CE forward/backward/clip/AdamW. Synthetic fixed batch2 context32, selected32targets for masked versus64for others. Teacher/production pipeline absent. Physical cache state uncontrolled; logical transfer bytes are not disk bandwidth. RSS includes construction/loading. Updated weight transfer excludes optimizer state.'})
    print('PASS',family,cpu,flush=True)
if __name__=='__main__':export() if sys.argv[1]=='export' else probe(sys.argv[1])
