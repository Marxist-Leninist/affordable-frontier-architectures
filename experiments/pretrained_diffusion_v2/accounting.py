"""Isolated actual weight-file transfers and selected-block training probe.

Run export, then probe normal and probe block in separate processes. This is a
scratch offload microbenchmark, not the in-memory study's per-step disk traffic,
not uncached disk bandwidth, and not a 512B throughput prediction.
"""
import run
from run import *
from safetensors.torch import save_file,load as load_bytes
from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
import io,resource
Q=P/'io_probe'
def io_counters():
    result={}
    for line in Path('/proc/self/io').read_text().splitlines():
        key,val=line.split(':');result[key]=int(val.strip())
    return result
def diff(a,b):return {k:b[k]-a[k] for k in a}
def read_weights(path):
    a=io_counters();t=time.perf_counter()
    with open(path,'rb') as f:raw=f.read()
    state=load_bytes(raw);dt=time.perf_counter()-t;b=io_counters()
    return state,dict(file=path.name,logical_bytes=len(raw),wall_read_deserialize_seconds=dt,proc_io_delta=diff(a,b))
def restore(kind,seed,endpoint):
    torch.manual_seed(seed);original=load_original();m=DirectBlocks(original) if kind.startswith('direct') else freeze_maps(original)
    state=torch.load(P/'checkpoints'/f'{kind}_{seed}_{endpoint}.pt',weights_only=True)
    assert set(state)=={n for n,p in m.named_parameters() if p.requires_grad};result=m.load_state_dict(state,strict=False);assert not result.unexpected_keys
    return m
@torch.inference_mode()
def fingerprint(m,mode,block_index=0):
    m.eval();x=torch.arange(BS*T).reshape(BS,T)+10
    z=torch.sin(torch.arange(BS*T*D).float()).reshape(BS,T,D);s=torch.full((BS,1,1),.2)
    logits=m(x,use_cache=False).logits if mode=='normal' else m.denoise(x,z,s,block_index)
    return hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest()
def export():
    begin_cpu=time.process_time();begin_wall=time.perf_counter()
    Q.mkdir(exist_ok=True);m=restore('normal_ce',11,'equal_data')
    shared={n:p.detach().contiguous() for n,p in m.named_parameters() if not p.requires_grad}
    save_file(shared,Q/'shared.safetensors')
    save_file({n:p.detach().contiguous() for n,p in m.named_parameters() if p.requires_grad},Q/'normal_body.safetensors');refs={'normal':fingerprint(m,'normal')};del m,shared
    m=restore('direct_kd',11,'equal_data')
    refs['block']=fingerprint(m,'block',1)
    for b in range(3):
        state={}
        for group in ['groups','norms','context_adapters','noise_adapters','times']:
            for name,param in getattr(m,group)[b].named_parameters():state[f'{group}.0.{name}']=param.detach().contiguous()
        save_file(state,Q/f'block_{b}.safetensors')
    write('io_manifest.json',dict(files={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(Q.glob('*.safetensors'))},reference_logits_sha256=refs,export_cpu_seconds=time.process_time()-begin_cpu,export_wall_seconds=time.perf_counter()-begin_wall,scope='One-time sharding loads full original/model checkpoints. FP32 scratch weights; shared input/output lexical matrices separate from ordinary full body or individual converted block. No optimizer state in these files. Not published.'))
def build(mode):
    c=GPTNeoXConfig.from_json_file(str(P/'original'/'config.json'));c._attn_implementation='eager';c.num_hidden_layers=6 if mode=='normal' else 2
    shared,sr=read_weights(Q/'shared.safetensors');body,br=read_weights(Q/('normal_body.safetensors' if mode=='normal' else 'block_1.safetensors'))
    base=GPTNeoXForCausalLM(c).float().eval();base.load_state_dict(shared,strict=False)
    m=freeze_maps(base) if mode=='normal' else DirectBlocks(base,full_depth=True)
    assert set(body)=={n for n,p in m.named_parameters() if p.requires_grad}
    loaded=m.load_state_dict(body,strict=False);assert not loaded.unexpected_keys
    return m,[sr,br]
def probe(mode):
    torch.manual_seed(183);m,reads=build(mode)
    actual_fingerprint=fingerprint(m,mode);assert actual_fingerprint==json.loads((P/'io_manifest.json').read_text())['reference_logits_sha256'][mode]
    m.train();params=[p for p in m.parameters() if p.requires_grad]
    dataset=data()['train'];x,y=getbatch(dataset,train_windows(11)[0]);noise=torch.randn(BS,T,D);s=sample_sigma(1,BS,torch.Generator().manual_seed(123))
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),weight_decay=.01)
    def forward_loss():
        logits=m(x,use_cache=False).logits if mode=='normal' else m.training_logits(x,y,noise,s,0)
        return F.cross_entropy(logits.flatten(0,1),y.flatten())
    def step():
        opt.zero_grad(set_to_none=True);loss=forward_loss();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
    for _ in range(2):step()
    persistent={p.untyped_storage().data_ptr() for p in list(m.parameters())+list(m.buffers())};seen={}
    def pack(t):
        st=t.untyped_storage()
        if st.data_ptr() not in persistent:seen[st.data_ptr()]=st.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):step()
    cpu=time.process_time();wall=time.perf_counter()
    for _ in range(8):step()
    cpu=(time.process_time()-cpu)/8;wall=(time.perf_counter()-wall)/8
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True,record_shapes=True) as prof:step()
    flops=sum(e.flops for e in prof.key_averages())
    bodyfile=Q/('normal_body.safetensors' if mode=='normal' else 'block_1.safetensors');warm=[]
    for _ in range(3):state,r=read_weights(bodyfile);warm.append(r);del state
    # Persist only this probe's updated trainable weight copy. Original study and
    # all input files remain unchanged. fsync is scoped to this one new file.
    buf=io.BytesIO();torch.save({n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad},buf);raw=buf.getvalue()
    before=io_counters();t=time.perf_counter()
    with open(Q/f'probe_updated_{mode}.pt','wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    wr=dict(logical_bytes=len(raw),wall_serialize_excluded_fsync_seconds=time.perf_counter()-t,proc_io_delta=diff(before,io_counters()))
    out=dict(mode=mode,loaded_weight_files=[r['file'] for r in reads],loaded_subset_reproduces_full_source_logits=True,reference_logits_sha256=actual_fingerprint,cold_component_logical_read_bytes=sum(r['logical_bytes'] for r in reads),initial_reads=reads,warm_shared_maps_body_reads=warm,updated_trainable_weight_write=wr,saved_nonparameter_storage_bytes=sum(seen.values()),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,trainable_parameters=count(m),warmed_training_cpu_seconds=cpu,warmed_training_wall_seconds=wall,profiled_matrix_operator_flops_per_step=flops,scope='Separate process; full CE forward/backward/clip/AdamW step. Normal six native layers versus one loaded two-layer conversion block. Includes interpreter/library/model/optimizer in RSS. Tensor-save storage is not peak RAM. FLOPs are profiler-supported matrix ops, not complete hardware instructions. Teacher/cache construction and KL excluded from probe timing; full recipe costs recorded in main runs. Filesystem cache is not flushed; physical read_bytes may be zero. Shared vocabulary maps dominate this small model.')
    write(f'accounting_{mode}.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':export() if sys.argv[1]=='export' else probe(sys.argv[1])
