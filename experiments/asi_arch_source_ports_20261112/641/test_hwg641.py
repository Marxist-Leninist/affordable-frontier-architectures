"""Bounded one-thread tensor/gradient checks; zero optimizer or latent updates."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import hashlib,importlib,importlib.util,json,pathlib,sys,copy,ast
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1)
ROOT=pathlib.Path(__file__).resolve().parent
REUSE=ROOT
SIM=ROOT/"agillm_mini.py"
sys.path.insert(0,str(REUSE));sys.path.insert(0,str(ROOT))
PORT_NAME=sys.argv[1] if len(sys.argv)>1 else "hwg641"
assert PORT_NAME in ("hwg641","hwg641_dirac")
port=importlib.import_module(PORT_NAME)
DIRAC=PORT_NAME.endswith("_dirac")
import fla_cpu
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
assert sha(SIM)=="dc5c3c28ddbf18538ff9c7fd7de0679fd6f3fc19215284937d127d54732ee7d2"
assert sha(ROOT/"upstream641.py")==port.SOURCE_PROGRAM_SHA256
PROVENANCE=json.loads((ROOT/"provenance.json").read_text())
assert PROVENANCE["catalog_sha256"]=="aa77e8289971b3922102f000fddf44a6a9a31ca1f3b59c834722ac130f63759c"
assert sha(REUSE/"fla_cpu.py")=="855f8365a668504386e2938b62698fecf5b242b1285e61d33d193c101e57fe17"
expected_reference="\n".join(line for line in (ROOT/"upstream641.py").read_text().splitlines() if not line.strip().startswith("@torch.compile"))
expected_reference=expected_reference.replace("inv = inv.to(torch.bfloat16)","inv = inv.to(v.dtype)  # explicit dtype-preserving reference adaptation")
assert (ROOT/"reference641_dtype_preserving.py").read_text()==expected_reference
fla_cpu.install()
refmod=load("ref641",ROOT/"reference641_dtype_preserving.py")
sim=load("sim641",SIM)
checks=[];maxima={};details={}
def check(name,**kw):checks.append(dict(name=name,passed=True,**kw))
def close(name,x,y,atol=2e-5,rtol=2e-4):
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
    error=float((x-y).abs().max()) if x.numel() else 0
    maxima[name]=max(maxima.get(name,0),error)
# Verify pinned NATIVE FLA constructor, without importing GPU/Triton kernels.
native_path=ROOT/"native_fla_022_convolution.py"
assert sha(native_path)=="d0b107a222cde18acc15ba3f84045da5c4c661b0d700c3c27ee0553e21eaec28"
assert sha(ROOT/"ASI_ARCH_REQUIREMENTS.txt")=="5e0814c8c93864394a6d0a1a71791f1a0b0acd641a1eee81f3d11e958ce07edd"
assert "flash-linear-attention==0.2.2" in (ROOT/"ASI_ARCH_REQUIREMENTS.txt").read_text()
node=next(n for n in ast.parse(native_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="ShortConvolution")
assert not any(isinstance(n,ast.FunctionDef) and n.name=="reset_parameters" for n in node.body)
init=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
node.body=[init]
module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
ns=dict(nn=nn,torch=torch,Optional=Optional)
exec(compile(module,"<pinned native FLA constructor>","exec"),ns)
with torch.random.fork_rng():
    torch.manual_seed(100);native=ns["ShortConvolution"](12,4,bias=False,use_fast_conv1d=False)
    torch.manual_seed(100);expected=nn.Conv1d(12,12,4,groups=12,bias=False,padding=3)
    close("pinned_native_fla_initialization",native.weight,expected.weight,0,0)
check("pinned_native_fla022_constructor_matches_nn_conv1d")
class TinyBase(nn.Module):
    def __init__(self,c,index):
        super().__init__();self.c=c;self.H=c["H"]
        self.qkv=nn.Linear(c["d"],3*c["d"],bias=False)
    def mix(self,h,causal):return h
cfg={"d":12,"H":3,"L":4}
PB,_=port.patch(TinyBase,None,cfg)
pairs=[("short_q","q_conv1d.weight"),("short_k","k_conv1d.weight"),("short_v","v_conv1d.weight"),
       ("b.weight","b_proj.weight"),("fs","local_fir_short.filters"),("fl","local_fir_long.filters"),
       ("fusion_weight","fusion_weight"),("fusion_bias","fusion_bias")]
def make_pair(dtype,random_gate=True):
    p=PB(cfg,0).to(dtype)
    r=refmod.DeltaNet(hidden_size=cfg["d"],num_heads=cfg["H"]).to(dtype)
    # Test MIXER boundary: source's head RMS/o are explicitly excluded.
    r.o_norm=nn.Identity();r.o_proj=nn.Identity()
    with torch.no_grad():
        if random_gate:
            p.fusion_weight.normal_(0,.08);p.fusion_bias.normal_(0,.2)
        for n,w in zip(("q_proj","k_proj","v_proj"),p.qkv.weight.chunk(3)):
            getattr(r,n).weight.copy_(w)
        rp=dict(r.named_parameters());pp=dict(p.named_parameters())
        for pn,rn in pairs:rp[rn].copy_(pp[pn])
    return p,r
torch.manual_seed(641)
for dtype in (torch.float32,torch.float64):
    for T in (1,7,31,32,33,65):
        p,r=make_pair(dtype)
        x=torch.randn(2,T,12,dtype=dtype,requires_grad=True);xr=x.detach().clone().requires_grad_(True)
        y=p.mix(x,True);yr=r(xr)[0]
        close("mixer_output_"+str(dtype),y,yr)
        g=torch.randn_like(y);(y*g).sum().backward();(yr*g).sum().backward()
        close("input_gradient_"+str(dtype),x.grad,xr.grad,atol=6e-5,rtol=6e-4)
        pp=dict(p.named_parameters());rp=dict(r.named_parameters())
        close("qkv_gradient_"+str(dtype),p.qkv.weight.grad,torch.cat([getattr(r,n).weight.grad for n in ("q_proj","k_proj","v_proj")]),atol=1e-4,rtol=7e-4)
        for pn,rn in pairs:
            close("parameter_gradient_"+str(dtype),pp[pn].grad,rp[rn].grad,atol=1e-4,rtol=7e-4)
        check("copied_weight_output_gradient",dtype=str(dtype),length=T,batch=2,random_nonzero_gate=True)
# Initialization facts, including all four active softmax branches.
p,r=make_pair(torch.float32,False)
assert torch.count_nonzero(p.fusion_weight)==0
close("warm_bias",p.fusion_bias,torch.tensor([[0.,0.,0.,2.]]).expand(3,-1),0,0)
prob=p.fusion_bias.softmax(-1)
assert bool((prob>0).all())
assert p.b.bias is None
for w in (p.short_q,p.short_k,p.short_v):
    if DIRAC:
        expect=torch.zeros_like(w);expect[:,:,-1]=1;close("practical_dirac_init",w,expect,0,0)
    else:
        assert bool((w.abs()<=.5).all()) and bool((w[:,:,-1]!=1).any())
# Test initializer deterministically against nn.Conv1d and source FIR classes.
torch.manual_seed(9);source_fir=refmod.DepthwiseFIRConv1d(num_heads=3,head_dim=4,kernel_size=31)
torch.manual_seed(9);expected=torch.zeros(3,4,31);expected[:,:,-1]=1;expected+=torch.randn_like(expected)*.02
close("fir_init_contract",source_fir.filters,expected,0,0)
assert p.fs.shape==(3,4,3) and p.fl.shape==(3,4,31)
with torch.random.fork_rng():
    torch.manual_seed(46);actual=PB(cfg,0)
    torch.manual_seed(46);dummy=TinyBase(cfg,0)
    if DIRAC:
        shorts=[torch.zeros(12,1,4) for _ in range(3)]
        for w in shorts:w[:,:,-1]=1
    else:
        shorts=[nn.Conv1d(12,12,4,groups=12,bias=False).weight for _ in range(3)]
    beta_init=nn.Linear(12,3,bias=False).weight
    sf=refmod.DepthwiseFIRConv1d(num_heads=3,head_dim=4,kernel_size=3)
    lf=refmod.DepthwiseFIRConv1d(num_heads=3,head_dim=4,kernel_size=31)
    for a,b in zip((actual.short_q,actual.short_k,actual.short_v),shorts):close("short_initialisation",a,b,0,0)
    close("beta_initialisation",actual.b.weight,beta_init,0,0)
    close("short_fir_initialisation",actual.fs,sf.filters,0,0)
    close("long_fir_initialisation",actual.fl,lf.filters,0,0)
check("explicit_initialisation_contract",variant=port.META["cohort"],warm_start_weights=prob[0].tolist())
# Independent scalar-time recurrence and chunk size invariance.
def serial(q,k,v,beta):
    q,k=port.l2norm(q),port.l2norm(k)
    S=q.new_zeros(q.shape[0],q.shape[1],q.shape[-1],v.shape[-1]);outs=[]
    for t in range(q.shape[2]):
        residual=v[:,:,t]-torch.einsum("bhd,bhde->bhe",k[:,:,t],S)
        S=S+beta[:,:,t,None,None]*k[:,:,t,:,None]*residual[:,:,:,None].transpose(-1,-2)
        outs.append(torch.einsum("bhd,bhde->bhe",q[:,:,t],S))
    return torch.stack(outs,2)
q,k,v=[torch.randn(2,3,65,4,dtype=torch.float64) for _ in range(3)]
beta=torch.randn(2,3,65,dtype=torch.float64).sigmoid()
oracle=serial(q,k,v,beta)
for C in (1,7,16,32,64):
    close("independent_serial_delta",port.delta_chunked(q,k,v,beta,C),oracle,2e-12,2e-12)
    check("chunk_boundary_ragged_padding",chunk=C,length=65)
# Low precision conversion contract explicitly checked against fp32.
for dtype in (torch.bfloat16,torch.float16):
    xs=[t.to(dtype) for t in (q,k,v,beta)]
    out=port.delta_chunked(*xs);fp=port.delta_chunked(*(x.float() for x in xs)).to(dtype)
    close("low_precision_promoted_"+str(dtype),out,fp,0,0)
    check("promoted_kernel_dtype_contract",dtype=str(dtype))
# Mixer prefix, right-padded ragged batch, causality=False still causal, batch isolation.
p,_=make_pair(torch.float32);x=torch.randn(2,65,12)
with torch.no_grad():
    y=p.mix(x,True);z=x.clone();z[:,33:]=torch.randn_like(z[:,33:])
    close("mixer_causal_prefix",y[:,:33],p.mix(z,True)[:,:33])
    close("mixer_noncausal_flag",y,p.mix(x,False),0,0)
    close("mixer_batch_isolation",y[:1],p.mix(x[:1],True))
    for T in (1,31,32,33,64):
        close("ragged_prefix",y[:,:T],p.mix(x[:,:T],True))
    state={n:t.detach().clone() for n,t in p.state_dict().items()}
    p.eval();e=p.mix(x,True);p.train();tr=p.mix(x,True)
    close("train_eval_equal",e,tr,0,0)
    assert all(torch.equal(state[n],t) for n,t in p.state_dict().items())
check("mixer_causality_batch_ragged_eval_purity")
# Five full-host objective forward/backward cases, no training updates.
c=dict(sim.SIM,V=64,d=16,H=2,L=4,ff=32,ctx=33,batch=2,mask=63,unk=62,expert_counts=[2]*4)
BC,MC=port.patch(sim.Block,sim.Model,c);model=MC(c,BC)
assert [b.attn_layer for b in model.blocks]==[False,False,False,True]
for mode in ("ar","sat_fixed","sat_variable","nat","local"):
    model.zero_grad(set_to_none=True)
    for m in model.modules():
        if isinstance(m,sim.BinLinear):m._w=None
    if mode=="local":
        ids=torch.randint(0,62,(2,33))
        logits,aux=model.forward_local(ids,0,.1,torch.randn(2,33,16))
        target=ids.reshape(-1);selected=logits.reshape(-1,64)
    else:
        window=torch.randint(0,62,(2,34))
        ids,target,sel,causal=sim.prepare(window,mode,np.random.default_rng(641),63)
        logits,aux=model.logits(ids,causal)
        selected=logits.reshape(-1,64)
        if sel is not None:selected=selected[sel.reshape(-1)]
        target=target.reshape(-1)
    loss=F.cross_entropy(selected,target)+c["router_balance"]*aux
    assert torch.isfinite(loss)
    assert model.take_extra_loss()==0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for m in model.modules():
        if isinstance(m,sim.BinLinear) and m._w is not None and m._w.grad is not None:
            assert torch.isfinite(m._w.grad).all()
    check("finite_masked_objective_backward",mode=mode,loss_for_correctness_only=float(loss.detach()),optimizer_updates=0)
# Every binary projection retains sole int8 latents and row scales; no FP duplicates.
for b in model.blocks:
    for m in (b.qkv,b.o):
        assert isinstance(m,sim.BinLinear) and m.w is None and m.P.dtype==torch.int8
        assert set(dict(m.named_parameters()))=={"scale"}
        assert set(dict(m.named_buffers()))=={"P","R","C","t"}
    assert not any(n.startswith(("q_proj","k_proj","v_proj","o_proj","o_norm")) for n,_ in b.named_modules())
hits={"qkv":0,"no":0,"o":0};handles=[]
for name in hits:
    def hook(mod,inp,out,name=name):hits[name]+=1
    handles.append(getattr(model.blocks[0],name).register_forward_hook(hook))
with torch.no_grad():model.blocks[0](torch.randn(2,7,16),True)
for h in handles:h.remove()
assert hits=={"qkv":1,"no":1,"o":1}
att=sim.Block(c,3);att.load_state_dict(model.blocks[3].state_dict())
with torch.no_grad():
    x=torch.randn(2,7,16)
    for causal in (False,True):close("unchanged_attention_fallback",att.mix(x,causal),model.blocks[3].mix(x,causal),0,0)
check("binary_projection_single_ownership_no_duplicate_rms",forward_calls=hits)
# Exact tensor accounting, including persistent optimizer buffers and bytes.
def account(m):
    ps=list(m.named_parameters());bs=list(m.named_buffers())
    return dict(parameter_numel=sum(p.numel() for _,p in ps),
                parameter_bytes=sum(p.numel()*p.element_size() for _,p in ps),
                buffer_numel=sum(p.numel() for _,p in bs),
                buffer_bytes=sum(p.numel()*p.element_size() for _,p in bs),
                parameters={n:dict(shape=list(p.shape),dtype=str(p.dtype),numel=p.numel()) for n,p in ps},
                buffers={n:dict(shape=list(p.shape),dtype=str(p.dtype),numel=p.numel()) for n,p in bs})
base=sim.Model(c,sim.Block);ac,baseac=account(model),account(base)
assert ac["parameter_numel"]-baseac["parameter_numel"]==3*port.extra_parameter_count(16,2)
assert ac["buffer_numel"]==baseac["buffer_numel"]
assert ac["buffer_bytes"]==baseac["buffer_bytes"]
details["tiny_accounting"]=ac
details["tiny_baseline_totals"]={k:v for k,v in baseac.items() if k not in ("parameters","buffers")}
full=dict(sim.CFG_600,L=560,expert_counts=[2+int((i+1)*36//80>i*36//80) for i in range(560)])
FB,FM=port.patch(sim.Block,sim.Model,full)
basec=sim.census(sim.Model,sim.Block,full);fullc=sim.census(FM,FB,full)
assert fullc["fp_params"]-basec["fp_params"]==420*port.extra_parameter_count(8192,64)
assert fullc["latent_bytes"]==basec["latent_bytes"]==886910746624
details["full_560_layer_accounting"]=dict(baseline=basec,adapted=fullc,mixer_layers=420,
    gate_per_mixer=4*64*8192+12*8192+4*64,added_per_mixer=port.extra_parameter_count(8192,64),
    added_total=420*port.extra_parameter_count(8192,64),
    caveat="kv_bytes_per_token is inherited host attention accounting, not supported cache memory")
def storage_totals(M,B,c):
    with torch.device("meta"):m=M(c,B)
    return dict(parameter_numel=sum(p.numel() for p in m.parameters()),
        parameter_bytes=sum(p.numel()*p.element_size() for p in m.parameters()),
        buffer_numel=sum(p.numel() for p in m.buffers()),
        buffer_bytes=sum(p.numel()*p.element_size() for p in m.buffers()),
        buffers_by_dtype={str(dt):dict(numel=sum(p.numel() for p in m.buffers() if p.dtype==dt),
            bytes=sum(p.numel()*p.element_size() for p in m.buffers() if p.dtype==dt))
            for dt in (torch.int8,torch.float32,torch.int64)})
bst,pst=storage_totals(sim.Model,sim.Block,full),storage_totals(FM,FB,full)
assert pst["buffer_numel"]==bst["buffer_numel"] and pst["buffer_bytes"]==bst["buffer_bytes"]
assert pst["parameter_numel"]==fullc["fp_params"]
details["full_560_layer_storage"]=dict(baseline=bst,adapted=pst)
check("exact_parameter_buffer_accounting_and_full_meta_census")
# Whole-model causal prefix, isolation and evaluation state purity.
for m in model.modules():
    if isinstance(m,sim.BinLinear):m._w=None
model.eval()
before={n:t.detach().clone() for n,t in model.state_dict().items()}
with torch.no_grad():
    ids=torch.randint(0,62,(2,33));y,_=model.logits(ids,True)
    changed=ids.clone();changed[:,17:]=torch.randint(0,62,changed[:,17:].shape)
    z,_=model.logits(changed,True)
    close("full_model_prefix",y[:,:17],z[:,:17])
    one,_=model.logits(ids[:1],True);close("full_model_batch_isolation",y[:1],one)
    y2,_=model.logits(ids,True);close("repeat_eval",y,y2,0,0)
assert all(torch.equal(before[n],t) for n,t in model.state_dict().items())
check("full_model_causality_isolation_eval_purity")
# Optional original literal BF16 diagnostic (no equivalence claim).
literal=fla_cpu.load_program(641,(ROOT/"upstream641.py").read_text())
r=literal.DeltaNet(hidden_size=12,num_heads=3).to(torch.bfloat16)
r.o_norm=nn.Identity();r.o_proj=nn.Identity()
with torch.no_grad():
    inp=torch.randn(1,33,12,dtype=torch.bfloat16);original=r(inp)[0]
assert torch.isfinite(original).all()
check("literal_bf16_source_finite_only")
report=dict(status="isolated full adapter prepared; zero optimizer updates; no GPU jobs; not a quality result",
    environment=dict(python=sys.version.split()[0],torch=torch.__version__,device="CPU",threads=1),
    source=dict(index=641,name="delta_net_hwg",title="HeadWiseGateNet",cohort=port.META["cohort"],catalog_sha256=PROVENANCE["catalog_sha256"]),
    hashes={p.name:sha(p) for p in [ROOT/"upstream641.py",ROOT/"reference641_dtype_preserving.py",ROOT/(PORT_NAME+".py"),ROOT/"test_hwg641.py",REUSE/"fla_cpu.py",SIM,ROOT/"native_fla_022_convolution.py"]},
    checks=checks,checks_passed=len(checks),max_abs_errors=maxima,details=details,
    limits=["No official fused FLA parity","No cached decoding or packed/reset masks",
            "Ragged means right-padded/prefix full sequences only","No trained quality/throughput/GPU claim",
            "No LR/LRK variants prepared","Host kv census field is inherited estimate, not cache support"])
(ROOT/("preparation_results_dirac.json" if DIRAC else "preparation_results.json")).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="details"},indent=2))
