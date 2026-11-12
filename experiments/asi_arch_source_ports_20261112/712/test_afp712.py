"""Bounded one-thread tensor/gradient checks; zero optimizer or latent updates."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
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
PORT_NAME="afp712_dirac"
port=importlib.import_module(PORT_NAME)
DIRAC=PORT_NAME.endswith("_dirac")
import fla_cpu
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
assert sha(SIM)=="dc5c3c28ddbf18538ff9c7fd7de0679fd6f3fc19215284937d127d54732ee7d2"
assert sha(ROOT/"upstream712.py")==port.SOURCE_PROGRAM_SHA256
PROVENANCE=json.loads((ROOT/"provenance.json").read_text())
assert PROVENANCE["catalog_sha256"]=="aa77e8289971b3922102f000fddf44a6a9a31ca1f3b59c834722ac130f63759c"
assert sha(REUSE/"fla_cpu.py")=="855f8365a668504386e2938b62698fecf5b242b1285e61d33d193c101e57fe17"
expected_reference="\n".join(line for line in (ROOT/"upstream712.py").read_text().splitlines() if not line.strip().startswith("@torch.compile"))
expected_reference=expected_reference.replace("inv = inv.to(torch.bfloat16)","inv = inv.to(v.dtype)  # explicit dtype-preserving reference adaptation")
assert (ROOT/"reference712_dtype_preserving.py").read_text()==expected_reference
fla_cpu.install()
refmod=load("ref712",ROOT/"reference712_dtype_preserving.py")
sim=load("sim712",SIM)
checks=[];maxima={};details={}
def check(name,**kw):checks.append(dict(name=name,passed=True,**kw))
def close(name,x,y,atol=2e-5,rtol=2e-4):
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
    error=float((x-y).detach().abs().max()) if x.numel() else 0
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
       ("b.weight","b_proj.weight"),("fs","fir_short.filters"),("fl","fir_long.filters"),
       ("gate_weight","gate_weight"),("gate_bias","gate_bias"),
       ("floor_param","floor_param"),("log_temp","log_temp")]
def make_pair(dtype,random_gate=True):
    p=PB(cfg,0).to(dtype)
    r=refmod.DeltaNet(hidden_size=cfg["d"],num_heads=cfg["H"]).to(dtype)
    # Test MIXER boundary: source's head RMS/o are explicitly excluded.
    r.o_norm=nn.Identity();r.o_proj=nn.Identity()
    with torch.no_grad():
        if random_gate:
            p.gate_weight.normal_(0,.08);p.gate_bias.normal_(0,.2)
            p.floor_param.normal_(0,.6);p.log_temp.uniform_(-.8,.8)
        for n,w in zip(("q_proj","k_proj","v_proj"),p.qkv.weight.chunk(3)):
            getattr(r,n).weight.copy_(w)
        rp=dict(r.named_parameters());pp=dict(p.named_parameters())
        for pn,rn in pairs:rp[rn].copy_(pp[pn])
    return p,r
torch.manual_seed(712)
for dtype in (torch.float32,torch.float64):
    for T in (1,7,31,32,33,63,64,65,129):
        p,r=make_pair(dtype)
        x=torch.randn(2,T,12,dtype=dtype,requires_grad=True);xr=x.detach().clone().requires_grad_(True)
        y=p.mix(x,True);yr=r(xr)[0]
        close("mixer_output_"+str(dtype),y,yr,*( (2e-10,2e-10) if dtype==torch.float64 else (2e-5,2e-4)))
        _,diag=p.mix_with_diagnostics(x,True)
        close("diagnostic_source_gate_"+str(dtype),diag,r.last_fusion_weights.detach())
        assert not diag.requires_grad and diag.device.type=="cpu"
        g=torch.randn_like(y);(y*g).sum().backward();(yr*g).sum().backward()
        close("input_gradient_"+str(dtype),x.grad,xr.grad,atol=2e-10 if dtype==torch.float64 else 6e-5,rtol=2e-10 if dtype==torch.float64 else 6e-4)
        pp=dict(p.named_parameters());rp=dict(r.named_parameters())
        close("qkv_gradient_"+str(dtype),p.qkv.weight.grad,torch.cat([getattr(r,n).weight.grad for n in ("q_proj","k_proj","v_proj")]),atol=2e-10 if dtype==torch.float64 else 1e-4,rtol=2e-10 if dtype==torch.float64 else 7e-4)
        for pn,rn in pairs:
            close("parameter_gradient_"+str(dtype),pp[pn].grad,rp[rn].grad,atol=2e-10 if dtype==torch.float64 else 1e-4,rtol=2e-10 if dtype==torch.float64 else 7e-4)
        check("copied_weight_output_gradient",dtype=str(dtype),length=T,batch=2,random_nonzero_gate=True)
# Initialization: source gate, floor, temperature and FIR; practical short Dirac.
p,r=make_pair(torch.float32,False)
assert torch.count_nonzero(p.gate_weight)>0
close("warm_bias",p.gate_bias,torch.tensor([[-.5,-.5,.5,1.5]]).expand(3,-1),0,0)
close("floor_init",p.floor_param,torch.zeros(3,4),0,0)
close("temp_init",p.log_temp,torch.zeros(4),0,0)
prob=.025+.9*p.gate_bias.softmax(-1)
assert bool((prob>=.025).all())
assert p.b.bias is None
for w in (p.short_q,p.short_k,p.short_v):
    expect=torch.zeros_like(w);expect[:,:,-1]=1;close("practical_dirac_init",w,expect,0,0)
torch.manual_seed(9);source_fir=refmod.DepthwiseFIRConv1d(num_heads=3,head_dim=4,kernel_size=64)
torch.manual_seed(9);expected=torch.randn(3,4,64)*.02
close("source_fir_init_contract",source_fir.filters,expected,0,0)
assert p.fs.shape==(3,4,5) and p.fl.shape==(3,4,64)
with torch.random.fork_rng():
    torch.manual_seed(46);actual=PB(cfg,0)
    torch.manual_seed(46);dummy=TinyBase(cfg,0)
    beta_init=nn.Linear(12,3,bias=False).weight
    sf=refmod.DepthwiseFIRConv1d(num_heads=3,head_dim=4,kernel_size=5)
    lf=refmod.DepthwiseFIRConv1d(num_heads=3,head_dim=4,kernel_size=64)
    gate_init=torch.empty(3,12+12*4,4)
    nn.init.kaiming_uniform_(gate_init,a=5**.5)
    close("beta_initialisation",actual.b.weight,beta_init,0,0)
    close("short_fir_initialisation",actual.fs,sf.filters,0,0)
    close("long_fir_initialisation",actual.fl,lf.filters,0,0)
    close("gate_kaiming_initialisation",actual.gate_weight,gate_init,0,0)
    assert actual.gate_weight.abs().max()<=1/(4*(12+12*4))**.5
fresh_source=refmod.DeltaNet(hidden_size=12,num_heads=3)
close("native_bias_init",fresh_source.gate_bias,p.gate_bias,0,0)
close("native_floor_init",fresh_source.floor_param,p.floor_param,0,0)
close("native_temp_init",fresh_source.log_temp,p.log_temp,0,0)
assert fresh_source.base_floor==port.BASE_FLOOR==.05
assert fresh_source.gate_weight.shape==p.gate_weight.shape
assert bool((fresh_source.q_conv1d.weight[:,:,-1]!=1).any())
check("explicit_initialisation_contract",variant=port.META["cohort"],
      zero_feature_initial_weights=prob[0].tolist(),source_short_initialization="random Conv1d",practical_short="Dirac")
# Source gate statistics average heads at each token and retain own-head norm.
t=torch.randn(2,7,3,4,dtype=torch.float64)
source=refmod.DeltaNet(hidden_size=12,num_heads=3).double()
close("source_branch_statistics",port.branch_stats(t),torch.cat(source.stat_f(t),-1),0,0)
close("stat_no_time_aggregation",port.branch_stats(t)[:,:3],port.branch_stats(t[:,:3]),0,0)
# Temperature clamp/floor extremes and their zero saturation gradients.
p,r=make_pair(torch.float64)
with torch.no_grad():
    p.log_temp.copy_(torch.tensor([-8.,0.,.7,8.],dtype=torch.float64));r.log_temp.copy_(p.log_temp)
    p.floor_param.copy_(torch.tensor([[-20.,-1.,2.,20.]],dtype=torch.float64).expand(3,-1));r.floor_param.copy_(p.floor_param)
x=torch.randn(2,33,12,dtype=torch.float64,requires_grad=True);xr=x.detach().clone().requires_grad_(True)
y,diag=p.mix_with_diagnostics(x);yr=r(xr)[0]
close("clamped_gate_output",y,yr,2e-10,2e-10)
close("clamped_gate_diagnostics",diag,r.last_fusion_weights.detach(),2e-10,2e-10)
floor=port.BASE_FLOOR*p.floor_param.sigmoid()
assert bool((diag>=floor.detach()[None,None,:,:]).all())
close("gate_simplex",diag.sum(-1),torch.ones_like(diag[...,0]),2e-15,2e-15)
g=torch.randn_like(y);(g*y).sum().backward();(g*yr).sum().backward()
for pn,rn in pairs:
    close("clamped_gate_gradients",dict(p.named_parameters())[pn].grad,dict(r.named_parameters())[rn].grad,2e-10,2e-10)
assert p.log_temp.grad[0]==0 and p.log_temp.grad[3]==0
check("source_gate_statistics_floor_temperature_clamp")
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
        ids,target,sel,causal=sim.prepare(window,mode,np.random.default_rng(712),63)
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
    gate_per_mixer=port.gate_parameter_count(8192,64),added_per_mixer=port.extra_parameter_count(8192,64),
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
# Actual reduced host geometry is counted separately from the tiny tensor model.
reduced=dict(sim.SIM,expert_counts=[2+int((i+1)*36//80>i*36//80) for i in range(sim.SIM["L"])])
RB,RM=port.patch(sim.Block,sim.Model,reduced)
rb,rp=storage_totals(sim.Model,sim.Block,reduced),storage_totals(RM,RB,reduced)
assert rp["parameter_numel"]-rb["parameter_numel"]==9*port.extra_parameter_count(256,4)
assert rp["buffer_numel"]==rb["buffer_numel"] and rp["buffer_bytes"]==rb["buffer_bytes"]
details["reduced_12_layer_storage"]=dict(baseline=rb,adapted=rp,mixer_layers=9,
    gate_per_mixer=port.gate_parameter_count(256,4),added_per_mixer=port.extra_parameter_count(256,4),
    geometry={k:reduced[k] for k in ("V","d","H","L","ff","expert_counts")})
check("exact_reduced_geometry_meta_census")
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
# State contract: diagnostics are explicit/detached; no graph-bearing hidden state.
p,_=make_pair(torch.float32);p.eval()
state={n:t.detach().clone() for n,t in p.state_dict().items()}
attrs=set(p.__dict__);x=torch.randn(2,33,12,requires_grad=True)
y,diag=p.mix_with_diagnostics(x)
assert set(p.__dict__)==attrs and not hasattr(p,"last_fusion_weights")
assert not diag.requires_grad and diag.grad_fn is None and diag.device.type=="cpu"
assert all(torch.equal(state[n],t) for n,t in p.state_dict().items())
clone=PB(cfg,0);clone.load_state_dict(state,strict=True);clone.eval()
close("checkpoint_mixer_roundtrip",p.mix(x,True),clone.mix(x,True),0,0)
required={"short_q","short_k","short_v","b.weight","fs","fl","gate_weight","gate_bias","floor_param","log_temp"}
assert required<=set(state)
assert not list(p.named_buffers())
# Runtime RNG is unchanged in eval; diagnostics do not need checkpointing.
rng=torch.random.get_rng_state().clone()
with torch.no_grad():p.mix_with_diagnostics(x)
assert torch.equal(rng,torch.random.get_rng_state())
for bad in ({"attention_mask":torch.ones(2,33)},{"use_cache":True},{"cu_seqlens":torch.tensor([0,33,66])}):
    try:p.mix(x,True,**bad)
    except TypeError:pass
    else:raise AssertionError("unsupported API silently accepted")
try:p.mix(x[:,:0],True)
except ValueError:pass
else:raise AssertionError("empty sequence silently accepted")
check("diagnostic_eval_rng_checkpoint_state_contract",schedules=0,built_in_auxiliary_losses=0,
      source_mutable_attribute="last_fusion_weights",adaptation="detached explicit diagnostic return",
      external_gate_regularization="unsupported; needs separate trainer integration")
# Audit exact source forward state mutation and complete host checkpoint tensors.
tree=ast.parse((ROOT/"upstream712.py").read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="DeltaNet")
fw=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="forward")
mutations={n.attr for n in ast.walk(fw) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)
           and isinstance(n.value,ast.Name) and n.value.id=="self"}
assert mutations=={"last_fusion_weights"}
assert not any(isinstance(n,ast.FunctionDef) and n.name in ("post_step","take_extra_loss") for n in cls.body)
restored=MC(c,BC);restored.load_state_dict(before,strict=True);restored.eval()
with torch.no_grad():restored_y,_=restored.logits(ids,True)
close("full_model_checkpoint_roundtrip",y2,restored_y,0,0)
assert all(torch.equal(before[n],t) for n,t in restored.state_dict().items())
check("source_state_audit_and_full_host_checkpoint_coverage",source_mutations=sorted(mutations))
# Optional original literal BF16 diagnostic (no equivalence claim).
literal=fla_cpu.load_program(712,(ROOT/"upstream712.py").read_text())
r=literal.DeltaNet(hidden_size=12,num_heads=3).to(torch.bfloat16)
r.o_norm=nn.Identity();r.o_proj=nn.Identity()
with torch.no_grad():
    inp=torch.randn(1,33,12,dtype=torch.bfloat16);original=r(inp)[0]
assert torch.isfinite(original).all()
check("literal_bf16_source_finite_only")
report=dict(status="isolated full adapter prepared; zero optimizer updates; no GPU jobs; not a quality result",
    environment=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,einops=importlib.import_module("einops").__version__,device="CPU",threads=1),
    source=dict(index=712,name="delta_net_afp",title="AdaptiveGateNet-AFP",cohort=port.META["cohort"],catalog_sha256=PROVENANCE["catalog_sha256"]),
    hashes={p.name:sha(p) for p in [ROOT/"upstream712.py",ROOT/"reference712_dtype_preserving.py",ROOT/(PORT_NAME+".py"),ROOT/"test_afp712.py",REUSE/"fla_cpu.py",SIM,ROOT/"native_fla_022_convolution.py"]},
    checks=checks,checks_passed=len(checks),max_abs_errors=maxima,details=details,
    limits=["No official fused FLA parity","No cached decoding or packed/reset masks",
            "Ragged means right-padded/prefix full sequences only","No trained quality/throughput/GPU claim",
            "No LR/LRK variants prepared","Host kv census field is inherited estimate, not cache support",
            "No trainer optimizer/RNG/data-cursor resumability certification",
            "No graph-bearing last_fusion_weights; external gate regularization unsupported"])
(ROOT/("preparation_results_dirac.json" if DIRAC else "preparation_results.json")).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="details"},indent=2))
