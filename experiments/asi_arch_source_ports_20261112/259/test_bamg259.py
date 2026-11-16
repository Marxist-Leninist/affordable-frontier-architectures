"""Bounded CPU-only fidelity/functional preparation, not a quality benchmark."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"
import copy, hashlib, json, math, pathlib, statistics, sys, types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import bamg259 as port
import fla_cpu
import agillm_mini as sim
torch.set_num_threads(1);torch.set_num_interop_threads(1)
ROOT=pathlib.Path(__file__).resolve().parent
records=json.loads(pathlib.Path(sys.argv[1]).read_text())
rec=next(r for r in records if r["index"]==259)
assert hashlib.sha256(rec["program"].encode()).hexdigest()==port.SOURCE_PROGRAM_SHA256
assert hashlib.sha256((ROOT/"agillm_mini.py").read_bytes()).hexdigest()=="dc5c3c28ddbf18538ff9c7fd7de0679fd6f3fc19215284937d127d54732ee7d2"
fla_cpu.install()
source="\n".join(s for s in rec["program"].splitlines() if not s.strip().startswith("@torch.compile"))
assert source.count("attn = attn.to(torch.bfloat16)")==1
(ROOT/"upstream259.py").write_text(rec["program"])
transformed_source=source.replace("attn = attn.to(torch.bfloat16)","attn = attn.to(v.dtype)")
(ROOT/"reference259_dtype_preserving.py").write_text(transformed_source)

mod=types.ModuleType("source259_dtype_preserving")
exec(compile(source.replace("attn = attn.to(torch.bfloat16)","attn = attn.to(v.dtype)"),"source259_dtype_preserving","exec"),mod.__dict__)
literal=types.ModuleType("source259_literal")
exec(compile(source,"source259_literal","exec"),literal.__dict__)
class Base(nn.Module):
    def __init__(self,c,index):
        super().__init__();self.H=c["H"];self.qkv=nn.Linear(c["d"],3*c["d"],bias=False)
    def mix(self,h,causal):
        return h+7 # sentinel verifies untouched attention dispatch

def setup(dtype,reference=mod):
    cfg={"d":16,"H":2,"L":4}
    Cls,_=port.patch(Base,None,cfg)
    p=Cls(cfg,0).to(dtype)
    r=reference.DeltaNet(hidden_size=16,num_heads=2).to(dtype)
    r.o_norm=nn.Identity();r.o_proj=nn.Identity()
    with torch.no_grad():
        for i,key in enumerate(("q","k","v")):
            getattr(r,key+"_proj").weight.copy_(p.qkv.weight[i*16:(i+1)*16])
            getattr(r,key+"_conv1d").weight.copy_(getattr(p,"short_"+key))
        r.b_proj.weight.copy_(p.b.weight)
        r.mix_proj.load_state_dict(p.mix_proj.state_dict())
        r.mix_bias.copy_(p.mix_bias)
    return p,r,Cls

checks=[]; maxima={"output_f64":0.0,"output_f32":0.0,"input_grad_f64":0.0,"input_grad_f32":0.0,"parameter_grad_f64":0.0,"parameter_grad_f32":0.0}
torch.manual_seed(259)
for dtype in (torch.float64,torch.float32):
    suffix="f64" if dtype==torch.float64 else "f32"
    atol=2e-10 if dtype==torch.float64 else 2e-5
    gatol=5e-9 if dtype==torch.float64 else 2e-4
    for length in (1,31,32,33,63,64,65,127,257):
        p,r,Cls=setup(dtype)
        x=torch.randn(2,length,16,dtype=dtype,requires_grad=True)
        xr=x.detach().clone().requires_grad_()
        y=p.mix(x,True);yr=r(xr)[0]
        err=float((y-yr).detach().abs().max()); maxima["output_"+suffix]=max(maxima["output_"+suffix],err)
        torch.testing.assert_close(y,yr,atol=atol,rtol=atol)
        target=torch.randn_like(y)
        (y*target).sum().backward();(yr*target).sum().backward()
        ge=float((x.grad-xr.grad).abs().max());maxima["input_grad_"+suffix]=max(maxima["input_grad_"+suffix],ge)
        torch.testing.assert_close(x.grad,xr.grad,atol=gatol,rtol=gatol)
        pairs=[(p.qkv.weight.grad,torch.cat([r.q_proj.weight.grad,r.k_proj.weight.grad,r.v_proj.weight.grad])),
               (p.b.weight.grad,r.b_proj.weight.grad),(p.mix_proj.weight.grad,r.mix_proj.weight.grad),
               (p.mix_proj.bias.grad,r.mix_proj.bias.grad),(p.mix_bias.grad,r.mix_bias.grad)]
        pairs += [(getattr(p,"short_"+k).grad,getattr(r,k+"_conv1d").weight.grad) for k in ("q","k","v")]
        for a,b in pairs:
            maxima["parameter_grad_"+suffix]=max(maxima["parameter_grad_"+suffix],float((a-b).abs().max()))
            torch.testing.assert_close(a,b,atol=gatol,rtol=gatol)
        checks.append({"kind":"dtype_adapted_upstream_output_and_gradients","dtype":suffix,"length":length,"pass":True})
# Source's literal BF16 behavior: measure separately, never claim exact FP32 identity.
p,r,_=setup(torch.float32,literal)
x=torch.randn(2,65,16)
with torch.no_grad(),torch.autocast("cpu",dtype=torch.bfloat16):
    yr=r(x)[0]
with torch.no_grad():yp=p.mix(x,True)
bf16_difference=float((yp-yr.float()).abs().max())
# Structural invariants including block-boundary cases.
p,r,Cls=setup(torch.float64)
x=torch.randn(2,97,16,dtype=torch.float64)
with torch.no_grad():
    y=p.mix(x,True)
    for cut in (1,31,32,33,64,65,96):
        changed=x.clone();changed[:,cut:]=torch.randn_like(changed[:,cut:])*3
        torch.testing.assert_close(y[:,:cut],p.mix(changed,True)[:,:cut],atol=1e-12,rtol=1e-12)
        checks.append({"kind":"causality","cut":cut,"pass":True})
    torch.testing.assert_close(y[:1],p.mix(x[:1],True),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(y,p.mix(x,False),atol=0,rtol=0)
    attention=Cls({"d":16,"H":2,"L":4},3).double()
    torch.testing.assert_close(attention.mix(x,False),x+7,atol=0,rtol=0)
    gate=torch.sigmoid(p.mix_proj(torch.zeros(1,1,16,dtype=torch.float64))+p.mix_bias)
    torch.testing.assert_close(gate,torch.full_like(gate,1/(1+math.exp(2))),atol=1e-15,rtol=1e-15)
checks += [{"kind":k,"pass":True} for k in ("batch_isolation","mixer_causal_in_noncausal_mode","attention_dispatch","double_bias_init")]
# Same frozen simulator API, binary matrices, backward and finite optimizer microsteps only.
cfg=dict(sim.SIM,V=64,d=16,H=2,L=4,ff=32,ctx=33,batch=2,mask=63,unk=62,expert_counts=[2]*4)
Block,Model=port.patch(sim.Block,sim.Model,cfg)
model=Model(cfg,Block);opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
micro=[]
for mode in ("ar","sat_fixed","sat_variable","nat","local"):
    opt.zero_grad(set_to_none=True)
    if mode=="local":
        ids=torch.randint(0,62,(2,33))
        logits,aux=model.forward_local(ids,0,.1,torch.randn(2,33,16))
        targets=ids.reshape(-1); selected_logits=logits.reshape(-1,64)
    else:
        window=torch.randint(0,62,(2,34))
        ids,targets,sel,causal=sim.prepare(window,mode,np.random.default_rng(259),63)
        logits,aux=model.logits(ids,causal=causal)
        selected_logits=logits.reshape(-1,64)
        if sel is not None: selected_logits=selected_logits[sel.reshape(-1)]
        targets=targets.reshape(-1)
    loss=F.cross_entropy(selected_logits,targets)+.01*aux
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for m in model.modules():
        if isinstance(m,sim.BinLinear):assert m.latent_step(.01)[2]
    opt.step()
    micro.append({"mode":mode,"finite":True,"loss_for_functional_check_only":float(loss.detach())})
with torch.no_grad():
    ids=torch.randint(0,62,(2,33)); logits,_=model.logits(ids,True)
    changed=ids.clone();changed[:,17:]=torch.randint(0,62,changed[:,17:].shape)
    after,_=model.logits(changed,True)
    torch.testing.assert_close(logits[:,:17],after[:,:17],atol=1e-5,rtol=1e-5)
    alone,_=model.logits(ids[:1],True)
    torch.testing.assert_close(logits[:1],alone,atol=1e-5,rtol=1e-5)
checks += [{"kind":k,"pass":True} for k in ("full_simulator_causality","full_simulator_batch_isolation")]
# Meta-only full geometry accounting.
real=dict(sim.CFG_600,L=560,expert_counts=[2+int((i+1)*36//80>i*36//80) for i in range(560)])
BaseC=sim.census(sim.Model,sim.Block,real)
BC,MC=port.patch(sim.Block,sim.Model,real);PortC=sim.census(MC,BC,real)
extra=420*(2*8192*64+12*8192+2*64)
assert PortC["fp_params"]-BaseC["fp_params"]==extra
assert PortC["latent_bytes"]==BaseC["latent_bytes"]==886910746624
# Seed-level uncertainty, not a pooled window bootstrap.
metrics=json.loads(pathlib.Path(sys.argv[2]).read_text())
v=metrics["confirmation_extended"]["259"]["versus"]["458"]["per_seed"]
ci={}
for dataset in ("wikitext","fineweb_edu"):
    xs=[v[str(s)][dataset+"_delta"] for s in (2,3,4,5)]
    mean=statistics.mean(xs);sd=statistics.stdev(xs);half=3.182446305*sd/2
    ci[dataset]={"paired_candidate_minus_dlgm":xs,"mean":mean,"sample_sd":sd,"paired_t95_df3":[mean-half,mean+half]}
report={"source":{"index":259,"name":rec["name"],"name_new":rec.get("name_new"),"metadata_parameters":rec.get("parameters"),"program_sha256":port.SOURCE_PROGRAM_SHA256,
 "dtype_preserving_test_source_sha256":hashlib.sha256(transformed_source.encode()).hexdigest(),
 "source_without_compile_decorator_sha256":hashlib.sha256(source.encode()).hexdigest(),
 "gallery_json_sha256":hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(),
 "simulator_sha256":hashlib.sha256((ROOT/"agillm_mini.py").read_bytes()).hexdigest(),
 "plugin_sha256":hashlib.sha256((ROOT/"bamg259.py").read_bytes()).hexdigest()},
 "environment":{"torch":torch.__version__,"python":sys.version.split()[0],"device":"CPU","threads":1},

 "status":"isolated preparation only; no training benchmark, no queue or GPU run",
 "comparison_protocol":"three-arm v2 must be frozen separately; leave existing index v1 unchanged",
 "checks":checks,"maxima":maxima,"checks_passed":len(checks),
 "literal_bf16_vs_fp32_max_abs_output_difference":bf16_difference,
 "bf16_scope":"unmodified upstream core under CPU BF16 autocast versus FP32 port, with same copied weights and CPU FLA stand-ins; not Triton validation",
 "zero_hidden_memory_gate":float(gate[0,0,0]),"microsteps":micro,
 "census560":{"base":BaseC,"candidate":PortC,"geometry":{"D":8192,"H":64,"head_dim":128,"L":560,"attention_layers":140,"candidate_layers":420,"experts":sum(real["expert_counts"])},
 "base_total_params":BaseC["latent_bytes"]+BaseC["table_bytes"]+BaseC["fp_params"],
 "candidate_total_params":PortC["latent_bytes"]+PortC["table_bytes"]+PortC["fp_params"],
 "added_fp_params":extra,"added_gate_and_beta_fp_params":420*(2*8192*64+2*64),"added_short_conv_fp_params":420*12*8192,
 "added_fixed_buffers_bytes":0,
 "added_fp32_weights_bytes":extra*4,"added_fp32_gradients_bytes":extra*4,
 "added_fp32_adam_m_bytes":extra*4,"added_fp32_adam_v_bytes":extra*4,"added_fp32_weight_grad_adam_m_v_bytes":extra*16,
 "fullscale_allocated":False,"cache_implemented":False,
 "prospective_cache_fp32_batch1_T2048_bytes":140*2*8192*4*2048+420*4*(64*128*128+9*8192),
 "cache_note":"prospective streaming implementation only: delta state plus three (k-1) short-conv histories; current hook inherits all-attention cache accounting and is not a working cached decoder"},
 "selection_evidence":ci}
(ROOT/"preparation_results.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
