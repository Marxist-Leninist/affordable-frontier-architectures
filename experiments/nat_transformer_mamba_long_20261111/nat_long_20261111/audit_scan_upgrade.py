"""Optional benchmark static/mock audit; no numerical model/tensor execution."""
import ast,fcntl,hashlib,importlib.util,json,sys,tempfile,types
from pathlib import Path
P=Path(__file__).resolve().parent;path=P/'benchmark_scan_upgrade.py';source=path.read_text();tree=ast.parse(source)
spec=importlib.util.spec_from_file_location('scan_bench_mock',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks={}
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)/'nat';root.mkdir();(root.parent/'dory_loop_compare_20261111').mkdir();m.ROOT=root
 oldargv=sys.argv;sys.argv=['benchmark_scan_upgrade.py','--go-after-postprocess']
 (root/'postprocess_status.json').write_text('{"status":"waiting_for_training"}')
 try:m.main();checks['before_completed_gate_blocks']=False
 except AssertionError:checks['before_completed_gate_blocks']=True
 (root/'postprocess_status.json').write_text('{"status":"completed"}')
 lock=(root/'.evaluation_timing.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:m.main();checks['nat_phase_lock_blocks']=False
 except BlockingIOError:checks['nat_phase_lock_blocks']=True
 lock.close();lock=(root.parent/'dory_loop_compare_20261111'/'.execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:m.main();checks['dory_phase_lock_blocks']=False
 except BlockingIOError:checks['dory_phase_lock_blocks']=True
 lock.close();(root/'scan_upgrade_whole_model.json').write_text('{"status":"previous"}')
 try:m.main();checks['duplicate_result_blocks']=False
 except AssertionError as e:checks['duplicate_result_blocks']='duplicate' in str(e)
 checks['duplicate_receipt_unchanged']=json.loads((root/'scan_upgrade_whole_model.json').read_text())['status']=='previous'
 m.atomic_result(root/'mock.json',{'status':'completed'});checks['atomic_receipt_complete']=json.loads((root/'mock.json').read_text())=={'status':'completed'} and not (root/'mock.tmp').exists()
 sys.argv=oldargv
# Execute only nested step with dummy model/tensors/optimizer, not torch.
step=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step')
clones=[];updates=[]
class Dummy:
 def flatten(self,*a):return self
 def detach(self):return self
 def clone(self):clones.append(1);return self
 def backward(self):pass
class Param(Dummy):
 def __init__(self):self.grad=Dummy()
param=Param();model=types.SimpleNamespace(named_parameters=lambda:[('weight',param)],parameters=lambda:[param])
class Model:
 def __call__(self,*a):return Dummy()
 def named_parameters(self):return [('weight',param)]
 def parameters(self):return [param]
opt=types.SimpleNamespace(zero_grad=lambda **kw:None,step=lambda:updates.append(1))
ns={'models':{'v':Model()},'opts':{'v':opt},'p':Dummy(),'y':Dummy(),'time':types.SimpleNamespace(process_time=lambda:0,perf_counter=lambda:0),'F':types.SimpleNamespace(cross_entropy=lambda *a:Dummy()),'torch':types.SimpleNamespace(isfinite=lambda x:True,nn=types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=lambda *a,**kw:None)))}
exec(compile(ast.Module(body=[step],type_ignores=[]),'<step-only>','exec'),ns)
ns['step']('v');checks['timed_step_no_inspection_clones']=len(clones)==0 and len(updates)==1
ns['step']('v',True);checks['inspection_step_copies_logits_gradients_parameters']=len(clones)==3 and len(updates)==2
checks['checked_step_not_appended_to_trials']=source.index("fast=step('native_linear',True)")<source.index('rows=[]') and "row=step(str(variant))" in source
checks['fresh_optimizer_pair_and_weight_assertions']='assert all(not opt.state for opt in opts.values())' in source and "'Initial weights differ'" in source
checks['checkpoint_input_hashes_verified_before_after']=source.count('checkpoint_hash')>=3 and "'Input batch changed'" in source and "'Checkpoint receipt mismatch'" in source
checks['gradients_postclipping_labeled']='After shared norm1 clipping' in source
checks['cap_includes_import_initialization']='signal.alarm(180)' in source and source.index('signal.alarm(180)')<source.index('import numpy as np')
checks['no_checkpoint_write']='torch.save' not in source
checks['failure_receipt_has_no_valid_timing_claim']="'timing_claim_valid':False" in source
checks['original_frozen_sources_unchanged']=all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in json.loads((P/'frozen_sources.json').read_text()).items())
checks['gated_postprocess_sources_unchanged']=all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in json.loads((P/'postprocess_source_hashes.json').read_text()).items())
r={'passed':all(checks.values()),'checks':checks,'benchmark_script_sha256':hashlib.sha256(source.encode()).hexdigest(),'scope':'AST, pre-import gate/lock mocks, atomic temporary receipt, and extracted step with dummy objects only. No real model/tensor operations, checkpoints, data or numerical timing.','findings':['Same complete pretrained model, same batch, same fresh optimizer grouping/state; only MambaDirection scan pointer changes.','Inspection copies occur only in first warm-up; recorded steps use inspect=False, with zero clone calls in mock test.','FP32 tolerances are explicit and fail closed. They must be numerically verified on actual pretrained checkpoint later; mock audit cannot certify those tolerances pass.','Compared gradients are clipped; small parameter divergence during repeated throwaway updates is permitted and not a saved-checkpoint quality comparison.','Seven paired timing trials provide descriptive CPU measurements, not broad hardware or GPU speed inference.','Alarm delivery may wait until an ongoing native/PyTorch operation returns;180s cap is checked by signal, not external hard-kill watchdog.']}
(P/'scan_upgrade_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));assert r['passed']
