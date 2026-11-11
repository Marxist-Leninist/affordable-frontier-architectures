"""Synthetic/mock audit only; never loads held-out benchmark data or fits."""
import hashlib,json,math,tempfile
from pathlib import Path
import numpy as np
import torch
import evaluate_long as e
from train_long import evaluate
P=Path(__file__).resolve().parent
torch.set_num_threads(1)
checks={}
class Uniform(torch.nn.Module):
 def forward(self,prefix,out_len=32):return torch.zeros(len(prefix),out_len,256)
m=Uniform();p=torch.zeros(8,96,dtype=torch.long);y=torch.arange(256).view(8,32)
a=evaluate(m,p,y,batch=3,return_rows=True)
checks['suffix_loss_aggregation']=abs(a['bpb']-8)<1e-5 and a['n']==8 and abs(a['byte_accuracy']-1/256)<1e-10 and a['exact_match']==0
checks['per_example_rows_preserve_dataset_order']=len(a['per_example_loss'])==8 and a['per_example_accuracy'][0]==1/32 and sum(a['per_example_accuracy'][1:])==0
class Tiny(torch.nn.Module):
 def __init__(self,kind):super().__init__();self.weight=torch.nn.Parameter(torch.tensor([0.]))
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);(root/'pipeline_status.json').write_text(json.dumps({'status':'final_training'}))
 originalroot=e.ROOT;e.ROOT=root
 def forbidden(*args,**kw):raise AssertionError('held-out loader called before gate')
 oldtext,oldfine=e.text_sets,e.fineweb;e.text_sets=forbidden;e.fineweb=forbidden
 try:e.main();checks['training_completion_gate_before_test_loading']=False
 except AssertionError as error:checks['training_completion_gate_before_test_loading']='Do not evaluate' in str(error)
 e.ROOT=originalroot;e.text_sets=oldtext;e.fineweb=oldfine
 oldnat=e.NAT;e.NAT=Tiny
 path=root/'mock.pt';torch.save({'model':{'weight':torch.tensor([7.])},'step':123,'extra':{'threshold_seconds':900}},path)
 loaded,receipt=e.load('mamba',path);checks['checkpoint_restoration_and_receipt']=loaded.weight.item()==7 and not loaded.training and receipt['step']==123 and receipt['extra']['threshold_seconds']==900
 e.NAT=oldnat
# Independent unigram/rotation math on synthetic bytes.
train=np.array([0,0,1],dtype=np.uint8);counts=np.bincount(train,minlength=256).astype(float)+.5
checks['smoothed_unigram_normalized']=np.isclose((counts/counts.sum()).sum(),1) and np.all(counts>0)
pp=torch.arange(8*96).reshape(8,96);yy=y.clone();rotated=pp.roll(1,0)
checks['conditioning_rotation_keeps_targets_and_reassigns_all_prefixes']=torch.equal(yy,y) and bool((rotated!=pp).any(1).all()) and torch.equal(rotated[0],pp[-1])
sources={x:(P/x).read_text() for x in ['evaluate_long.py','benchmark_long.py']}
checks['per_document_groups_retained_in_manifest']="'group_ids':g.tolist()" in sources['evaluate_long.py']
checks['missing900_explicitly_unavailable']="'unavailable_no_threshold_crossing'" in sources['evaluate_long.py']
checks['task_checkpoint_separate']="directory/'tasks_final.pt'" in sources['evaluate_long.py']
checks['benchmark_does_not_save_checkpoint']='torch.save' not in sources['benchmark_long.py']
checks['benchmark_inference_no_grad']='torch.inference_mode()' in sources['benchmark_long.py']
checks['benchmark_warns_updated_throwaway_models']='Inference follows those bounded throwaway steps' in sources['benchmark_long.py']
checks={k:bool(v) for k,v in checks.items()}
r={'passed':all(checks.values()),'checks':checks,'source_sha256':{x:hashlib.sha256(v.encode()).hexdigest() for x,v in sources.items()},'test_metrics_opened':False,'scope':'Synthetic aggregation/model restoration, mocked pre-completion gate, math and source inspection only. No held-out corpus loading, real checkpoints, training, or timing.','findings':['Full WikiText/FineWeb rows scored in order; aligned document group IDs retained for later grouped bootstrap. Statistical aggregation/bootstrap code is not included in these scripts and needs later audit.','Missing900s snapshots are marked unavailable; comparison must restrict to truly matched available seeds and report gaps. Missing required final.pt should raise rather than share optional-threshold handling.','Strengthen preflight: all6 result/task completed statuses, expected steps, and final checkpoint hashes should be checked before held-out sets loaded. Current pipeline completion flag gates access and orchestrator sets it only after all runs.','Benchmark must run serially after evaluation exits. Pipeline training_complete alone does not exclude another post-training job.','Unigram uses only training byte counts with half-count smoothing; diagnostic rotates selected prefixes, not independent random permutation, and may preserve same-document context.','Standalone inference is after45 throwaway training updates; clearly labeled runtime-only, not final-checkpoint quality. No saved parameters modified.','Peak RSS is combined process high-water mark, not architecture-specific memory usage.','CPU-only fixed128-position dense FP32 benchmark cannot establish GPU speed, broad intelligence, or 600B transfer.']}
(P/'eval_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'passed':r['passed'],'checks':checks},indent=2))
assert r['passed']
