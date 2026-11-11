"""Synthetic summary fixtures only; never opens experiment test results."""
import contextlib,fcntl,hashlib,io,json,math,tempfile
from pathlib import Path
import numpy as np
import summarize_long as s
P=Path(__file__).resolve().parent
checks={}
def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);s.P=root;rows=[];selection={'transformer':.001,'mamba':.001}
 for seed in [1,2,3]:
  for kind in selection:
   d=root/'runs'/f'final_{kind}_s{seed}_lr0.001'
   common={'training_cpu_seconds':10,'training_wall_seconds':11,'end_to_end_wall_seconds':12,'peak_rss_kib':100,'parameters':700,'batch_sha256':str(seed),'initial_common_hashes':{'head':str(seed)}}
   write(d/'result.json',common);write(d/'task_result.json',common)
   for budget in ['fixed_data','cpu_900s','separate_task_adaptation']:
    delta=np.array([0,2,2,2],dtype=float)+(seed-2)*.1
    bpb=np.ones(4)*5+(delta if kind=='mamba' else 0)
    metrics={name:{'bpb':float(bpb.mean()),'byte_accuracy':.1,'exact_match':0,'per_example_loss':(bpb*math.log(2)).tolist()} for name in ['wikitext','fineweb']}
    rows.append({'kind':kind,'seed':seed,'budget':budget,'status':'completed','metrics':metrics,'checkpoint_step':10,'checkpoint_extra':{'training_cpu_seconds':900.1},'conditioning_diagnostic':{'real_prefix':{'bpb':5},'shuffled_prefix':{'bpb':6}}})
 e={'selection':selection,'unigram_bpb':{'wikitext':6,'fineweb':6},'rows':rows}
 write(root/'evaluation_results.json',e);write(root/'evaluation'/'test_manifest.json',{name:{'group_ids':[0,1,1,1]} for name in ['wikitext','fineweb']})
 with contextlib.redirect_stdout(io.StringIO()):s.main()
 r=json.loads((root/'summary_long.json').read_text());b=r['paired_document_bootstrap']['fixed_data_wikitext']
 checks['nats_to_bpb_and_seed_averaging']=abs(b['mamba_minus_transformer_bpb']-1.5)<1e-12
 checks['target_weighted_not_equal_document_mean']=abs(b['mamba_minus_transformer_bpb']-1.0)>.1
 checks['document_resampling_not_independent_windows']=np.allclose(b['conditional_document_95_interval'],[0,2]) and b['documents']==2
 checks['paired_seed_means_retained']=np.allclose(b['seed_paired_means'],[1.4,1.5,1.6])
 checks['separate_sample_seed_spread']=abs(r['models']['mamba']['metrics']['fixed_data']['wikitext']['bpb']['sample_sd']-.1)<1e-12
 checks['conditioning_difference_correct_sign']=r['models']['mamba']['conditioning_bpb_difference_shuffled_minus_real']['mean']==1
 checks['conditional_scope_explicit']='Conditional on these three fitted seeds' in b['scope']
 e['rows']=[x for x in rows if not(x['seed']==3 and x['kind']=='mamba' and x['budget']=='cpu_900s')];write(root/'evaluation_results.json',e)
 with contextlib.redirect_stdout(io.StringIO()):s.main()
 r=json.loads((root/'summary_long.json').read_text())
 checks['incomplete900_not_silently_pooled']=r['models']['mamba']['metrics']['cpu_900s']['status']=='incomplete' and 'cpu_900s_wikitext' not in r['paired_document_bootstrap']
 d=root/'runs'/'final_mamba_s2_lr0.001';m=json.loads((d/'result.json').read_text());m['batch_sha256']='bad';write(d/'result.json',m)
 try:
  with contextlib.redirect_stdout(io.StringIO()):s.main()
  checks['pair_hash_mismatch_rejected']=False
 except AssertionError:checks['pair_hash_mismatch_rejected']=True
 a=(root/'lock').open('a');z=(root/'lock').open('a');fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:fcntl.flock(z,fcntl.LOCK_EX|fcntl.LOCK_NB);checks['exclusive_phase_lock_blocks_competing_handle']=False
 except BlockingIOError:checks['exclusive_phase_lock_blocks_competing_handle']=True
 a.close();z.close()
evalsrc=(P/'evaluate_long.py').read_text();bench=(P/'benchmark_long.py').read_text()
checks['all_fit_preflight_before_testsets']=evalsrc.index("assert r['status']=='completed'")<evalsrc.index("text_sets('test')") and evalsrc.index("assert sha((d/'tasks_final.pt').read_bytes())")<evalsrc.index("text_sets('test')")
checks['missing_final_raises']="if budget=='fixed_data':raise FileNotFoundError(path)" in evalsrc
checks['evaluation_timing_share_nonblocking_lock']=all("ROOT/'.evaluation_timing.lock'" in text and 'fcntl.LOCK_EX|fcntl.LOCK_NB' in text for text in [evalsrc,bench])
checks['timing_requires_evaluation_completion_file']="assert (ROOT/'evaluation_results.json').exists()" in bench
r={'passed':all(checks.values()),'checks':checks,'source_sha256':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['summarize_long.py','evaluate_long.py','benchmark_long.py']},'test_metrics_opened':False,'scope':'Synthetic complete/incomplete/mismatched-pair JSON fixtures and temporary lock only. No real held-out metrics or frozen sources altered.','interpretation':['Loss is mean nats per target byte converted by log(2); all windows have32 targets, so window-count weighting equals target-count weighting.','Fitted seeds are averaged first, then full documents resampled with their window sums/counts; this preserves within-document dependence rather than independently bootstrapping windows.','Interval is conditional on three fitted seeds and observed cohort, not uncertainty from training reruns or arbitrary population/architecture sampling. Across-seed sample SD is separately reported.','900s paired interval omitted unless all3 paired seeds exist; no incomplete-cohort point estimate substituted.','Manifest groups must align with per-example arrays; evaluator generates matching order. More explicit array-length/unique-seed assertions would harden malformed external artifact handling.','No inference about general intelligence, GPU performance or 600B transfer is justified.']}
(P/'summary_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'passed':r['passed'],'checks':checks},indent=2));assert r['passed']
