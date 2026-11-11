import hashlib,json,math,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
r=json.loads((P/'results.json').read_text()); s=json.loads((P/'summary.json').read_text());runs=r['runs']; checks={}
def check(name,value):
 checks[name]=bool(value)
 if not value:raise AssertionError(name)
check('all_six_final_runs_retained',len(runs)==6 and {(x['kind'],x['seed']) for x in runs}=={(k,i) for k in ['mamba','transformer'] for i in [1,2,3]})
check('all_final_800_steps',all(x['steps']==800 for x in runs))
check('paired_training_batch_hashes',all(len({x['batch_sha256'] for x in runs if x['seed']==i})==1 for i in [1,2,3]))
score=lambda m:.5*m['text']['loss_nats']+sum(m[t]['loss_nats'] for t in ['copy','reverse','sort'])/6
selected={}
for k in ['transformer','mamba']:
 tuning=[json.loads((P/f'tune_{k}_{lr}.json').read_text()) for lr in [.001,.003]]
 selected[k]=min(tuning,key=lambda x:score(x['curve'][-1]['validation']))['lr']
 check(k+'_selected_lr_validation',selected[k]==r['selection'][k] and all(x['lr']==selected[k] for x in runs if x['kind']==k))
 for x in (x for x in runs if x['kind']==k):
  check(f'{k}_{x["seed"]}_checkpoint_sha',hashlib.sha256((P/'checkpoints'/f'final_{k}_{x["seed"]}.pt').read_bytes()).hexdigest()==x['checkpoint_sha256'])
  check(f'{k}_{x["seed"]}_final_json_matches',json.loads((P/f'final_{k}_{x["seed"]}.json').read_text())==x)
 a=s['architectures'][k];rs=[x for x in runs if x['kind']==k]
 check(k+'_mean_training_seconds',math.isclose(a['mean_training_seconds'],statistics.mean(x['wall_seconds'] for x in rs),abs_tol=1e-12))
 for mode in ['test_fixed_steps','test_fixed_time']:
  for task in ['text','copy','reverse','sort']:
   for metric in ['bpb','token_accuracy','exact_match']:
    check(f'{k}_{mode}_{task}_{metric}',math.isclose(a[mode][task][metric]['mean'],statistics.mean(x[mode][task][metric] for x in rs),abs_tol=1e-12))
 for batch in [1,16]:
  vals=[x['ms'] for x in r['inference_trials'] if x['name'].startswith(k) and x['batch']==batch]
  check(f'{k}_batch{batch}_p50',len(vals)==180 and math.isclose(statistics.median(vals),a[f'inference_batch{batch}']['p50_ms'],abs_tol=1e-12))
T=s['architectures']['transformer']; M=s['architectures']['mamba']
check('training_ratio',math.isclose(s['training_speed_transformer_over_mamba'],M['mean_training_seconds']/T['mean_training_seconds'],abs_tol=1e-12))
check('near5s_transformer_better_text_bpb',T['test_fixed_time']['text']['bpb']['mean']<M['test_fixed_time']['text']['bpb']['mean'])
check('near5s_transformer_better_all_toy_byte_accuracy',all(T['test_fixed_time'][t]['token_accuracy']['mean']>M['test_fixed_time'][t]['token_accuracy']['mean'] for t in ['copy','reverse','sort']))
source=(P/'run.py').read_text()
check('run_source_unchanged_since_unit_audit',hashlib.sha256(source.encode()).hexdigest()==json.loads((P/'audit.json').read_text())['source_sha256']['run.py'])
check('test_evaluation_after_all_fits_source_order',source.index("test,tm=eval_sets('test')")>source.index("finals.append(train("))
result={'passed':all(checks.values()),'checks':checks,'selected_lr':selected,'time_budget_overshoot_seconds':{k:[x['time_budget_receipt']['train_wall_seconds']-5 for x in runs if x['kind']==k] for k in ['transformer','mamba']},'reviewed_report_source':'make_report.py','limitations':['Source review plus complete retained runs supports no test-based selection; provenance is not proof against arbitrary unlogged external experiments.','Near5s Transformer lead means text suffix bpb and toy byte accuracy, not every metric: reverse exact match is tied at zero.','Primary test only first-half coverage; secondary full-file extension is labeled and must not replace primary based on scores.','Speed is this short-sequence CPU implementation with optimized SDPA versus unfused associative scan, not general architecture/GPU speed.','Three tiny dense FP32 seeds and narrow tasks provide no broad intelligence ranking, convergence guarantee, or evidence for live 600B architecture changes.','Text bootstrap is conditional example uncertainty, not training-seed/document population uncertainty.','Arithmetic MAC estimates omit substantial work and do not establish matched compute.']}
(P/'final_claim_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'n_checks':len(checks),'selected_lr':selected,'overshoot':result['time_budget_overshoot_seconds']},indent=2))
w=json.loads((P/'whole_split_evaluation.json').read_text());report=(P/'REPORT.md').read_text()
assert len(w['runs'])==12 and w['examples']==1024
extension={budget:{kind:statistics.mean(x['metrics']['bpb'] for x in w['runs'] if x['budget']==budget and x['kind']==kind) for kind in ['transformer','mamba']} for budget in ['steps','time']}
for vals in extension.values():
 for val in vals.values(): assert f'{val:.5f}' in report
assert w['target16_matches_used_train']==63
for phrase in ['not establish a general CPU or GPU architecture ranking','No live 600B change was made','not autoregressive perplexity','near-equal budget','approximately the first half','not a representative population sample','not general reasoning or knowledge']:
 assert phrase in report
result['reviewed_report_source']='REPORT.md and make_report.py'
result['report_sha256']=hashlib.sha256(report.encode()).hexdigest()
result['whole_file_extension']={'all_12_seed_family_budget_combinations_retained':len({(x['seed'],x['kind'],x['budget']) for x in w['runs']})==12,'examples':1024,'mean_bpb':extension,'numbers_match_report':True}
result['report_review']='Numeric primary tables and secondary text means match artifacts. Explicit limitations reject general intelligence, GPU ranking and 600B extrapolation. Hardware, private AGILLM provenance and live-service status are parent-verified, outside this audit.'
(P/'final_claim_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('REPORT and secondary means verified:',extension)
