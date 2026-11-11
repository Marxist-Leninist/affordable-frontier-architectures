"""Post-hoc preference index. No models, corpus loading, or new timed execution."""
import hashlib,json,math,random,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TASKS=('copy','reverse','sort','prefixsum_mod10')

def read(name):return json.loads((ROOT/name).read_text())
def avg(v):return statistics.mean(v)
def geo(logs):return math.exp(avg(logs))
def quantile(xs,q):
    xs=sorted(xs);f=(len(xs)-1)*q;i=int(f);j=min(i+1,len(xs)-1);return xs[i]+(xs[j]-xs[i])*(f-i)
def index(text,task,speed,quality_weight=.5,text_weight=.5):
    return 100*math.exp(quality_weight*(text_weight*text+(1-text_weight)*task)+(1-quality_weight)*speed)
def main():
    assert read('postprocess_status.json')['status']=='completed'
    e=read('evaluation_results.json');t=read('standalone_timings.json');p=read('protocol_long.json')
    rows={(r['kind'],r['seed'],r['budget']):r for r in e['rows']}
    assert len(rows)==18 and all(r['status']=='completed' for r in rows.values())
    components={};seed_rows=[]
    for seed in p['seeds']:
        z={};matched=[];task_loss=[]
        for name in ('wikitext','fineweb'):
            a=rows['transformer',seed,'fixed_data']['metrics'][name];b=rows['mamba',seed,'fixed_data']['metrics'][name]
            z[name]=-math.log(2)*(b['bpb']-a['bpb'])
            ta=rows['transformer',seed,'cpu_900s']['metrics'][name];tb=rows['mamba',seed,'cpu_900s']['metrics'][name]
            matched.append(-math.log(2)*(tb['bpb']-ta['bpb']))
        for name in TASKS:
            a=rows['transformer',seed,'separate_task_adaptation']['metrics'][name];b=rows['mamba',seed,'separate_task_adaptation']['metrics'][name]
            assert a['byte_accuracy']>0 and b['byte_accuracy']>0, 'Zero accuracy makes this geometric ratio undefined; do not add an arbitrary epsilon'
            z[name]=math.log(b['byte_accuracy']/a['byte_accuracy'])
            task_loss.append(-math.log(2)*(b['bpb']-a['bpb']))
        text=avg(z[k] for k in ('wikitext','fineweb'));task=avg(z[k] for k in TASKS)
        seed_rows.append({'seed':seed,'component_log_ratios':z,'text_log_ratio':text,'task_log_ratio':task,'matched_cpu_text_log_ratio':avg(matched),'task_log_probability_ratio':avg(task_loss),'task_without_prefix_sum_log_ratio':avg(z[k] for k in TASKS if k!='prefixsum_mod10')})
    text=avg(r['text_log_ratio'] for r in seed_rows);task=avg(r['task_log_ratio'] for r in seed_rows)
    speed=math.log(t['transformer']['train_step_p50_ms']/t['mamba']['train_step_p50_ms'])
    for name in ('wikitext','fineweb')+TASKS:
        vals=[r['component_log_ratios'][name] for r in seed_rows]
        components[name]={'mamba_over_transformer_geometric_ratio':geo(vals),'paired_seed_ratios':[math.exp(v) for v in vals],'quality_weight':.25 if name in ('wikitext','fineweb') else .125}
    trials={}
    for row in t['training']:trials.setdefault(row['trial'],{})[row['kind']]=row['wall_ms']
    assert len(trials)==40 and all(set(v)=={'transformer','mamba'} for v in trials.values())
    pairs=list(trials.values());rng=random.Random(881791);boot=[]
    for _ in range(5000):
        sr=[rng.choice(seed_rows) for _ in seed_rows];tr=[rng.choice(pairs) for _ in pairs]
        sp=math.log(statistics.median(x['transformer'] for x in tr)/statistics.median(x['mamba'] for x in tr))
        boot.append(index(avg(r['text_log_ratio'] for r in sr),avg(r['task_log_ratio'] for r in sr),sp))
    surface=[{'quality_weight':a/100,'text_weight_within_quality':b/100,'transformer':100.,'mamba':index(text,task,speed,a/100,b/100)} for a in range(101) for b in range(101)]
    text_threshold=-task/(text-task) if text!=task else None
    textonly_quality_threshold=-speed/(text-speed) if text!=speed else None
    actual=[]
    for seed in p['seeds']:
        secs={}
        for kind in ('transformer','mamba'):
            r=read(f"runs/final_{kind}_s{seed}_lr{e['selection'][kind]}/result.json")
            a=read(f"runs/final_{kind}_s{seed}_lr{e['selection'][kind]}/task_result.json")
            secs[kind]=r['training_cpu_seconds']+a['training_cpu_seconds']
        actual.append(math.log(secs['transformer']/secs['mamba']))
    out={'status':'completed','analysis_status':'Post-hoc user-requested preference index; weights chosen after component results existed, not a preregistered primary endpoint','normalization':'Transformer=100 reference. Text uses the ratio of geometric mean probability assigned to correct target bytes, 2**(bpb_T-bpb_M). Tasks use ratios of held-out per-byte accuracy. Ratios are geometrically averaged across paired seeds. No arbitrary min-max scaling, clipping, pseudocounts, benchmark-standard or intelligence interpretation.','default_weights':{'quality':.5,'training_throughput':.5,'within_quality_text':.5,'within_quality_synthetic_tasks':.5,'within_text_each_corpus':.5,'within_tasks_each_task':.25},'formula':'R=100*exp(alpha*(beta*mean_seed(mean_text[-ln(2)*(bpb_M-bpb_T)])+(1-beta)*mean_seed(mean_task[ln(acc_M/acc_T)]))+(1-alpha)*ln(median_step_ms_T/median_step_ms_M)); alpha=beta=0.5 by assistant default, not user-stated preferences.','components':components,'per_seed':seed_rows,'speed_ratio':math.exp(speed),'text_log_ratio':text,'task_log_ratio':task,'speed_log_ratio':speed,'quality_only_index':100*math.exp(.5*(text+task)),'default_index':{'transformer':100.,'mamba':index(text,task,speed)},'default_paired_seed_scores_fixed_timing':[index(r['text_log_ratio'],r['task_log_ratio'],speed) for r in seed_rows],'conditional_bootstrap_95_interval_mamba':[quantile(boot,.025),quantile(boot,.975)],'bootstrap_scope':'5000 resamples of three paired fitted seeds and 40 paired standalone timing trials. Conditional on fixed evaluation inputs and this host/session. This does not include fresh data/domain sampling, training hyperparameter uncertainty, other hardware or many independent model fits; three seeds are weak evidence for population inference.','matched_900_cpu_text_only_index':{'transformer':100.,'mamba':100*geo([r['matched_cpu_text_log_ratio'] for r in seed_rows])},'sensitivity':{'quality_only_text_weight_crossover':text_threshold,'text_only_quality_weight_crossover_when_speed_included':textonly_quality_threshold,'default_without_prefix_sum_task':index(text,avg(r['task_without_prefix_sum_log_ratio'] for r in seed_rows),speed),'default_using_task_correct_byte_log_probability_instead_of_accuracy':index(text,avg(r['task_log_probability_ratio'] for r in seed_rows),speed),'default_using_full_18000_update_cpu_throughput':index(text,task,avg(actual))},'conditions':['Quality combines the 16000-update text-only checkpoint and separate 2000-update task-adapted checkpoint for each paired seed. It is a multi-workload workflow index, not one checkpoint simultaneously evaluated on every task.','The default speed component is the standalone median whole-model training step with equal batch and shapes. It is not time to a target quality, learning efficiency, full training cost, or monetary utility.','The 900-CPU-second comparison is kept separate and contains text only. No speed multiplier is added to that already time-matched score.','Exact-sequence accuracy and task loss remain in the full report; exact match is not included again as an overlapping metric. Zero exact match on reverse/prefix-sum and shortcut-level prefix-sum byte accuracy prevent an arithmetic or intelligence claim.','A large throughput ratio can dominate modest predictive-probability ratios. Equal weights on log ratios encode that preference; other transformations and task choices are also judgments.','No universal overall winner follows. Score differences are not percentages of intelligence, economic value, or frontier-scale benefit.'],'input_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('evaluation_results.json','standalone_timings.json','protocol_long.json')},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    assert abs(index(0,0,0)-100)<1e-12
    assert abs(index(text,task,speed,1,1)-100*math.exp(text))<1e-12
    assert abs(index(text,task,speed,0,0)-100*math.exp(speed))<1e-12
    (ROOT/'composite_index.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    (ROOT/'composite_sensitivity.json').write_text(json.dumps(surface,separators=(',',':'),allow_nan=False)+'\n')
    lines=['# Post-hoc composite: explicit preferences, not intelligence','',f"Under the disclosed default, Transformer = 100 and Mamba = {out['default_index']['mamba']:.2f}. This ranking favors Transformer for a workload equally weighting relative quality and training throughput. It is not a universal architecture winner.",'','## Normalization and weights',out['normalization'],out['formula'],'Defaults are assistant-chosen for this requested comparison, not learned user preferences. Quality receives 50% and standalone training throughput 50%. Within quality, text gets 50% (two corpora equally) and adapted tasks get 50% (four tasks equally). The earlier full report and all component metrics are unchanged.','',f"Quality-only index at the same text/task weights: Mamba {out['quality_only_index']:.2f}, Transformer 100. Relative training throughput: {math.exp(speed):.4f} for Mamba versus Transformer.",'','## Components']
    for name,value in components.items():lines.append(f"- {name}: Mamba/Transformer ratio {value['mamba_over_transformer_geometric_ratio']:.5f}; quality weight {value['quality_weight']:.3f}; all paired seed ratios {value['paired_seed_ratios']}.")
    lines += ['','## Uncertainty',f"Default Mamba score by paired seed with timing fixed: {out['default_paired_seed_scores_fixed_timing']}. Conditional empirical 95% bootstrap interval: {out['conditional_bootstrap_95_interval_mamba']}.",out['bootstrap_scope'],'','## Sensitivity and matched time',f"With no speed weight, Mamba overtakes Transformer when text receives more than {100*text_threshold:.2f}% of quality weight. With text-only quality, Mamba wins only when quality receives more than {100*textonly_quality_threshold:.2f}% and speed receives the remainder. These thresholds reflect this particular ratio normalization, not a universal utility tradeoff.",f"At 900 training CPU seconds, the separate text-only index is Mamba {out['matched_900_cpu_text_only_index']['mamba']:.2f}, Transformer 100: Mamba's text prediction is better at matched compute time despite slower optimizer steps. Throughput and learning efficiency are different."]
    for name,value in out['sensitivity'].items():
        if name.startswith('default_'):lines.append(f'- {name}: Mamba {value:.2f}, Transformer 100.')
    lines += ['The complete two-weight sensitivity grid is in composite_sensitivity.json; axes cover every integer percentage from 0 to 100. This is analytic reweighting of existing measurements, not new model training.','','## Conditions and limits']+['- '+x for x in out['conditions']]
    (ROOT/'COMPOSITE_INDEX.md').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps({k:out[k] for k in ('default_index','quality_only_index','matched_900_cpu_text_only_index','conditional_bootstrap_95_interval_mamba','sensitivity')},indent=2))
if __name__=='__main__':main()
