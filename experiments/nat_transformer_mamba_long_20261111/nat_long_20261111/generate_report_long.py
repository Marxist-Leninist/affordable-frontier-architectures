"""Render completed JSON results only. Never loads corpora, checkpoints, or models."""
import argparse, hashlib, json, os, statistics, tempfile
from pathlib import Path

KINDS = ('transformer', 'mamba')
BUDGETS = ('fixed_data', 'cpu_900s', 'separate_task_adaptation')

def read(p):
    return json.loads(p.read_text())

def preflight(root):
    protocol = read(root/'protocol_long.json')
    for name, digest in read(root/'frozen_sources.json').items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, f'Frozen source changed: {name}'
    assert read(root/'pipeline_status.json')['status'] == 'training_complete', 'Training is not complete'
    selection = read(root/'selection_long.json')
    runs = {}
    for kind in KINDS:
        for seed in protocol['seeds']:
            d = root/'runs'/f"final_{kind}_s{seed}_lr{selection['selected_lrs'][kind]}"
            r, t = read(d/'result.json'), read(d/'task_result.json')
            assert r['status'] == t['status'] == 'completed'
            assert r['steps'] == protocol['final_steps'] and t['steps'] == protocol['task_steps']
            runs[kind, seed] = (r, t)
    evaluation = read(root/'evaluation_results.json')
    summary = read(root/'summary_long.json')
    timing = read(root/'standalone_timings.json')
    manifest = read(root/'evaluation'/'test_manifest.json')
    expected = {(k,s,b) for k in KINDS for s in protocol['seeds'] for b in BUDGETS}
    keys = [(r['kind'], r['seed'], r['budget']) for r in evaluation['rows']]
    assert len(keys) == len(set(keys)) and set(keys) == expected, 'Missing or duplicate evaluation rows'
    for row in evaluation['rows']:
        if row['budget'] == 'cpu_900s' and row['status'] == 'unavailable_no_threshold_crossing':
            assert runs[row['kind'],row['seed']][0]['training_cpu_seconds'] < 900
            continue
        assert row['status'] == 'completed'
        if row['budget'] == 'cpu_900s':
            assert row['checkpoint_extra']['training_cpu_seconds'] >= 900
        if row['budget'] != 'separate_task_adaptation':
            for dataset in ('wikitext','fineweb'):
                m = row['metrics'][dataset]
                assert len(m['per_example_loss']) == m['n'] == len(manifest[dataset]['group_ids'])
    for seed in protocol['seeds']:
        a,b = runs['transformer',seed],runs['mamba',seed]
        assert a[0]['batch_sha256'] == b[0]['batch_sha256']
        assert a[1]['batch_sha256'] == b[1]['batch_sha256']
        assert a[0]['initial_common_hashes'] == b[0]['initial_common_hashes']
        assert str(seed) in summary['pairing_verified']
    for kind in KINDS:
        assert timing[kind]['train_step_p50_ms'] > 0
    return protocol,selection,runs,evaluation,summary,timing,manifest

def fmt(v):
    if isinstance(v,dict) and 'mean' in v:
        sd = v['sample_sd']
        return f"{v['mean']:.5f} ± {sd:.5f}" if sd is not None else f"{v['mean']:.5f}"
    return str(v)

def render(root,out,data):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'nat_report_mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p,selection,runs,e,s,t,manifest = data
    out.mkdir(parents=True,exist_ok=True)
    colors={'transformer':'#176B9B','mamba':'#C35431'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    charts=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/name,bbox_inches='tight');plt.close(fig);charts.append(name)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for kind in KINDS:
        for seed in p['seeds']:
            curve=runs[kind,seed][0]['validation_curves']
            for ax,key,label in zip(axes,('step','training_cpu_seconds','training_wall_seconds'),('Optimizer updates','Training CPU seconds','Training wall seconds')):
                ax.plot([x[key] for x in curve],[x['validation']['bpb'] for x in curve],marker='o',markersize=3,color=colors[kind],linestyle=['-','--',':'][seed-1],label=f'{kind}, seed {seed}')
                ax.set(xlabel=label,ylabel='Validation suffix bits/byte (lower better)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('All three seeds: validation learning curves, not held-out test curves')
    save(fig,'learning_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,dataset in zip(axes,('wikitext','fineweb')):
        for kind in KINDS:
            for budget,marker in [('fixed_data','o'),('cpu_900s','s')]:
                rows=[r for r in e['rows'] if r['kind']==kind and r['budget']==budget and r['status']=='completed']
                for r in rows:
                    x=runs[kind,r['seed']][0]['training_cpu_seconds'] if budget=='fixed_data' else r['checkpoint_extra']['training_cpu_seconds']
                    ax.scatter(x,r['metrics'][dataset]['bpb'],color=colors[kind],marker=marker)
                    ax.annotate(str(r['seed']),(x,r['metrics'][dataset]['bpb']),xytext=(4,3),textcoords='offset points',fontsize=7)
                if rows: ax.scatter([],[],color=colors[kind],marker=marker,label=f'{kind}, {budget}')
        ax.set(title=dataset,xlabel='Actual training CPU seconds',ylabel='Held-out suffix bits/byte');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Quality versus training time: individual seeds, no interpolated test curve')
    save(fig,'quality_vs_time.png')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for i,(name,v) in enumerate(s['paired_document_bootstrap'].items()):
        lo,hi=v['conditional_document_95_interval'];value=v['mamba_minus_transformer_bpb']
        ax.plot([lo,hi],[i,i],color='#444444');ax.scatter([value],[i],color='#176B9B');ax.scatter(v['seed_paired_means'],[i+.12]*len(v['seed_paired_means']),marker='x',color='#C35431')
    ax.set_yticks(range(len(s['paired_document_bootstrap'])),list(s['paired_document_bootstrap']));ax.axvline(0,color='gray',linestyle=':');ax.set(xlabel='Mamba minus Transformer bits/byte (negative favors Mamba)',title='Conditional document intervals (lines); individual paired seeds (crosses)')
    save(fig,'paired_uncertainty.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,title in zip(axes,('train_step_p50_ms','infer_batch1','infer_batch8'),('Training update','Inference batch 1','Inference batch 8')):
        vals=[t[k][key] if key=='train_step_p50_ms' else t[k][key]['p50_ms'] for k in KINDS]
        p95=[t[k]['train_step_p95_ms'] if key=='train_step_p50_ms' else t[k][key]['p95_ms'] for k in KINDS]
        ax.bar(KINDS,vals,color=[colors[k] for k in KINDS]);ax.scatter(KINDS,p95,color='black',marker='_',s=100,label='p95')
        ax.set(title=title,ylabel='Milliseconds (bars: median)');ax.legend(fontsize=8)
    fig.suptitle('Standalone whole-model timing; p95 markers are not uncertainty intervals')
    save(fig,'standalone_timing.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    tasks=('copy','reverse','sort','prefixsum_mod10')
    for ax,metric in zip(axes,('byte_accuracy','exact_match')):
        for kind,offset in zip(KINDS,(-.12,.12)):
            for i,task in enumerate(tasks):
                v=s['models'][kind]['metrics']['separate_task_adaptation'][task][metric]
                ax.scatter([i+offset]*len(v['values']),v['values'],color=colors[kind],marker='x',alpha=.7)
                ax.scatter(i+offset,v['mean'],color=colors[kind],label=kind if i==0 else None)
        ax.set_xticks(range(len(tasks)),tasks,rotation=15);ax.set(ylabel=metric.replace('_',' ')+' (fraction)',ylim=(-.03,1.03));ax.legend(fontsize=8)
    fig.suptitle('Separate synthetic adaptation: circles are means, crosses are all seeds')
    save(fig,'task_adaptation.png')
    lines=['# Single-pass NAT: Transformer versus optimized bidirectional Mamba-1','',
      '## Scope and design',
      f"Three-layer width-128 FP32 models: Transformer {p['architecture']['transformer_parameters']:,} parameters; Mamba {p['architecture']['mamba_parameters']:,}. Fixed 96-byte prefix and 32 fully blanked suffix positions. One forward pass predicts all suffix bytes independently conditional on the prefix. This is neither diffusion nor autoregressive generation.",
      f"Matched-data fits: {p['final_steps']:,} updates × batch {p['batch_size']} × 32 targets = {p['final_steps']*p['batch_size']*32:,} supervised target presentations per seed. Seeds: {p['seeds']}. Data order and common embedding/output initialization hashes are paired; architecture-specific weights differ.",
      'Text-only primary training differs from the earlier mixed-task pilot, so pilot differences cannot be attributed solely to scale or scan optimization.',
      f"Learning-rate grid {p['learning_rates']}; selected {selection['selected_lrs']}. Both selections are at the lowest tested boundary (0.0005), so this search does not locate an optimum. Selection uses 1,000-step seed-101 validation only, with the same 16,000-step schedule prefix.",
      '', '## Test quality',
      'All numbers below are across-seed mean ± sample SD, not a confidence interval. Bits/byte is conditional suffix cross-entropy, not autoregressive full-text likelihood. Byte accuracy and 32-byte exact-match are fractions.']
    for budget in BUDGETS:
        lines += ['',f'### {budget}']
        if budget=='separate_task_adaptation':
            lines += ['Separate 2,000-update adaptation on balanced-in-expectation synthetic tasks, LR 0.001; 1,024 unseen inputs per task. Text-only checkpoints are preserved. This measures task adaptation, not zero-shot capability or general intelligence. Uniform digit guesses have 10% expected byte accuracy. Merely copying the input unchanged gives 12.8125% expected byte accuracy on prefix-sum modulo 10 (the first byte is always correct, the other 31 match with probability 0.1). A score near that shortcut baseline is not evidence of cumulative arithmetic; exact 32-byte match is also reported.']
        for kind in KINDS:
            metrics=s['models'][kind]['metrics'][budget]
            if metrics.get('status')=='incomplete':
                lines += [f"- {kind}: unavailable as a complete three-seed comparison; completed seeds {metrics['completed_seeds']}."];continue
            for dataset,m in metrics.items():
                lines += [f"- {kind} / {dataset}: bits/byte {fmt(m['bpb'])}; byte accuracy {fmt(m['byte_accuracy'])}; exact-match {fmt(m['exact_match'])}."]
    lines += ['', '## Matched-time interpretation and uncertainty',
      '900-second checkpoints are the first whole update crossing 900 training CPU seconds. Overshoot and update counts are reported below. Training-update CPU and wall measurements include forward, loss, backward, gradient clipping and optimizer work, but exclude batch generation and hashing, logging, validation, checkpoint saving, and compilation. CPU time also excludes waiting. These are training-update compute measurements, not end-to-end elapsed time. End-to-end text wall time is reported separately. Missing threshold checkpoints are never replaced by final checkpoints. 300-second snapshots are archival only, not a planned held-out comparison here.',
      'The document bootstrap averages the three fitted seeds first, then resamples paired documents 5,000 times with target-count weighting. Its 95% intervals are conditional on those fitted seeds and observed cohorts. Across-seed SD and paired seed differences are separate; neither establishes general population or architectural uncertainty. Comparisons beyond the primary endpoint are exploratory.']
    for kind in KINDS:
        m=s['models'][kind]
        lines += [f"- {kind}: 900-second steps {m.get('time_budget_steps','unavailable')}; actual CPU seconds {m.get('time_budget_actual_cpu_seconds','unavailable')}."]
    for name,v in s['paired_document_bootstrap'].items():
        lines += [f"- {name}: Mamba − Transformer = {v['mamba_minus_transformer_bpb']:.5f} bits/byte, conditional document 95% interval {v['conditional_document_95_interval']}; {v['documents']} documents; paired seed means {v['seed_paired_means']}."]
    lines += ['', '## Baselines and conditioning',f"Training-byte unigram baseline (half-count smoothing): {e['unigram_bpb']} bits/byte."]
    for kind in KINDS:
        lines += [f"- {kind}: shuffled-prefix minus real-prefix bits/byte {fmt(s['models'][kind]['conditioning_bpb_difference_shuffled_minus_real'])} on 2,048 fixed WikiText test windows. Positive means the real prefix helps. Rotation can retain same-document context; this is a diagnostic, not an independent causal test."]
    compute=read(root/'compute_accounting_long.json')
    lines += ['', '## Active-compute accounting (partial static estimate)',
      'Matching parameter counts is not matching compute. At batch 1, sequence length 128, width 128, three layers and 32 output positions:',
      f"- Transformer matrix MACs: {compute['transformer_matrix_macs']:,}. Formula: {compute['transformer_formula']}.",
      f"- Bidirectional Mamba matrix/convolution MACs: {compute['mamba_matrix_and_convolution_macs']:,}. Formula: {compute['mamba_formula']}.",
      'The Mamba depthwise convolution produces 131 positions with padding before truncation to 128; the partial total includes the 18,432 nominal MACs for those discarded positions across all branches/layers.',
      f"- Mamba state elements updated per forward: {compute['mamba_state_elements_updated_per_forward']:,}. Formula: {compute['state_elements_formula']}. This is a separate state-element count, not MACs or FLOPs.",
      'The partial MAC totals omit normalization, activations, RoPE, softmax, the selective scan recurrence and other pointwise operations. They also omit backward/optimizer work, memory movement and dispatch. One MAC means one multiply-accumulate; these are not complete FLOP totals. Do not infer equal FLOPs or less total work for Mamba.',
      'Both dense arms invoke all layer modules; embedding lookup uses indexed rows, and neither arm has an expert router. Interpret these partial estimates alongside measured whole-model runtime below.']
    lines += ['', '## Timing and memory',t['host'],t['scope'], 'Training timings from concurrent pinned runs and serial standalone throughput are different measurements.']
    for kind in KINDS:
        m=s['models'][kind]
        for key in ('training_cpu_seconds','training_wall_seconds','end_to_end_text_wall_seconds','task_training_cpu_seconds','peak_rss_kib'):
            lines += [f'- {kind} {key}: {fmt(m[key])}.']
        lines += [f"- {kind} compiler/cache-load seconds by seed: {[runs[kind,seed][0]['compiler_load_seconds'] for seed in p['seeds']]}",f"- {kind} standalone training p50/p95: {t[kind]['train_step_p50_ms']:.3f}/{t[kind]['train_step_p95_ms']:.3f} ms; {t[kind]['supervised_byte_targets_per_second']:.1f} supervised target bytes/s."]
        for batch in (1,8):
            m=t[kind][f'infer_batch{batch}'];lines += [f"- {kind} standalone inference batch {batch} p50/p95: {m['p50_ms']:.3f}/{m['p95_ms']:.3f} ms."]
    lines += [f"Standalone training Transformer/Mamba p50 ratio: {t['transformer']['train_step_p50_ms']/t['mamba']['train_step_p50_ms']:.3f}. Greater than one means Mamba faster.",f"Standalone benchmark process peak RSS: {t['peak_process_rss_kib']} KiB, combined high-water mark, not per-architecture memory."]
    final_text_cpu=sum(runs[k,seed][0]['training_cpu_seconds'] for k in KINDS for seed in p['seeds'])
    final_task_cpu=sum(runs[k,seed][1]['training_cpu_seconds'] for k in KINDS for seed in p['seeds'])
    tuning=[]
    for kind in KINDS:
        for lr in p['learning_rates']:
            path=root/'runs'/f"tune_{kind}_s{p['tune_seed']}_lr{lr}"/'result.json'
            if path.exists():
                trial=read(path)
                if trial['status']=='completed' and trial['steps']==p['tune_steps']:tuning.append(trial)
    expected_tuning=len(KINDS)*len(p['learning_rates'])
    tuning_cpu=sum(r['training_cpu_seconds'] for r in tuning) if len(tuning)==expected_tuning else None
    economy={'final_text_update_cpu_seconds':final_text_cpu,'separate_task_update_cpu_seconds':final_task_cpu,'tuning_update_cpu_seconds':tuning_cpu,'expected_tuning_runs':expected_tuning,'available_completed_tuning_runs':len(tuning),'total_measured_update_cpu_hours':(final_text_cpu+final_task_cpu+tuning_cpu)/3600 if tuning_cpu is not None else None,'pipeline_elapsed_wall_seconds':read(root/'pipeline_status.json').get('elapsed_seconds'),'monetary_cost':'Not measured. No rental or purchase action was made for this experiment. Existing platform billing is not inferred.','scope':'Measured training-update CPU time only; excludes preparation, validation/testing, saving, compilation, reporting and backup. Pipeline elapsed wall time includes concurrent jobs and is not summed CPU time.'}
    (out/'resource_accounting.json').write_text(json.dumps(economy,indent=2)+'\n')
    lines += ['', '## Resource accounting and monetary limits',
      f"Measured update CPU seconds: final text {final_text_cpu:.3f}; separate task adaptation {final_task_cpu:.3f}; tuning {tuning_cpu if tuning_cpu is not None else 'unavailable'}. Total measured update core-hours: {economy['total_measured_update_cpu_hours'] if tuning_cpu is not None else 'unavailable'}.",
      f"Whole training-pipeline elapsed wall seconds: {economy['pipeline_elapsed_wall_seconds']}. This includes overlapping pinned jobs and must not be added to their CPU seconds.",
      economy['scope'], economy['monetary_cost'],
      'These measurements do not establish dollar cost per successful task, dollar savings, or a 600B-scale cost projection. Runtime and quality are reported separately rather than combined into an invented intelligence or value score.', '', '## Optimization provenance']
    fast=read(root/'fast_scan_results.json')
    lines += ['Mamba architecture/initialization reference: Tri Dao and Albert Gu, state-spaces/mamba commit e9594ce1c732d97440f0332fdc43170a2294dbfa, Apache License 2.0. This is an independent implementation. Private user-owned AGILLM source informed context but is not redistributed. See THIRD_PARTY_NOTICES.md.', 'Custom native C++ affine recurrence and first-order backward replace the generic PyTorch scan with linear work. FP32/FP64 CPU parity, gradients and AdamW-update tests were checked; no official fused GPU Mamba-2 kernel is used.']
    for b in fast['benchmarks']:
        lines += [f"- MambaDirection block microbenchmark B={b['B']}, T={b['T']}, width={b['d']}, state={b['state']}: {b['speedup_vs_parallel']:.3f}× over prior generic parallel scan. This is not a whole-model speedup or a Transformer comparison."]
    upgrade_path=root/'scan_upgrade_whole_model.json'
    if upgrade_path.exists():
        upgrade=read(upgrade_path)
        if upgrade['status']=='completed':
            speed=upgrade['whole_model_training_speedup']
            expected=upgrade['summary']['torch_parallel_reference']['median_seconds']/upgrade['summary']['native_linear']['median_seconds']
            assert abs(speed-expected)<1e-10 and speed>0, 'Invalid whole-model speed ratio'
            lines += [f"- Separate complete-model scan upgrade: {speed:.3f}× training-step speedup for the same 765,184-parameter bidirectional Mamba, batch 8, length 128, CPU FP32. Median native-linear / generic-parallel step seconds: {upgrade['summary']['native_linear']['median_seconds']:.6f} / {upgrade['summary']['torch_parallel_reference']['median_seconds']:.6f}. Two warm-up steps, seven randomized paired trials; only the scan changes. This is not a Transformer comparison.",
              f"  First-step maximum absolute differences: {upgrade['parity_max_absolute_error']}. Gradients are compared after common norm-1 clipping. Both copies use identical final seed-1 weights and fresh AdamW states. Forward, backward and updated parameters pass the recorded FP32 tolerances, not bitwise equality. No saved checkpoint is modified."]
        else:
            lines += ['- The optional complete-model scan upgrade check did not complete successfully. No speedup claim is made from it; primitive/block parity and timing above remain separate evidence.']
    lines += [f"Cold compile {fast['compile_seconds']:.4f}s; first use {fast['first_use_seconds']:.4f}s, excluded from steady-state block timing.",'','## Corpus provenance and exclusions',
      'WikiText: Salesforce/wikitext, wikitext-2-raw-v1, revision b08601e04326c79dfdd32d625aee71d232d685c3. UTF-8 LF-joined original rows; 10,951,562 training bytes. Previously inspected metadata for this same pinned dataset reports CC-BY-SA-3.0 and GFDL; other licensing prose links CC-BY-SA-4.0. Preserve the original-source rights and this unresolved licensing inconsistency; see THIRD_PARTY_NOTICES.md. Corpus content is not included in these deliverables.',
      'Training samples uniform valid article-contained 128-byte windows; windows crossing article boundaries and short documents are ineligible. Evaluation uses nonoverlapping article-contained windows, excludes exact 128-byte training/duplicate windows, and retains/report counts of 32-byte target overlaps.',
      'FineWeb-Edu: HuggingFaceFW/fineweb-edu revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, rows 1500–1599, each document truncated to its first 4,097 UTF-8 bytes. Dataset card ODC-BY; underlying web text may retain third-party rights. This cohort was used in earlier research, but is new to this NAT comparison and was not used for its tuning. It is a bounded cross-domain cohort, not a representative sample of the whole web. The bounded viewer download script fails closed if its revision or content changes; exact future retrieval depends on that snapshot remaining available. Cached corpus bytes are not redistributed.',
      'Official WikiText test and separate FineWeb cohort are held out from this fit/tuning, not proof of universal independence or absence of all overlap.']
    for name,m in manifest.items():
        lines += [f"- {name}: {m['evaluated_windows']} windows / {m['documents']} documents; excluded exact128 train/duplicate {m['excluded_exact128_train_or_duplicate']}; target32 training matches {m['target32_matches_training_all_windows']}; selection SHA-256 {m['selection_sha256']}."]
    lines += ['', '## Limitations']+[f'- {x}' for x in p['limitations']]+['- No claim about intelligence, 600B-scale models, GPU performance, AGILLM routing/tokenizer equivalence, or convergence follows from this experiment.']
    if (root/'composite_index.json').exists():
        ci=read(root/'composite_index.json')
        lines += ['', '## Requested post-hoc composite',f"Under an explicit assistant-default weighting of 50% relative quality and 50% training throughput, Transformer is anchored at 100 and Mamba scores {ci['default_index']['mamba']:.2f}. Quality gives equal weight to text and separately adapted tasks. This is a preference-dependent multi-workload index, not an intelligence score, a preregistered endpoint, or a universal winner. Mamba still wins the separately time-matched text comparison. See [the complete normalization, uncertainty and weight sensitivity](../COMPOSITE_INDEX.md); all component metrics above remain unchanged."]
    lines += ['', '## Figures']+[f'![{x}]({x})' for x in charts]
    (out/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
    (out/'THIRD_PARTY_NOTICES.md').write_text((root/'THIRD_PARTY_NOTICES.md').read_text())
    (out/'compute_accounting_long.json').write_text((root/'compute_accounting_long.json').read_text())
    receipts={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('protocol_long.json','selection_long.json','evaluation_results.json','summary_long.json','standalone_timings.json','compute_accounting_long.json')}
    if upgrade_path.exists():receipts[upgrade_path.name]=hashlib.sha256(upgrade_path.read_bytes()).hexdigest()
    (out/'report_input_hashes.json').write_text(json.dumps(receipts,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true')
    a=parser.parse_args()
    try: data=preflight(a.root)
    except (AssertionError,FileNotFoundError,KeyError,ValueError) as exc:
        raise SystemExit(f'REPORT NOT READY: {exc}')
    if a.check_only: print('All report inputs passed preflight; no output written.');return
    render(a.root,a.output or a.root/'report',data)
    print('Report and figures written. No corpora or checkpoint content read or included.')

if __name__=='__main__':main()
