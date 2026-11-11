"""Finalized bounded sweep reporting. Saved metrics only; no model imports/compute.

The original report_sweep.py is unchanged and supplies only the SVG drawing
function. This generator validates the prospective amendment, prior synthetic
calibration debit, finalization receipts, selections and checkpoint identities.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from report_sweep import plot as base_plot

PHASES = {'all101': list(range(101)), 'selected': [0, 2, 100]}
WINDOWS = {'all101': 8, 'selected': 64}
KINDS, SEEDS, DOMAINS = ('split', 'whole'), (1, 2, 3), ('test', 'fineweb')
JOBS = tuple(f'{k}_{s}' for s in SEEDS for k in KINDS)
METRICS = ('suffix_bpb', 'byte_accuracy', 'exact_32_byte_match',
           'forward_cpu_seconds', 'forward_wall_seconds')
STATUSES = {'completed', 'capped', 'nonfinite_logits', 'nonfinite_loss',
    'missing_resource_cap', 'missing_checkpoint', 'not_run', 'missing_cell',
    'missing_file', 'malformed_cell'}
ORIGINAL_RECEIPTS = ('calibration_raw.json', 'cohort_selection.json', 'blocker.json')
ORIGINAL_SOURCES = {'model.py', 'data_loop.py', 'train.py', 'test_correctness.py',
    'run_all.py', 'protocol.json', 'summarize.py', 'sweep.py', 'sweep_protocol.json',
    'control.py', 'runtime_plan.py', 'test_runtime_plan.py'}
AMENDED_SOURCES = {'bounded_sweep.py', 'bounded_runtime_plan.py',
    'bounded_sweep_protocol.json', 'test_bounded_runtime_plan.py',
    'BOUNDED_AMENDMENT.md', 'source_hashes.json'}
EXPECTED_PROTOCOL = dict(counts=list(range(101)), selected_counts=[0, 2, 100],
    cohort_windows_per_domain=8, selected_windows_per_domain=64, batch=8,
    domains=['test', 'fineweb'], seeds=[1, 2, 3], topologies=['split', 'whole'],
    global_wall_cap_seconds=7200, per_checkpoint_wall_cap_seconds=1200,
    margin=1.5, global_reserve_seconds=600, per_checkpoint_prediction_limit_seconds=1050,
    checkpoint='matched_step.pt', trained_loop_count=2, unique_parameters=764304,
    invoked_module_parameters_zero_loop=353424,
    invoked_module_parameters_positive_loop=764304,
    outer_blocks={'input': 8, 'output': 8}, hidden=117, core=1, threads=1)


class ReportError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ReportError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def hash_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def reject_constant(value):
        raise ReportError(f'Nonfinite JSON constant in {path.name}: {value}')
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key in {path.name}: {key}')
            result[key] = value
        return result
    return json.loads(path.read_text(), parse_constant=reject_constant, object_pairs_hook=unique_keys)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def csvout(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def linspace_indices(total, n):
    """Match np.linspace(0,total-1,n,dtype=int64), including float boundaries."""
    step=(total-1)/(n-1)
    result=[int(i*step) for i in range(n)]
    result[-1]=total-1
    return result


def plot(path,rows,phase,domain,metric):
    base_plot(path,rows,phase,domain,metric)
    old='Exploratory sweep; trained r=2. Gaps indicate missing/invalid cells. All seeds shown.'
    subtitle=('Bounded amendment: 8 windows/domain, all 101 counts. Trained r=2; gaps are missing/invalid cells.'
              if phase=='all101' else
              'Bounded amendment: 64 windows/domain at r=0,2,100 only. Trained r=2; connecting lines are guides.')
    path.write_text(path.read_text().replace(old,subtitle))


def verify_original(raw, original, blocker):
    require(blocker.get('status') == 'blocked_no_fit' and blocker.get('heldout_loaded') is False,
            'Original sweep must be blocked before held-out loading')
    require(original.get('status') == 'blocked_no_fit' and original.get('windows_per_domain') is None,
            'Original sweep selection must be blocked_no_fit')
    for key, expected in dict(synthetic_input='arange(8*96) modulo256',
            synthetic_target='arange(8*32) modulo256', batch=8, counts=[0,1,2,8,32,100],
            repetitions=2, warmup_loop_count=2).items():
        require(raw.get(key) == expected, 'Original calibration input mismatch: '+key)
    warmups, samples = raw.get('warmups', []), raw.get('samples', [])
    require(len(warmups) == 6 and {w['job'] for w in warmups} == set(JOBS)
            and all(w['status'] == 'completed' for w in warmups), 'Incomplete original warmups')
    require(len(samples) == 72, 'Incomplete original synthetic calibration')
    prior, elapsed = original.get('elapsed_debited_seconds'), original.get('calibration_elapsed_seconds')
    require(finite(prior) and finite(elapsed) and 0 <= elapsed <= 180 and prior >= elapsed,
            'Invalid original calibration runtime debit')
    envelopes = {}
    for job in JOBS:
        rows = [s for s in samples if s['job'] == job]
        expected = {(count, repeat) for count in (0,1,2,8,32,100) for repeat in (0,1)}
        require(len(rows) == 12 and {(s['loops'],s['repeat']) for s in rows} == expected
                and all(s['status'] == 'completed' and finite(s['seconds']) and s['seconds'] > 0 for s in rows),
                'Invalid original calibration samples')
        a = max(s['seconds'] for s in rows if s['loops'] == 0)
        b = max(max(0.,s['seconds']-a)/s['loops'] for s in rows if s['loops'] > 0)
        envelopes[job] = {'a':a,'b':b}
        require(original.get('envelopes',{}).get(job) == envelopes[job], 'Original runtime envelope mismatch')
    candidates = original.get('candidates', [])
    require(len(candidates) == 3 and [c['windows_per_domain'] for c in candidates] == [32,16,8],
            'Original candidate plan mismatch')
    for candidate in candidates:
        n = candidate['windows_per_domain']
        times = {j:(n/4)*(101*e['a']+5050*e['b'])+32*(7*e['a']+147*e['b']) for j,e in envelopes.items()}
        require(set(candidate['predicted_seconds_per_checkpoint']) == set(JOBS) and
                all(close(v, candidate['predicted_seconds_per_checkpoint'][j]) for j,v in times.items()),
                'Original prediction mismatch')
        bound = prior+600+1.5*sum(times.values())
        fits = bound <= 7200 and all(1.5*t <= 1050 for t in times.values())
        require(close(bound,candidate['global_bound_seconds']) and candidate.get('fits') is fits and not fits,
                'Original no-fit decision mismatch')
    require(set(original.get('checkpoint_hashes',{})) == set(JOBS) and
            all(hash_valid(v) for v in original['checkpoint_hashes'].values()), 'Original checkpoint identities incomplete')
    return prior, envelopes


def collect(src, original_dir=None):
    src = src.resolve()
    root = src.parent
    original_dir = (original_dir or root/'sweep_results').resolve()
    # status.json is emitted only after every planned cell has a terminal ledger entry.
    # Check it before opening any model score file.
    status = read(src/'status.json')
    runs = status.get('runs', [])
    require(len(runs) == 6 and {(r.get('kind'),r.get('seed')) for r in runs} ==
            {(k,s) for k in KINDS for s in SEEDS}, 'Require finalized six-run status')
    for r in runs:
        keys = ('completed_cells','nonfinite_cells','missing_or_capped_cells','total_cells')
        require(all(integer(r.get(k)) and r[k] >= 0 for k in keys) and
                r['total_cells'] == 208 and sum(r[k] for k in keys[:-1]) == 208,
                'Final status must account for all 208 cells per model')
    require(all(finite(status.get(k)) and status[k] >= 0 for k in
                ('prior_elapsed_debited_seconds','new_elapsed_seconds','combined_elapsed_seconds'))
            and isinstance(status.get('global_capped'),bool), 'Require final timing and cap status')
    require(math.isclose(status['combined_elapsed_seconds'], status['prior_elapsed_debited_seconds']+
                        status['new_elapsed_seconds'], rel_tol=1e-6, abs_tol=.01), 'Combined runtime debit mismatch')
    manifest, selection, amendment = (read(src/name) for name in
        ('manifest.json','cohort_selection.json','amendment_receipt.json'))
    raw_hashes = {name:sha(src/name) for name in
        ('status.json','manifest.json','cohort_selection.json','amendment_receipt.json')}
    protocol = manifest['protocol']
    require(amendment.get('protocol') == protocol, 'Amendment protocol differs from cohort manifest')
    for key,value in EXPECTED_PROTOCOL.items():
        require(protocol.get(key) == value, 'Bounded protocol mismatch: '+key)
    require(amendment.get('heldout_loaded') is False, 'Amendment was not frozen before held-out loading')
    require(manifest.get('selection') == selection, 'Frozen selection differs from manifest')
    original_sources, amended_sources = read(root/'source_hashes.json'), read(root/'bounded_source_hashes.json')
    require(set(original_sources) == ORIGINAL_SOURCES and set(amended_sources) == AMENDED_SOURCES,
            'Unexpected original/amended source manifest scope')
    require(manifest.get('source_hashes') == amendment.get('original_source_hashes') == original_sources and
            manifest.get('amended_source_hashes') == amendment.get('amended_source_hashes') == amended_sources,
            'Original/amended source identity mismatch')
    source_hashes = {}
    for identities in (original_sources, amended_sources):
        for name, digest in identities.items():
            require(hash_valid(digest) and sha(root/name) == digest, 'Source changed: '+name)
            source_hashes[name] = digest
    source_hashes['bounded_source_hashes.json'] = sha(root/'bounded_source_hashes.json')
    calibration_manifest = amendment['calibration_receipt_manifest']
    require(calibration_manifest.get('original_source_manifest_sha256') == sha(root/'source_hashes.json') and
            set(calibration_manifest.get('receipts_sha256',{})) == set(ORIGINAL_RECEIPTS),
            'Calibration receipt/source binding mismatch')
    calibration_hashes = {}
    for name in ORIGINAL_RECEIPTS:
        digest = sha(original_dir/name)
        require(digest == calibration_manifest['receipts_sha256'][name], 'Original calibration receipt changed: '+name)
        calibration_hashes[name] = digest
    raw, original, blocker = (read(original_dir/name) for name in ORIGINAL_RECEIPTS)
    prior, envelopes = verify_original(raw, original, blocker)
    require(selection.get('status') == 'selected' and selection.get('windows_per_domain') == 8 and
            selection.get('selected_windows_per_domain') == 64 and selection.get('selected_counts') == [0,2,100],
            'Bounded selection must be fixed 8/64 windows and selected counts 0/2/100')
    require(selection.get('checkpoint_hashes') == original['checkpoint_hashes'] and
            selection.get('envelopes') == envelopes, 'Amended checkpoint/calibration identity mismatch')
    require(close(selection.get('prior_elapsed_debited_seconds'),prior) and
            close(status['prior_elapsed_debited_seconds'],prior), 'Original runtime debit was not retained')
    setup = selection.get('new_setup_debited_seconds')
    require(finite(setup) and 0 <= setup <= status['new_elapsed_seconds'], 'Invalid new setup runtime debit')
    expected_times = {j:2*(101*e['a']+5050*e['b'])+16*(3*e['a']+102*e['b']) for j,e in envelopes.items()}
    require(set(selection.get('predicted_seconds_per_checkpoint',{})) == set(JOBS) and
            all(close(v,selection['predicted_seconds_per_checkpoint'][j]) for j,v in expected_times.items()),
            'Bounded runtime prediction mismatch')
    expected_bound = prior+setup+600+1.5*sum(expected_times.values())
    require(close(selection.get('global_bound_seconds'),expected_bound) and expected_bound <= 7200 and
            all(1.5*t <= 1050 for t in expected_times.values()), 'Bounded prediction does not fit original caps')
    verified = amendment.get('standard_evaluations_verified', [])
    require(len(verified) == 6 and {(r['kind'],r['seed']) for r in verified} ==
            {(k,s) for k in KINDS for s in SEEDS}, 'Missing standard-evaluation receipt identities')
    for receipt in verified:
        require(all(hash_valid(receipt.get(k)) for k in
            ('training_receipt_sha256','standard_result_sha256','standard_evaluation_sha256')),
            'Invalid standard-evaluation receipt hash')
    require(set(manifest['cohorts']) == set(PHASES), 'Wrong cohort phases')
    cohort_hashes = {}
    for phase in PHASES:
        require(set(manifest['cohorts'][phase]) == set(DOMAINS), 'Wrong cohort domains')
        for domain in DOMAINS:
            c = manifest['cohorts'][phase][domain]
            n, total = WINDOWS[phase], (10059 if domain == 'test' else 2134)
            indices = c.get('indices', [])
            require(c.get('windows') == n and hash_valid(c.get('selection_sha256')) and
                    c.get('source',{}).get('evaluated_windows') == total and
                    len(indices) == n and all(integer(v) for v in indices) and
                    indices == linspace_indices(total,n),
                    'Invalid fixed evenly-spaced cohort manifest')
            cohort_hashes[f'{phase}/{domain}'] = c['selection_sha256']

    rows, runtimes, groups_reference, batch_hashes = [], [], {}, {}
    source_status_by_run = {}
    for kind in KINDS:
        for seed in SEEDS:
            filename = f'{kind}_s{seed}.json'
            path, lookup = src/filename, {}
            if path.exists():
                obj = read(path)
                raw_hashes[filename] = sha(path)
                require(obj.get('kind') == kind and obj.get('seed') == seed and
                        obj.get('checkpoint_sha256') == selection['checkpoint_hashes'][f'{kind}_{seed}'],
                        'Checkpoint identity mismatch')
                meta = obj.get('checkpoint_meta',{})
                require(meta.get('tag') == 'matched_step' and meta.get('step') == 2000 and
                        hash_valid(meta.get('batch_sha256')), 'Not a matched-2000-update checkpoint')
                batch_hashes[kind,seed] = meta['batch_sha256']
                require(finite(obj.get('elapsed')) and obj['elapsed'] >= 0, 'Invalid per-checkpoint evaluation runtime')
                runtimes.append(dict(kind=kind,seed=seed,evaluation_wall_seconds=obj['elapsed'],
                    cap_seconds=1200,cap_overshoot_seconds=max(0.,obj['elapsed']-1200)))
                raw_statuses = []
                for rawrow in obj.get('rows',[]):
                    key = (rawrow.get('phase'),rawrow.get('loops'))
                    require(key[0] in PHASES and integer(key[1]) and key[1] in PHASES[key[0]] and key not in lookup,
                            'Duplicate or unexpected count/phase row')
                    loops = key[1]
                    require(rawrow.get('trained_loops') == 2 and rawrow.get('applied_layers') == 16+20*loops and
                            rawrow.get('recurrent_applications') == 20*loops and rawrow.get('unique_parameters') == 764304 and
                            rawrow.get('invoked_module_parameters') == (353424 if loops == 0 else 764304),
                            'Invalid loop/parameter provenance')
                    require(set(rawrow.get('datasets',{})) <= set(DOMAINS), 'Unexpected cell domain')
                    lookup[key] = rawrow
                    raw_statuses.extend(v.get('status','malformed_cell') for v in rawrow['datasets'].values())
                source_status_by_run[kind,seed] = Counter(raw_statuses)
            else:
                runtimes.append(dict(kind=kind,seed=seed,evaluation_wall_seconds=None,
                    cap_seconds=1200,cap_overshoot_seconds=None))
            for phase,counts in PHASES.items():
                for loops in counts:
                    for domain in DOMAINS:
                        cell = lookup.get((phase,loops),{}).get('datasets',{}).get(domain,
                            {'status':'missing_cell' if path.exists() else 'missing_file'})
                        raw_status = cell.get('status','malformed_cell')
                        require(isinstance(raw_status,str), 'Malformed cell status')
                        stat = raw_status if raw_status in STATUSES else 'unrecognized_status'
                        n = WINDOWS[phase]
                        evaluated = cell.get('evaluated_windows',0)
                        evaluated = evaluated if integer(evaluated) and 0 <= evaluated <= n else 0
                        if stat == 'completed':
                            valid = all(finite(cell.get(m)) and cell[m] >= 0 for m in METRICS)
                            valid = valid and cell.get('evaluated_windows') == n and 0 <= cell['exact_32_byte_match'] <= cell['byte_accuracy'] <= 1
                            losses, groups = cell.get('per_window_loss_nats'), cell.get('groups')
                            valid = valid and isinstance(losses,list) and len(losses) == n and all(finite(v) and v >= 0 for v in losses)
                            valid = valid and isinstance(groups,list) and len(groups) == n and all(integer(v) and v >= 0 for v in groups) and len(set(groups)) >= 4
                            if valid:
                                valid = close(cell['suffix_bpb'],statistics.mean(losses)/math.log(2))
                                valid = valid and abs(cell['byte_accuracy']*n*32-round(cell['byte_accuracy']*n*32)) < 1e-5
                                valid = valid and abs(cell['exact_32_byte_match']*n-round(cell['exact_32_byte_match']*n)) < 1e-5
                            if not valid:
                                stat = 'invalid_completed_metrics'
                            else:
                                group_key = phase,domain
                                if group_key in groups_reference:
                                    require(groups == groups_reference[group_key], 'Unequal paired document order')
                                else:
                                    groups_reference[group_key] = groups
                        row = dict(phase=phase,domain=domain,kind=kind,seed=seed,loops=loops,trained_loops=2,
                            loop_distribution='trained_count' if loops == 2 else 'extrapolation',unique_parameters=764304,
                            invoked_module_parameters=353424 if loops == 0 else 764304,applied_layers=16+20*loops,
                            recurrent_applications=20*loops,cohort_windows=n,
                            cohort_selection_sha256=cohort_hashes[f'{phase}/{domain}'],
                            status=stat,source_status=raw_status,evaluated_windows=evaluated)
                        row.update({m:cell[m] if stat == 'completed' else None for m in METRICS})
                        rows.append(row)
            # Missing files/cells remain visible. Compare source final ledger when fully available.
            if len(lookup) == 104 and all(set(r['datasets']) == set(DOMAINS) for r in lookup.values()):
                tally = source_status_by_run[kind,seed]
                summary = next(r for r in runs if (r['kind'],r['seed']) == (kind,seed))
                require(summary['completed_cells'] == tally['completed'] and summary['nonfinite_cells'] ==
                    sum(v for k,v in tally.items() if k.startswith('nonfinite')) and summary['missing_or_capped_cells'] ==
                    sum(v for k,v in tally.items() if k != 'completed' and not k.startswith('nonfinite')),
                    'Final run status does not match cell ledger')
    for seed in SEEDS:
        if all((k,seed) in batch_hashes for k in KINDS):
            require(batch_hashes['split',seed] == batch_hashes['whole',seed], 'Unequal paired training data stream identity')
    require(len(rows) == 1248, 'Internal planned-cell count error')
    for name,digest in raw_hashes.items():
        require(sha(src/name) == digest, 'Bounded input changed during reporting')
    original_smallest = original['candidates'][-1]
    provenance = dict(raw_file_sha256=raw_hashes,original_calibration_receipt_sha256=calibration_hashes,
        source_file_sha256=source_hashes,checkpoint_sha256=selection['checkpoint_hashes'],cohort_selection_sha256=cohort_hashes,
        standard_evaluation_receipt_identities=[{key:r[key] for key in ('kind','seed',
            'training_receipt_sha256','standard_result_sha256','standard_evaluation_sha256')} for r in verified],
        report_generator_sha256=sha(Path(__file__)),report_template_sha256=sha(Path(__file__).parent/'BOUNDED_REPORT_TEMPLATE.md'),
        plot_generator_sha256=sha(Path(__file__).parent/'report_sweep.py'),
        identity_verification='Frozen source bytes, original synthetic no-fit receipts and predictions, amendment/selection identities, matched-step checkpoint hashes from receipts, paired training-stream hashes and paired document order. Model tensors and corpus text never loaded; checkpoint bytes and earlier standard-evaluation files are not rehashed by this reporter.',
        budget=dict(original_status='blocked_no_fit',original_heldout_loaded=False,
            original_smallest_candidate_windows=8,original_selected_windows=128,
            original_selected_counts=[0,1,2,4,8,32,100],
            original_smallest_candidate_global_bound_seconds=original_smallest['global_bound_seconds'],
            original_smallest_candidate_max_margined_checkpoint_seconds=1.5*max(original_smallest['predicted_seconds_per_checkpoint'].values()),
            prior_elapsed_debited_seconds=prior,new_setup_debited_seconds=setup,
            amended_predicted_global_bound_seconds=selection['global_bound_seconds'],
            amended_max_margined_checkpoint_seconds=1.5*max(expected_times.values()),
            global_cap_seconds=7200,per_checkpoint_cap_seconds=1200,
            final_new_elapsed_seconds=status['new_elapsed_seconds'],final_combined_elapsed_seconds=status['combined_elapsed_seconds'],
            final_combined_cap_overshoot_seconds=max(0.,status['combined_elapsed_seconds']-7200),
            global_capped=status['global_capped']))
    return rows,runtimes,provenance


def summarize(values):
    return dict(mean=statistics.mean(values) if values else None,
        seed_sd=statistics.stdev(values) if len(values)>1 else None,
        minimum=min(values) if values else None,maximum=max(values) if values else None)


def aggregate(rows):
    result=[]
    for phase,counts in PHASES.items():
        for domain in DOMAINS:
            for kind in KINDS:
                for loops in counts:
                    group=[r for r in rows if (r['phase'],r['domain'],r['kind'],r['loops'])==(phase,domain,kind,loops)]
                    good=[r for r in group if r['status']=='completed']
                    item=dict(phase=phase,domain=domain,kind=kind,loops=loops,completed_seeds=len(good),expected_seeds=3,
                        complete=len(good)==3,status_counts=json.dumps(dict(Counter(r['status'] for r in group)),sort_keys=True))
                    for metric in METRICS:
                        item.update({metric+'_'+key:value for key,value in summarize([r[metric] for r in good]).items()})
                    result.append(item)
    return result


def paired(rows):
    lookup={(r['phase'],r['domain'],r['kind'],r['seed'],r['loops']):r for r in rows}
    pairs,summary=[],[]
    for phase,counts in PHASES.items():
        for domain in DOMAINS:
            for loops in counts:
                current=[]
                for seed in SEEDS:
                    a,b=(lookup[phase,domain,k,seed,loops] for k in KINDS)
                    complete=a['status']==b['status']=='completed'
                    row=dict(phase=phase,domain=domain,loops=loops,seed=seed,complete=complete,
                        split_status=a['status'],whole_status=b['status'])
                    row.update({m+'_difference_split_minus_whole':a[m]-b[m] if complete else None for m in METRICS})
                    pairs.append(row);current.append(row)
                good=[r for r in current if r['complete']]
                item=dict(phase=phase,domain=domain,loops=loops,completed_seed_pairs=len(good),complete=len(good)==3)
                for metric in METRICS:
                    item.update({metric+'_difference_'+key:value for key,value in summarize([r[metric+'_difference_split_minus_whole'] for r in good]).items()})
                summary.append(item)
    return pairs,summary


def generate(src,out,original_dir=None):
    out=out.resolve()
    require(out.is_relative_to(Path(__file__).resolve().parent) and out != Path(__file__).resolve().parent,
            'Output must be a new subdirectory under reporting/')
    require(not out.exists(),'Refusing output overwrite')
    rows,runtimes,provenance=collect(src,original_dir)
    summary=aggregate(rows);pairs,pair_summary=paired(rows)
    require(len(summary)==416,'Internal summary group count error')
    out.mkdir(parents=True,exist_ok=False)
    for name,value in (('cells',rows),('summary',summary),('paired_cells',pairs),
                       ('paired_summary',pair_summary),('checkpoint_runtime',runtimes)):
        dump(out/(name+'.json'),value);csvout(out/(name+'.csv'),value)
    figures=[]
    for phase in PHASES:
        for domain in DOMAINS:
            for metric in ('suffix_bpb','forward_wall_seconds'):
                filename=f'{phase}_{domain}_{metric}.svg'
                plot(out/filename,rows,phase,domain,metric);figures.append(filename)
    counts=dict(Counter(r['status'] for r in rows))
    dump(out/'provenance.json',provenance)
    template=(Path(__file__).parent/'BOUNDED_REPORT_TEMPLATE.md').read_text()
    replacements={'CELLS':str(len(rows)),'GROUPS':str(len(summary)),'STATUSES':json.dumps(counts,sort_keys=True),
        'COMPLETE_GROUPS':str(sum(s['complete'] for s in summary)),
        'PLOTS':'\n'.join(f'- [{f}]({f})' for f in figures)}
    replacements.update({key.upper():f'{value:.6f}' if finite(value) else str(value)
                         for key,value in provenance['budget'].items()})
    for key,value in replacements.items():template=template.replace('{{'+key+'}}',value)
    require('{{' not in template,'Unfilled report template placeholder')
    (out/'REPORT.md').write_text(template)
    dump(out/'report_status.json',dict(status='report_generated',evaluation_finalized=True,
        planned_cells=1248,summary_groups=416,all_cells_completed=all(r['status']=='completed' for r in rows),
        cell_status_counts=counts,complete_summary_groups=sum(s['complete'] for s in summary),
        publication_performed=False,model_compute_performed=False,bootstrap_performed=False))
    dump(out/'artifact_hashes.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return rows,summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--original-calibration',type=Path)
    args=parser.parse_args()
    try:
        rows,summary=generate(args.input,args.output,args.original_calibration)
        print(json.dumps(dict(status='report_generated',planned_cells=len(rows),summary_groups=len(summary))))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='blocked',reason=str(exc),comparison_generated=False)),file=sys.stderr)
        sys.exit(2)
