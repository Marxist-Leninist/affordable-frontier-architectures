"""Deterministic, standard-library-only report from completed saved metrics.

Never imports training code, loads model objects, opens corpus text, or performs
inference. All six completion receipts are checked before opening evaluation
files. Checkpoints are read only as bytes to verify SHA-256 identity.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys

KINDS = ('split', 'whole')
SEEDS = (1, 2, 3)
TAGS = ('matched_step', 'cpu_300', 'cpu_900')
DOMAINS = ('test', 'fineweb', 'validation')
WINDOWS = {'validation': 256, 'test': 10059, 'fineweb': 2134}
LABELS = {'matched_step': 'Matched 2,000 updates', 'cpu_300': '300 training CPU seconds',
          'cpu_900': '900 training CPU seconds', 'test': 'WikiText-2 test',
          'fineweb': 'FineWeb cohort', 'validation': 'WikiText-2 validation'}
METRICS = ('bpb', 'accuracy', 'exact_match')
METRIC_LABELS = {'bpb': 'Suffix bits/byte', 'accuracy': 'Byte accuracy',
                 'exact_match': 'Exact 32-byte match'}
MANIFEST_FIELDS = {'eligible_windows_before_limit', 'evaluated_windows', 'documents',
    'excluded_exact128_train_or_duplicate', 'target32_matches_training_all_windows',
    'selection_sha256', 'selection', 'source_dataset', 'revision', 'source_rows',
    'source_response_sha256', 'license', 'new_to_this_nat_comparison'}
REQUIRED_SOURCES = {'model.py', 'data_loop.py', 'train.py', 'protocol.json',
                    'summarize.py', 'test_correctness.py', 'run_all.py', 'sweep.py',
                    'sweep_protocol.json', 'control.py', 'runtime_plan.py', 'test_runtime_plan.py'}
EXPECTED_PROTOCOL = dict(seeds=[1, 2, 3], topologies=['split', 'whole'], width=48,
    hidden=117, unique_blocks=36, loops=2, effective_blocks=56, expected_parameters=764304,
    steps=2000, batch=8, lr=0.0005, time_checkpoints_cpu_seconds=[300, 900],
    threads=1, max_concurrent_jobs=1, core=1, validation_windows=256,
    fixed_lr_no_tuning=True, reinjection=False, input_blocks=8,
    recurrent_unique_blocks=20, output_blocks=8, recurrent_effective_blocks=40,
    zero_loop_invoked_parameters=353424)


class ReportError(ValueError):
    """An incomplete or inconsistent experiment must not be summarized."""


def require(ok, message):
    if not ok:
        raise ReportError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def valid_hash(v):
    return isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def integer(v):
    return isinstance(v, int) and not isinstance(v, bool)


def read(path):
    def bad_constant(value):
        raise ReportError(f'Nonfinite JSON constant in {path.name}: {value}')
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key in {path.name}: {key}')
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(), parse_constant=bad_constant, object_pairs_hook=unique_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f'Cannot read {path}: {exc}') from exc


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def csvout(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def schedule(kind):
    a, b = list(range(10)), list(range(10, 20))
    return a * 2 + b * 2 if kind == 'split' else (a + b) * 2


def collect(root):
    """Fail closed before returning even one score; do not write output here."""
    root = root.resolve()
    results = {}
    # This first gate intentionally cannot open evaluation.json or score data.
    for seed in SEEDS:
        for kind in KINDS:
            name = f'{kind}_s{seed}'
            result = read(root / 'runs' / name / 'result.json')
            require(result.get('status') == 'completed',
                    f'Full comparison unavailable: {name} result status is {result.get("status")!r}; require completed for all six runs')
            require(result.get('evaluated_checkpoints') == 3,
                    f'{name}: require three evaluated checkpoints')
            results[kind, seed] = result

    protocol = read(root / 'protocol.json')
    for key, expected in EXPECTED_PROTOCOL.items():
        require(protocol.get(key) == expected, f'Unexpected protocol {key}')
    frozen = read(root / 'source_hashes.json')
    require(isinstance(frozen, dict) and REQUIRED_SOURCES <= set(frozen), 'Incomplete frozen source manifest')
    file_hashes = {}
    for name, digest in sorted(frozen.items()):
        require(valid_hash(digest) and Path(name).name == name, 'Invalid frozen source identity')
        require(sha(root / name) == digest, f'Frozen source changed: {name}')
        file_hashes[name] = digest
    for name in ('source_hashes.json', 'completed_pair_checks.json'):
        file_hashes[name] = sha(root / name)
    audit = read(root / 'completed_pair_checks.json')
    require(audit.get('all_six_fits_complete') is True, 'Training audit is not complete')
    for seed in SEEDS:
        checks = audit.get('training_only_checks', {}).get(str(seed), {})
        require(all(checks.get(k) is True for k in ('identical_initial_parameters',
                'matched_step_data_hash_identical', 'checkpoint_hashes_verified')),
                f'Training pair audit incomplete for seed {seed}')

    training, evaluations, checkpoints, initial_hashes = {}, {}, {}, {}
    reference_manifests, reference_groups = {}, {}
    components, timing, run_timing = [], [], []
    # Verify training/checkpoint identities for all 18 checkpoints before scores.
    for seed in SEEDS:
        for kind in KINDS:
            name = f'{kind}_s{seed}'
            directory = root / 'runs' / name
            tr = read(directory / 'training.json')
            result = results[kind, seed]
            require(tr.get('status') == 'training_complete', f'{name}: training incomplete')
            require(all(result.get(k) == v for k, v in tr.items() if k != 'status'),
                    f'{name}: completion receipt differs from training identity')
            require(tr.get('kind') == kind and tr.get('seed') == seed, f'{name}: wrong run identity')
            require(tr.get('source_hashes') == frozen and tr.get('protocol') == protocol,
                    f'{name}: source/protocol identity mismatch')
            require(tr.get('parameters') == 764304 and tr.get('schedule') == schedule(kind)
                    and tr.get('input_blocks') == 8 and tr.get('output_blocks') == 8
                    and tr.get('applied_layers') == 56 and tr.get('recurrent_applications') == 40,
                    f'{name}: topology identity mismatch')
            require(valid_hash(tr.get('initial_hash')), f'{name}: invalid initialization hash')
            require(integer(tr.get('steps')) and tr['steps'] >= 2000 and
                    finite(tr.get('cpu')) and tr['cpu'] >= 900 and
                    finite(tr.get('wall')) and tr['wall'] > 0,
                    f'{name}: invalid completed training timing')
            require(finite(result.get('evaluation_wall_seconds')) and result['evaluation_wall_seconds'] > 0,
                    f'{name}: invalid evaluation wall time')
            ck_list = tr.get('checkpoints', [])
            require(len(ck_list) == 3 and {c.get('tag') for c in ck_list} == set(TAGS),
                    f'{name}: require exactly three unique checkpoints')
            by_tag = {c['tag']: c for c in ck_list}
            for tag in TAGS:
                c = by_tag[tag]
                require(integer(c.get('step')) and 0 < c['step'] <= tr['steps'] and
                        finite(c.get('cpu')) and 0 < c['cpu'] <= tr['cpu'] and
                        finite(c.get('wall')) and 0 < c['wall'] <= tr['wall'], f'{name}/{tag}: invalid checkpoint timing')
                require(valid_hash(c.get('batch_sha256')) and valid_hash(c.get('checkpoint_sha256')),
                        f'{name}/{tag}: invalid checkpoint/data hash')
                if tag == 'matched_step':
                    require(c['step'] == 2000, f'{name}: wrong matched update')
                else:
                    require(c['cpu'] >= int(tag[4:]), f'{name}/{tag}: CPU threshold not reached')
                ck_path = directory / (tag + '.pt')
                require(ck_path.is_file(), f'{name}/{tag}: checkpoint file missing')
                require(sha(ck_path) == c['checkpoint_sha256'], f'{name}/{tag}: checkpoint bytes changed')
                checkpoints[f'{name}/{tag}'] = c['checkpoint_sha256']
            for left, right in zip(sorted(ck_list, key=lambda c: c['step']), sorted(ck_list, key=lambda c: c['step'])[1:]):
                require(left['cpu'] <= right['cpu'] and left['wall'] <= right['wall'], f'{name}: nonmonotonic checkpoint timing')
            training[kind, seed] = tr
            initial_hashes[name] = tr['initial_hash']
            for f in ('training.json', 'result.json'):
                file_hashes[f'runs/{name}/{f}'] = sha(directory / f)
            run_timing.append(dict(kind=kind, seed=seed, final_steps=tr['steps'],
                final_training_update_cpu_seconds=tr['cpu'], final_training_update_wall_seconds=tr['wall'],
                evaluation_wall_seconds=result['evaluation_wall_seconds'], peak_rss_kib=tr.get('peak_rss_kib')))
        split, whole = (training[kind, seed] for kind in KINDS)
        require(split['initial_hash'] == whole['initial_hash'], f'Seed {seed}: unequal initialization')
        # Compare every checkpoint with equal update count, not only matched_step.
        for a in split['checkpoints']:
            for b in whole['checkpoints']:
                if a['step'] == b['step']:
                    require(a['batch_sha256'] == b['batch_sha256'], f'Seed {seed}: unequal equal-step data hashes')
    if 'total_training_update_cpu_seconds' in audit:
        require(math.isclose(sum(t['cpu'] for t in training.values()), audit['total_training_update_cpu_seconds'], rel_tol=1e-10),
                'Training CPU total differs from audit')

    for seed in SEEDS:
        for kind in KINDS:
            name = f'{kind}_s{seed}'
            path = root / 'runs' / name / 'evaluation.json'
            evaluation = read(path)
            require(isinstance(evaluation, list) and len(evaluation) == 3 and
                    {c.get('tag') for c in evaluation} == set(TAGS),
                    f'{name}: all three evaluated checkpoints required')
            ev = {c['tag']: c for c in evaluation}
            by_tag = {c['tag']: c for c in training[kind, seed]['checkpoints']}
            for tag in TAGS:
                record, ck = ev[tag], by_tag[tag]
                require(all(record.get(k) == v for k, v in ck.items()), f'{name}/{tag}: evaluation checkpoint identity mismatch')
                require(set(record.get('datasets', {})) == set(DOMAINS), f'{name}/{tag}: missing/unexpected evaluation domains')
                threshold = int(tag[4:]) if tag.startswith('cpu_') else None
                timing.append(dict(kind=kind, seed=seed, tag=tag, step=ck['step'],
                    supervised_suffix_byte_presentations=ck['step'] * 8 * 32,
                    training_update_cpu_seconds=ck['cpu'], training_update_wall_seconds=ck['wall'],
                    training_cpu_target_seconds=threshold,
                    cpu_overshoot_seconds=ck['cpu']-threshold if threshold else None,
                    updates_per_training_cpu_second=ck['step']/ck['cpu'],
                    batch_sha256=ck['batch_sha256'], checkpoint_sha256=ck['checkpoint_sha256']))
                for domain in DOMAINS:
                    dataset = record['datasets'][domain]
                    manifest, m = dataset['manifest'], dataset['metrics']
                    n = WINDOWS[domain]
                    require(manifest.get('evaluated_windows') == n and valid_hash(manifest.get('selection_sha256')),
                            f'{name}/{tag}/{domain}: wrong selection identity or window count')
                    groups, losses = m.get('groups'), m.get('per_window_loss')
                    require(isinstance(groups, list) and len(groups) == n and all(integer(g) and g >= 0 for g in groups),
                            f'{name}/{tag}/{domain}: invalid document groups')
                    require(manifest.get('documents') == len(set(groups)) and len(set(groups)) >= 4,
                            f'{name}/{tag}/{domain}: invalid document count')
                    require(isinstance(losses, list) and len(losses) == n and all(finite(v) and v >= 0 for v in losses),
                            f'{name}/{tag}/{domain}: invalid per-window losses')
                    require(all(finite(m.get(k)) and m[k] >= 0 for k in ('loss_nats', *METRICS))
                            and m['accuracy'] <= 1 and m['exact_match'] <= m['accuracy'],
                            f'{name}/{tag}/{domain}: invalid metrics')
                    require(math.isclose(statistics.mean(losses), m['loss_nats'], rel_tol=1e-8, abs_tol=1e-10)
                            and math.isclose(m['bpb'], m['loss_nats']/math.log(2), rel_tol=1e-8, abs_tol=1e-10),
                            f'{name}/{tag}/{domain}: aggregate loss does not match saved window losses')
                    # Exact matches and correct bytes must be feasible integer totals.
                    for value, denominator in ((m['accuracy'], n * 32), (m['exact_match'], n)):
                        require(abs(value * denominator - round(value * denominator)) < 1e-5,
                                f'{name}/{tag}/{domain}: impossible accuracy denominator')
                    if domain not in reference_manifests:
                        reference_manifests[domain], reference_groups[domain] = manifest, groups
                    require(manifest == reference_manifests[domain] and groups == reference_groups[domain],
                            f'{name}/{tag}/{domain}: unequal evaluation manifest or document order')
                    components.append(dict(tag=tag, domain=domain, kind=kind, seed=seed,
                        windows=n, documents=manifest['documents'], selection_sha256=manifest['selection_sha256'],
                        bpb=m['bpb'], accuracy=m['accuracy'], exact_match=m['exact_match'], loss_nats=m['loss_nats']))
            file_hashes[f'runs/{name}/evaluation.json'] = sha(path)
            evaluations[kind, seed] = ev

    # Detect any concurrent changes to receipts/sources, not just malformed files.
    for name, digest in file_hashes.items():
        require(sha(root / name) == digest, f'Input changed while reporting: {name}')
    provenance = dict(frozen_source_sha256=frozen, input_file_sha256=file_hashes,
        checkpoint_sha256=checkpoints, initialization_sha256=initial_hashes,
        evaluation_manifests={domain: {key: value for key, value in manifest.items() if key in MANIFEST_FIELDS}
                              for domain, manifest in reference_manifests.items()},
        verification='All six completion receipts, 18 checkpoint byte hashes, frozen source files, paired initialization and equal-step data hashes, and identical evaluation selections/document order verified. No models or corpus text loaded.',
        limitations='Initialization and batch-stream hashes are checked against saved receipts, not reconstructed by training; evaluation selection hashes are compared, not recomputed from raw corpus text.',
        generator_sha256=sha(Path(__file__)))
    for filename in ('test_report_topology.py', 'TOPOLOGY_REPORTING.md'):
        if (Path(__file__).parent / filename).exists():
            provenance[filename + '_sha256'] = sha(Path(__file__).parent / filename)
    return components, timing, run_timing, provenance


def stats(values):
    require(len(values) == 3, 'Statistics require all three paired seeds')
    return dict(mean=statistics.mean(values), seed_sd=statistics.stdev(values),
                minimum=min(values), maximum=max(values))


def aggregate(components, timing):
    lookup = {(r['tag'], r['domain'], r['kind'], r['seed']): r for r in components}
    summaries, pairs = [], []
    for tag in TAGS:
        for domain in DOMAINS:
            for metric in METRICS:
                split = [lookup[tag, domain, 'split', s][metric] for s in SEEDS]
                whole = [lookup[tag, domain, 'whole', s][metric] for s in SEEDS]
                diff = [a-b for a, b in zip(split, whole)]
                item = dict(tag=tag, domain=domain, metric=metric, paired_seeds=3,
                            difference_direction='split minus whole', favorable_difference='negative' if metric == 'bpb' else 'positive')
                for name, values in (('split', split), ('whole', whole), ('difference', diff)):
                    item.update({name + '_' + k: v for k, v in stats(values).items()})
                summaries.append(item)
                pairs.extend(dict(tag=tag, domain=domain, metric=metric, seed=s,
                    split=a, whole=b, difference_split_minus_whole=a-b)
                    for s, a, b in zip(SEEDS, split, whole))
    tlookup = {(r['tag'], r['kind'], r['seed']): r for r in timing}
    timing_summary = []
    for tag in TAGS:
        metrics = ['step', 'training_update_cpu_seconds', 'training_update_wall_seconds']
        if tag != 'matched_step':
            metrics.append('cpu_overshoot_seconds')
        for metric in metrics:
            item = dict(tag=tag, metric=metric, paired_seeds=3)
            a = [tlookup[tag, 'split', s][metric] for s in SEEDS]
            b = [tlookup[tag, 'whole', s][metric] for s in SEEDS]
            for name, values in (('split', a), ('whole', b), ('difference', [x-y for x, y in zip(a, b)])):
                item.update({name + '_' + k: v for k, v in stats(values).items()})
            timing_summary.append(item)
    return summaries, pairs, timing_summary


def svg_start(title, subtitle, height=430):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="{height}" viewBox="0 0 1140 {height}" role="img">',
        f'<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/><g font-family="Arial,sans-serif" fill="#17212b">',
        f'<text x="30" y="31" font-size="21" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="30" y="56" font-size="12">{html.escape(subtitle)}</text>']


def paired_plot(path, rows, tag, metric):
    """Each small multiple is a domain; no cross-domain composite."""
    subtitle = 'Three paired seeds: s1 circle, s2 square, s3 triangle. Lines connect the same seed; no confidence intervals.'
    parts = svg_start(f'{LABELS[tag]} | {METRIC_LABELS[metric]}', subtitle)
    for i, domain in enumerate(DOMAINS):
        group = [r for r in rows if r['tag'] == tag and r['domain'] == domain]
        values = [r[metric] for r in group]
        low, high = min(values), max(values)
        pad = (high-low) * .18 or (.02 if metric != 'bpb' else .1)
        low = max(0, low-pad)
        high = high+pad if metric == 'bpb' else min(1, high+pad)
        if high == low:
            high = low + .01
        tick_step = (high-low)/4*(1 if metric == 'bpb' else 100)
        precision = min(8, max(3 if metric == 'bpb' else 2, 1-math.floor(math.log10(tick_step))))
        left = 72 + i*370
        y = lambda v: 330 - 220*(v-low)/(high-low)
        parts.append(f'<text x="{left+112}" y="88" text-anchor="middle" font-size="15">{html.escape(LABELS[domain])}</text>')
        for tick in range(5):
            value = low+(high-low)*tick/4
            yy = y(value)
            label = f'{value:.{precision}f}' if metric == 'bpb' else f'{value*100:.{precision}f}%'
            parts.extend([f'<path d="M{left} {yy:.3f}h245" stroke="#e5e9ee"/>',
                          f'<text x="{left-7}" y="{yy+4:.3f}" text-anchor="end" font-size="11">{label}</text>'])
        for seed in SEEDS:
            a, b = ([r for r in group if r['kind'] == kind and r['seed'] == seed][0][metric] for kind in KINDS)
            xx = (left+55+(seed-2)*9, left+185+(seed-2)*9)
            parts.append(f'<path d="M{xx[0]} {y(a):.3f}L{xx[1]} {y(b):.3f}" stroke="#a9b2be" stroke-width="1.4"/>')
            for x, value, color in zip(xx, (a, b), ('#1167b1', '#bc4e18')):
                yy = y(value)
                tooltip = f'{domain}, seed {seed}: {value:.8g}'
                if seed == 1:
                    mark = f'<circle cx="{x}" cy="{yy:.3f}" r="4" fill="{color}"><title>{tooltip}</title></circle>'
                elif seed == 2:
                    mark = f'<rect x="{x-4}" y="{yy-4:.3f}" width="8" height="8" fill="{color}"><title>{tooltip}</title></rect>'
                else:
                    mark = f'<path d="M{x} {yy-5:.3f}l5 9h-10z" fill="{color}"><title>{tooltip}</title></path>'
                parts.append(mark)
        parts.append(f'<text x="{left+55}" y="355" text-anchor="middle" font-size="13" fill="#1167b1">Split</text>')
        parts.append(f'<text x="{left+185}" y="355" text-anchor="middle" font-size="13" fill="#bc4e18">Whole</text>')
    parts.append('<text x="30" y="392" font-size="12">Independent vertical scales per panel. Validation is descriptive, not used for selection. Percentages are on a 0–100 scale.</text>')
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts)+'\n')


def delta_plot(path, summaries, pairs):
    parts = svg_start('Paired suffix bits/byte differences | split minus whole',
        'Negative favors split. Small dots are seeds 1–3; diamond is mean; whisker is observed seed range, not a confidence interval.', 545)
    for i, domain in enumerate(DOMAINS):
        left = 105 + i*370
        selected = [r for r in summaries if r['domain'] == domain and r['metric'] == 'bpb']
        extent = max(abs(r[k]) for r in selected for k in ('difference_minimum', 'difference_maximum')) * 1.2 or .01
        x = lambda v: left+105+100*v/extent
        parts.append(f'<text x="{left+105}" y="93" text-anchor="middle" font-size="15">{html.escape(LABELS[domain])}</text>')
        parts.append(f'<path d="M{x(0):.3f} 115V407" stroke="#929ba5" stroke-dasharray="4 3"/>')
        for j, tag in enumerate(TAGS):
            yy = 150+j*105
            r = next(r for r in selected if r['tag'] == tag)
            parts.append(f'<text x="{left+105}" y="{yy-21}" text-anchor="middle" font-size="12">{html.escape(LABELS[tag])}</text>')
            parts.append(f'<path d="M{x(r["difference_minimum"]):.3f} {yy}H{x(r["difference_maximum"]):.3f}" stroke="#4c627a" stroke-width="2"/>')
            for p in [p for p in pairs if p['tag'] == tag and p['domain'] == domain and p['metric'] == 'bpb']:
                parts.append(f'<circle cx="{x(p["difference_split_minus_whole"]):.3f}" cy="{yy+(p["seed"]-2)*7}" r="3" fill="#58748e"/>')
            mx = x(r['difference_mean'])
            parts.append(f'<path d="M{mx:.3f} {yy-6}l6 6l-6 6l-6-6z" fill="#111a23"/>')
            parts.append(f'<text x="{left+105}" y="{yy+29}" text-anchor="middle" font-size="11">Mean {r["difference_mean"]:+.5f}; seed SD {r["difference_seed_sd"]:.5f}</text>')
        for value in (-extent, 0, extent):
            parts.append(f'<text x="{x(value):.3f}" y="436" text-anchor="middle" font-size="11">{value:+.4f}</text>')
    parts.append('<text x="30" y="480" font-size="12">Independent horizontal scales per domain. Three seeds describe these fits only; no architecture-population inference.</text>')
    parts.append('<text x="30" y="508" font-size="12">Matched updates and matched training CPU time are separate comparisons. No pooled score or best checkpoint is selected.</text>')
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts)+'\n')


def timing_plot(path, timing):
    parts = svg_start('Training timing and time-budget exposure',
        'All seed pairs shown. Update-only timers exclude data sampling, initialization, checkpoint writes and evaluation.')
    for i, tag in enumerate(TAGS):
        metric = 'training_update_cpu_seconds' if tag == 'matched_step' else 'step'
        rows = [r for r in timing if r['tag'] == tag]
        values = [r[metric] for r in rows]
        low, high = min(values), max(values)
        pad = (high-low)*.25 or 1
        low, high = max(0, low-pad), high+pad
        left = 80+i*370
        y = lambda value: 325-210*(value-low)/(high-low)
        parts.append(f'<text x="{left+110}" y="87" text-anchor="middle" font-size="14">{html.escape(LABELS[tag])}</text>')
        for j in range(5):
            v = low+(high-low)*j/4
            yy = y(v)
            parts.extend([f'<path d="M{left} {yy:.3f}h230" stroke="#e5e9ee"/>',
                          f'<text x="{left-7}" y="{yy+4:.3f}" text-anchor="end" font-size="11">{v:.1f}</text>'])
        for seed in SEEDS:
            a, b = ([r for r in rows if r['kind'] == kind and r['seed'] == seed][0][metric] for kind in KINDS)
            xa, xb = left+55+(seed-2)*8, left+180+(seed-2)*8
            parts.append(f'<path d="M{xa} {y(a):.3f}L{xb} {y(b):.3f}" stroke="#adb6c1"/>')
            for xx, v, color in ((xa, a, '#1167b1'), (xb, b, '#bc4e18')):
                parts.append(f'<circle cx="{xx}" cy="{y(v):.3f}" r="4" fill="{color}"><title>Seed {seed}: {v:.8g}</title></circle>')
        parts.extend([f'<text x="{left+55}" y="349" text-anchor="middle" font-size="12">Split</text>',
                      f'<text x="{left+180}" y="349" text-anchor="middle" font-size="12">Whole</text>',
                      f'<text x="{left+110}" y="375" text-anchor="middle" font-size="12">{"Training CPU seconds" if tag == "matched_step" else "Completed updates"}</text>'])
    parts.append('<text x="30" y="408" font-size="12">Time checkpoints use the first whole update crossing the target; actual CPU overshoots and wall timing are tabulated separately.</text>')
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts)+'\n')


def architecture_plot(path):
    parts = svg_start('Two recurrent orders, identical trainable blocks',
        'A and B are distinct ordered ten-block groups. Repeated labels reuse the same weights; I and O each contain eight unique blocks.', 370)
    for row, (label, groups) in enumerate((('Split: I A A B B O', ('I', 'A', 'A', 'B', 'B', 'O')),
                                          ('Whole: I A B A B O', ('I', 'A', 'B', 'A', 'B', 'O')))):
        yy = 112+row*102
        parts.append(f'<text x="30" y="{yy+22}" font-size="17">{label}</text>')
        for col, group in enumerate(groups):
            x = 310+col*130
            color = {'I': '#e6ebf2', 'O': '#e6ebf2', 'A': '#b9d7f0', 'B': '#f1cfbb'}[group]
            parts.append(f'<rect x="{x}" y="{yy-7}" width="108" height="48" rx="7" fill="{color}"/>')
            parts.append(f'<text x="{x+54}" y="{yy+23}" text-anchor="middle" font-size="20">{group}</text>')
            if col < 5:
                parts.append(f'<text x="{x+113}" y="{yy+22}" font-size="18">→</text>')
    parts.append('<text x="30" y="312" font-size="14">Training r=2: 36 unique blocks, 56 applications, 764,304 parameters. General form: I A^r B^r O versus I (AB)^r O.</text>')
    parts.append('<text x="30" y="342" font-size="13">r=0 bypasses only the middle: all 16 outer attention blocks remain. The separate 0–100 sweep does not retrain models.</text>')
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts)+'\n')


METHODS = [
    ('Question and architecture', 'This is a controlled recurrent-order NAT pilot, inspired by the layer-count motif in NVIDIA Dory. It does not use a Dory checkpoint and is not a Dory architecture replication. The unique blocks are 8 input + 20 middle + 8 output. A is the first ordered ten-block middle group, B the second. Split executes I A^r B^r O; whole executes I (AB)^r O. At the training count r=2, these are I A A B B O versus I A B A B O: 56 block applications and 764,304 unique trainable parameters in either arm. Width 48, four heads, FFN hidden width 117, RMSNorm and RoPE are used. There is no input reinjection, activation checkpointing or truncated backward. nGPT, Dory tokenizer/weights, hybrid attention, pretraining and KV caching are not reproduced.'),
    ('Objective and target isolation', 'Each example provides a 96-byte prefix plus 32 MASK slots. A single network call predicts all 32 suffix bytes in parallel using bidirectional attention; the loss is suffix-only cross entropy. Gold target bytes never enter the model inputs and there is no target-token feedback between layers or loops. Internal recurrence is latent computation, not autoregressive decoding. Parameters are counted once despite repeated middle-layer applications.'),
    ('Training and two estimands', 'Three paired seeds (1, 2, 3) have identical within-seed initialization and deterministic batch streams across arms. Training uses batch 8, AdamW, fixed base learning rate 0.0005, 100-step linear warmup then constant rate, matrix weight decay 0.01, no norm-vector decay and gradient clipping at 1. The learning rate is a common untuned pilot choice. Matched 2,000 updates means 512,000 supervised suffix-byte presentations per run and matched nominal block-operation counts. The separate 300- and 900-training-CPU-second comparisons stop at the first full update crossing each target and can expose the arms to different stream-prefix lengths. CPU overshoot and actual update counts are reported. No checkpoint was selected using validation or test outcomes.'),
    ('Evaluation and reused cohorts', 'Each of the 18 checkpoints is evaluated at r=2 on 256 validation windows, all 10,059 eligible nonoverlapping article-contained WikiText-2 test windows, and 2,134 windows from the existing 100-document FineWeb source cohort. Exact 128-byte training-window matches and duplicate windows are excluded by the pinned loader; matching 32-byte targets are counted in the manifests, not excluded. These cohorts were reused from earlier studies and are not newly held-out evidence for the wider research program. The saved FineWeb new_to_this_nat_comparison field is loader metadata, not a claim of a newly collected cohort. Test and FineWeb are reported separately; validation is descriptive only. Metrics weight equal-length windows/bytes, not documents equally. No raw corpus text is included in this report.'),
    ('Uncertainty and limits', 'Every component and paired seed difference is shown. Summary means, sample SD (denominator n−1), and observed minimum–maximum describe these three fitted seed pairs only. The reported uncertainty is not an architecture-population confidence interval, hypothesis test or evidence of convergence. No document bootstrap is performed. No weighted or invented composite score is used, and no best checkpoint/count is selected. Reused cohorts, only three seeds, an untuned common learning rate, small byte-level models and a narrow completion task limit interpretation. This pilot cannot establish general intelligence, 600B-model equivalence, GPU speed/efficiency, large-model scaling or NVIDIA Dory performance.'),
    ('Timing scope', 'Training CPU and training wall values sum timed updates only: zero-grad, forward/loss, backward, clipping and AdamW. Sampling, initialization, checkpoint serialization and evaluation are excluded. Training wall is not total end-to-end process wall. Evaluation wall is recorded separately and includes loading/setting up the held-out cohorts and checkpoint evaluations after model construction. End-to-end training process wall is not recorded in these receipts. Sequential one-core/one-thread CPU FP32 measurements are descriptive, not repeated microbenchmarks, FLOP measurements or GPU forecasts. Runs continue until both 2,000 updates and 900 CPU seconds are met, so final-run timing is not the matched-update budget.'),
    ('Separate inference-count sweep', 'The prepared sweep evaluates every integer r=0…100 using the six matched-update checkpoints trained only at r=2. It is inference-count extrapolation, not retraining at 101 counts. Total block applications are 16+20r; at r=0 the 16 outer blocks still execute and can condition on the prefix, with 353,424 invoked module parameters while 764,304 remain stored. This standard-checkpoint report does not load, summarize or claim completion of the sweep. Use report_sweep.py and its separate runtime-selected cohort protocol/report for that experiment.'),
]


def table(headers, rows):
    md = '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join('---' for _ in headers) + ' |\n'
    md += ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows)
    web = '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>'+html.escape(x)+'</th>' for x in headers) + '</tr></thead><tbody>'
    web += ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    return md, web + '</tbody></table></div>'


def report(out, components, timing, run_timing, summaries, pairs, provenance, figures):
    md = ['# Two local loops versus one long loop\n', '**Complete standard evaluation:** all six completed runs and all 18 evaluated checkpoints passed identity checks. Training/evaluation used r=2.\n']
    web = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Loop topology comparison</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:35px auto;padding:0 22px;color:#17212b}h1,h2{line-height:1.2}h2{margin-top:40px}img{max-width:100%;height:auto;border:1px solid #e3e7ec}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px 10px;border-bottom:1px solid #d9dfe5;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef3f7}.table-wrap{overflow-x:auto}a{color:#1167b1}.note{padding:14px;background:#eef3f7}</style></head><body><h1>Two local loops versus one long loop</h1><p class="note"><strong>Complete standard evaluation:</strong> all six completed runs and all 18 evaluated checkpoints passed identity checks. Training/evaluation used r=2.</p>']
    for title, text in METHODS:
        md.extend(['## '+title+'\n', text+'\n'])
        web.extend(['<h2>'+html.escape(title)+'</h2><p>'+html.escape(text)+'</p>'])
        if title == 'Question and architecture':
            md.append('![Topology diagram](architecture.svg)\n')
            web.append('<img src="architecture.svg" alt="Same blocks, split I A A B B O and whole I A B A B O">')
    def section(title, headers, rows):
        a, b = table(headers, rows)
        md.extend(['## '+title+'\n', a+'\n'])
        web.extend(['<h2>'+html.escape(title)+'</h2>', b])
    for tag in TAGS:
        rows = []
        for r in [r for r in summaries if r['tag'] == tag]:
            scale = 1 if r['metric'] == 'bpb' else 100
            units = 'bpb' if r['metric'] == 'bpb' else '%; delta pp'
            f = lambda name: f'{r[name]*scale:.5f}'
            rows.append([LABELS[r['domain']], METRIC_LABELS[r['metric']]+' ('+units+')',
                f('split_mean')+' ± '+f('split_seed_sd'), f('whole_mean')+' ± '+f('whole_seed_sd'),
                f('difference_mean')+' ± '+f('difference_seed_sd'),
                '['+f('difference_minimum')+', '+f('difference_maximum')+']'])
        section(LABELS[tag]+': component summaries',
                ['Domain', 'Metric', 'Split mean ± SD', 'Whole mean ± SD', 'Split−whole mean ± SD', 'Paired seed range'], rows)
        for metric in METRICS:
            filename = f'{tag}_{metric}.svg'
            md.append(f'![{LABELS[tag]} {METRIC_LABELS[metric]}]({filename})\n')
            web.append(f'<img src="{filename}" alt="{html.escape(LABELS[tag]+": "+METRIC_LABELS[metric])}">')
        rows = [[LABELS[r['domain']], r['kind'], r['seed'], f'{r["bpb"]:.6f}',
                 f'{100*r["accuracy"]:.5f}', f'{100*r["exact_match"]:.5f}']
                for r in sorted((r for r in components if r['tag'] == tag), key=lambda r:(DOMAINS.index(r['domain']),r['seed'],KINDS.index(r['kind'])))]
        section(LABELS[tag]+': all seeds', ['Domain', 'Arm', 'Seed', 'Suffix bpb', 'Byte accuracy %', 'Exact match %'], rows)
        paired = {(r['domain'], r['seed'], r['metric']): r['difference_split_minus_whole']
                  for r in pairs if r['tag'] == tag}
        rows = [[LABELS[domain], seed, f'{paired[domain,seed,"bpb"]:+.6f}',
                 f'{100*paired[domain,seed,"accuracy"]:+.5f}',
                 f'{100*paired[domain,seed,"exact_match"]:+.5f}']
                for domain in DOMAINS for seed in SEEDS]
        section(LABELS[tag]+': paired seed differences (split minus whole)',
                ['Domain', 'Seed', 'Suffix bpb difference', 'Byte accuracy difference, pp', 'Exact match difference, pp'], rows)
    for filename in ('paired_bpb_differences.svg', 'training_timing.svg'):
        md.append(f'![{filename}]({filename})\n')
        web.append(f'<img src="{filename}" alt="{filename}">')
    rows = [[LABELS[r['tag']], r['kind'], r['seed'], r['step'], f'{r["training_update_cpu_seconds"]:.6f}',
             f'{r["training_update_wall_seconds"]:.6f}',
             'N/A' if r['cpu_overshoot_seconds'] is None else f'{r["cpu_overshoot_seconds"]:.6f}',
             r['supervised_suffix_byte_presentations']] for r in sorted(timing, key=lambda r:(TAGS.index(r['tag']),r['seed'],KINDS.index(r['kind'])))]
    section('Training checkpoint timing and exposure', ['Checkpoint', 'Arm', 'Seed', 'Updates', 'Update CPU s', 'Update wall s', 'CPU overshoot s', 'Suffix-byte presentations'], rows)
    rows = [[r['kind'], r['seed'], r['final_steps'], f'{r["final_training_update_cpu_seconds"]:.6f}',
             f'{r["final_training_update_wall_seconds"]:.6f}', f'{r["evaluation_wall_seconds"]:.6f}', r['peak_rss_kib']] for r in run_timing]
    section('Full-run receipts, separate from checkpoint budgets', ['Arm', 'Seed', 'Final updates', 'Update CPU s', 'Update wall s', 'Evaluation wall s', 'Peak RSS KiB'], rows)
    rows = [[LABELS[d], m['evaluated_windows'], m['documents'], m['selection_sha256']]
            for d, m in provenance['evaluation_manifests'].items()]
    section('Evaluation selection identity', ['Domain', 'Windows', 'Documents', 'Selection SHA-256'], rows)
    note = provenance['verification']+' '+provenance['limitations']
    md.extend(['## Reproducibility and files\n', note+'\n', 'Generator SHA-256: '+provenance['generator_sha256']+'\n',
               'Machine-readable tables: components.csv/json, paired_differences.csv/json, summary.csv/json, checkpoint_timing.csv/json, timing_summary.csv/json and run_timing.csv/json. provenance.json records source/input/checkpoint identities; artifact_hashes.json records generated files. No corpus text or model tensors are embedded.\n',
               'Architectural motivation: https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/blob/main/README.md (previously inspected November 11, 2026; displayed revision b366b87).\n'])
    web.append('<h2>Reproducibility and files</h2><p>'+html.escape(note)+'</p><p>Generator SHA-256: '+provenance['generator_sha256']+'</p><ul>')
    for name in ('components', 'paired_differences', 'summary', 'checkpoint_timing', 'timing_summary', 'run_timing'):
        web.append(f'<li><a href="{name}.csv">{name}.csv</a> / <a href="{name}.json">JSON</a></li>')
    web.append('<li><a href="provenance.json">Provenance and identities</a></li><li><a href="artifact_hashes.json">Artifact hashes</a></li></ul><p>Architectural motivation: <a href="https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/blob/main/README.md">NVIDIA official Dory model card</a>, previously inspected November 11, 2026, displayed revision b366b87. No NVIDIA endorsement or license grant is implied.</p></body></html>')
    (out/'REPORT.md').write_text('\n'.join(md)+'\n')
    (out/'REPORT.html').write_text('\n'.join(web)+'\n')


def generate(root, out):
    root, out = root.resolve(), out.resolve()
    reporting_root = Path(__file__).resolve().parent
    require(out.is_relative_to(reporting_root) and out != reporting_root,
            'Output must be a new subdirectory under reporting; never write into frozen experiment/run paths')
    require(not out.exists(), 'Refusing to overwrite an existing report directory')
    components, timing, run_timing, provenance = collect(root)
    summaries, pairs, timing_summary = aggregate(components, timing)
    # No directory or report is created until every validation has succeeded.
    out.mkdir(parents=True, exist_ok=False)
    for filename, data in (('components', components), ('summary', summaries),
            ('paired_differences', pairs), ('checkpoint_timing', timing),
            ('timing_summary', timing_summary), ('run_timing', run_timing)):
        dump(out/(filename+'.json'), data)
        csvout(out/(filename+'.csv'), data)
    figures = []
    for tag in TAGS:
        for metric in METRICS:
            filename = f'{tag}_{metric}.svg'
            paired_plot(out/filename, components, tag, metric)
            figures.append(filename)
    architecture_plot(out/'architecture.svg')
    delta_plot(out/'paired_bpb_differences.svg', summaries, pairs)
    timing_plot(out/'training_timing.svg', timing)
    figures.extend(('architecture.svg', 'paired_bpb_differences.svg', 'training_timing.svg'))
    dump(out/'provenance.json', provenance)
    report(out, components, timing, run_timing, summaries, pairs, provenance, figures)
    dump(out/'report_status.json', dict(status='completed', runs=6, evaluated_checkpoints=18,
        component_rows=len(components), paired_metric_rows=len(pairs), figures=figures,
        publication_performed=False, model_compute_performed=False, bootstrap_performed=False,
        interpretation='Three paired seeds, descriptive only. No partial comparison or sweep result.'))
    dump(out/'artifact_hashes.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return dict(status='completed', report=str(out/'REPORT.html'), evaluated_checkpoints=18)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Experiment root, read only')
    parser.add_argument('--output', type=Path, required=True, help='New subdirectory under reporting/')
    args = parser.parse_args()
    try:
        print(json.dumps(generate(args.input, args.output), sort_keys=True))
    except (ReportError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status='blocked', reason=str(exc), comparison_generated=False)), file=sys.stderr)
        sys.exit(2)
