#!/usr/bin/env python3
"""Derive a Dory paired-training audit from actual receipts and checkpoint hashes.

Standard library only. No model/data imports, inference or deserialization.
Default --check is read-only. --write creates completed_pair_checks.json once.
Historical NAT readiness receipts are never read, generated or modified here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

SOURCES = frozenset('model.py data_loop.py train.py test_correctness.py run_all.py protocol.json summarize.py sweep.py sweep_protocol.json control.py runtime_plan.py test_runtime_plan.py'.split())
TAGS = frozenset(('matched_step', 'cpu_300', 'cpu_900'))


class AuditError(ValueError):
    pass


def require(value, message):
    if not value:
        raise AuditError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hash_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def read(path):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'Duplicate JSON key')
            result[k] = v
        return result

    def nonfinite(value):
        raise AuditError('Nonfinite JSON constant: ' + value)

    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=nonfinite)


def collect(root):
    root = root.absolute()
    output = root / 'completed_pair_checks.json'
    require(not output.exists() and not output.is_symlink(), 'Refuse existing completed_pair_checks.json; inspect it instead of overwriting')
    identities = {}

    def checked(relative, expected=None):
        name = PurePosixPath(relative)
        require(not name.is_absolute() and '..' not in name.parts and str(name) == relative, 'Unsafe relative input')
        p = root / relative
        require(p.is_file(), 'Missing input: ' + relative)
        for ancestor in [p, *p.parents]:
            require(not ancestor.is_symlink(), 'Symlink input: ' + relative)
            if ancestor == root:
                break
        digest = sha(p)
        require(expected is None or digest == expected, 'Hash mismatch: ' + relative)
        identities[relative] = digest
        return p

    frozen = read(checked('source_hashes.json'))
    require(isinstance(frozen, dict) and set(frozen) == SOURCES, 'Require exactly 12 frozen source/protocol files')
    for name, digest in frozen.items():
        require(hash_valid(digest), 'Malformed frozen source digest')
        checked(name, digest)
    protocol = read(root / 'protocol.json')
    expected = dict(loops=2, input_blocks=8, output_blocks=8, recurrent_unique_blocks=20,
                    steps=2000, expected_parameters=764304)
    require(all(protocol.get(k) == v for k, v in expected.items()), 'Unexpected Dory training protocol')
    runs, checks = [], {}
    for seed in (1, 2, 3):
        pair = []
        for kind in ('split', 'whole'):
            run = f'{kind}_s{seed}'
            tr = read(checked(f'runs/{run}/training.json'))
            require(tr.get('status') == 'training_complete', run + ': training incomplete')
            require(tr.get('kind') == kind and tr.get('seed') == seed and tr.get('parameters') == 764304, run + ': identity mismatch')
            require(tr.get('source_hashes') == frozen and tr.get('protocol') == protocol, run + ': source/protocol mismatch')
            schedule = list(range(10))*2 + list(range(10, 20))*2 if kind == 'split' else list(range(20))*2
            require(tr.get('schedule') == schedule and tr.get('input_blocks') == tr.get('output_blocks') == 8
                    and tr.get('applied_layers') == 56 and tr.get('recurrent_applications') == 40, run + ': topology mismatch')
            require(hash_valid(tr.get('initial_hash')), run + ': invalid initialization digest')
            require(isinstance(tr.get('steps'), int) and not isinstance(tr['steps'], bool) and tr['steps'] >= 2000
                    and finite(tr.get('cpu')) and tr['cpu'] >= 900
                    and finite(tr.get('wall')) and tr['wall'] > 0, run + ': training thresholds not reached')
            require(isinstance(tr.get('peak_rss_kib'), int) and not isinstance(tr['peak_rss_kib'], bool)
                    and tr['peak_rss_kib'] > 0, run + ': invalid peak RSS')
            checkpoints = tr.get('checkpoints')
            require(isinstance(checkpoints, list) and len(checkpoints) == 3 and {c.get('tag') for c in checkpoints} == TAGS, run + ': require three distinct checkpoints')
            for c in checkpoints:
                tag = c['tag']
                require(hash_valid(c.get('batch_sha256')) and hash_valid(c.get('checkpoint_sha256')), run + ': malformed checkpoint/data digest')
                require(isinstance(c.get('step'), int) and not isinstance(c['step'], bool) and 0 < c['step'] <= tr['steps']
                        and finite(c.get('cpu')) and 0 < c['cpu'] <= tr['cpu']
                        and finite(c.get('wall')) and 0 < c['wall'] <= tr['wall'], run + ': invalid checkpoint timing')
                require(c['step'] == 2000 if tag == 'matched_step' else c['cpu'] >= int(tag[4:]), run + ': checkpoint threshold not reached')
                checked(f'runs/{run}/{tag}.pt', c['checkpoint_sha256'])
            ordered = sorted(checkpoints, key=lambda c: c['step'])
            require(all(a['cpu'] <= b['cpu'] and a['wall'] <= b['wall'] for a, b in zip(ordered, ordered[1:])), run + ': nonmonotonic checkpoint timing')
            pair.append(tr)
            runs.append(tr)
        a, b = pair
        require(a['initial_hash'] == b['initial_hash'], f'Seed {seed}: unequal initialization')
        for ca in a['checkpoints']:
            for cb in b['checkpoints']:
                if ca['step'] == cb['step']:
                    require(ca['batch_sha256'] == cb['batch_sha256'], f'Seed {seed}: equal-step data hashes differ')
        checks[str(seed)] = dict(identical_initial_parameters=True, matched_step_data_hash_identical=True,
                                checkpoint_hashes_verified=True)
    for relative, digest in identities.items():
        require(sha(root / relative) == digest, 'Input changed during audit: ' + relative)
    audit = dict(training_only_checks=checks, all_six_fits_complete=True,
                 total_training_update_cpu_seconds=sum(t['cpu'] for t in runs),
                 max_training_process_peak_rss_kib=max(t['peak_rss_kib'] for t in runs),
                 heldout_scoring='This audit opened no held-out corpus or scores. Standard held-out checkpoint evaluation is a separate phase.')
    return audit, identities


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='Verify only (default); no output file')
    group.add_argument('--write', action='store_true', help='Create a new derived completed_pair_checks.json after all checks pass')
    args = p.parse_args(argv)
    try:
        audit, identities = collect(args.input)
        if args.write:
            for relative, digest in identities.items():
                require(sha(args.input / relative) == digest, 'Input changed before writing audit: ' + relative)
            with (args.input / 'completed_pair_checks.json').open('x') as f:
                json.dump(audit, f, indent=2, allow_nan=False)
                f.write('\n')
        print(json.dumps(dict(status='verified', training_runs=6, checkpoints=18,
                              output_created=args.write, model_compute_performed=False)))
        return 0
    except (AuditError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps(dict(status='blocked', reason=str(exc), output_created=False)), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
