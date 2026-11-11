"""Synthetic fixture tests for the stdlib-only paired training audit helper."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import audit_training_pairs as audit


class PairAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dory_pair_audit_fixture_')
        self.root = Path(self.temp.name)
        protocol = dict(loops=2, input_blocks=8, output_blocks=8, recurrent_unique_blocks=20,
                        steps=2000, expected_parameters=764304)
        for name in audit.SOURCES:
            (self.root/name).write_text(json.dumps(protocol) if name == 'protocol.json' else 'SYNTHETIC FIXTURE, NOT EXPERIMENTAL SOURCE\n')
        frozen = {name: audit.sha(self.root/name) for name in audit.SOURCES}
        self.dump('source_hashes.json', frozen)
        for seed in (1, 2, 3):
            for kind in ('split', 'whole'):
                relative = f'runs/{kind}_s{seed}'
                directory = self.root/relative
                directory.mkdir(parents=True)
                checkpoints = []
                for tag, step, cpu in (('cpu_300', 700, 300.1), ('matched_step', 2000, 850.0), ('cpu_900', 2100, 900.1)):
                    p = directory/(tag+'.pt')
                    # Deliberately not a tensor/archive; this helper hashes bytes only.
                    p.write_bytes(f'SYNTHETIC FIXTURE {kind} {seed} {tag}'.encode())
                    checkpoints.append(dict(tag=tag, step=step, cpu=cpu, wall=cpu+0.1,
                        checkpoint_sha256=audit.sha(p), batch_sha256=hashlib.sha256(f'{seed}:{step}'.encode()).hexdigest()))
                training = dict(kind=kind, seed=seed, status='training_complete', parameters=764304,
                    source_hashes=frozen, protocol=protocol, steps=2100, cpu=900.1, wall=900.2,
                    initial_hash=hashlib.sha256(str(seed).encode()).hexdigest(), checkpoints=checkpoints,
                    input_blocks=8, output_blocks=8, applied_layers=56, recurrent_applications=40,
                    peak_rss_kib=1000,
                    schedule=list(range(10))*2+list(range(10,20))*2 if kind == 'split' else list(range(20))*2)
                self.dump(relative+'/training.json', training)

    def tearDown(self):
        self.temp.cleanup()

    def dump(self, relative, value):
        (self.root/relative).write_text(json.dumps(value, indent=2)+'\n')

    def edit_training(self, function, run='whole_s1'):
        relative = f'runs/{run}/training.json'
        value = audit.read(self.root/relative)
        function(value)
        self.dump(relative, value)

    def call(self, *args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return audit.main(['--input', str(self.root), *args])

    def test_complete_fixture_derives_exact_reporter_schema_without_writes(self):
        result, identities = audit.collect(self.root)
        self.assertEqual(set(result), {'training_only_checks','all_six_fits_complete','total_training_update_cpu_seconds',
                                     'max_training_process_peak_rss_kib','heldout_scoring'})
        self.assertTrue(result['all_six_fits_complete'])
        self.assertEqual(len(result['training_only_checks']), 3)
        self.assertAlmostEqual(result['total_training_update_cpu_seconds'], 5400.6)
        self.assertEqual(len([p for p in identities if p.endswith('.pt')]), 18)
        self.assertFalse((self.root/'completed_pair_checks.json').exists())

    def test_default_cli_read_only(self):
        self.assertEqual(self.call(), 0)
        self.assertFalse((self.root/'completed_pair_checks.json').exists())

    def test_explicit_write_creates_only_derived_fixture_audit(self):
        before = {p.relative_to(self.root) for p in self.root.rglob('*')}
        self.assertEqual(self.call('--write'), 0)
        after = {p.relative_to(self.root) for p in self.root.rglob('*')}
        self.assertEqual(after-before, {Path('completed_pair_checks.json')})
        self.assertTrue(audit.read(self.root/'completed_pair_checks.json')['all_six_fits_complete'])

    def test_existing_audit_never_overwritten(self):
        self.dump('completed_pair_checks.json', {'preserve': True})
        original = (self.root/'completed_pair_checks.json').read_bytes()
        self.assertEqual(self.call('--write'), 2)
        self.assertEqual((self.root/'completed_pair_checks.json').read_bytes(), original)

    def test_partial_training_blocks_without_output(self):
        self.edit_training(lambda t: t.update(status='running'))
        self.assertEqual(self.call('--write'), 2)
        self.assertFalse((self.root/'completed_pair_checks.json').exists())

    def test_initialization_mismatch_rejected(self):
        self.edit_training(lambda t: t.update(initial_hash='0'*64))
        with self.assertRaisesRegex(audit.AuditError, 'unequal initialization'):
            audit.collect(self.root)

    def test_matched_step_data_mismatch_rejected(self):
        self.edit_training(lambda t: next(c for c in t['checkpoints'] if c['tag']=='matched_step').update(batch_sha256='1'*64))
        with self.assertRaisesRegex(audit.AuditError, 'data hashes differ'):
            audit.collect(self.root)

    def test_all_equal_step_data_hashes_checked_not_only_matched_step(self):
        self.edit_training(lambda t: next(c for c in t['checkpoints'] if c['tag']=='cpu_300').update(batch_sha256='2'*64))
        with self.assertRaisesRegex(audit.AuditError, 'data hashes differ'):
            audit.collect(self.root)

    def test_changed_checkpoint_bytes_rejected(self):
        (self.root/'runs/split_s2/cpu_900.pt').write_bytes(b'changed fixture')
        with self.assertRaisesRegex(audit.AuditError, 'Hash mismatch'):
            audit.collect(self.root)

    def test_changed_frozen_source_rejected(self):
        (self.root/'model.py').write_text('changed fixture')
        with self.assertRaisesRegex(audit.AuditError, 'Hash mismatch'):
            audit.collect(self.root)

    def test_missing_checkpoint_rejected(self):
        (self.root/'runs/whole_s3/matched_step.pt').unlink()
        with self.assertRaisesRegex(audit.AuditError, 'Missing input'):
            audit.collect(self.root)

    def test_wrong_checkpoint_threshold_rejected(self):
        self.edit_training(lambda t: next(c for c in t['checkpoints'] if c['tag']=='matched_step').update(step=1999))
        with self.assertRaisesRegex(audit.AuditError, 'threshold not reached'):
            audit.collect(self.root)

    def test_duplicate_checkpoint_rejected(self):
        self.edit_training(lambda t: t['checkpoints'].__setitem__(0, t['checkpoints'][1]))
        with self.assertRaisesRegex(audit.AuditError, 'three distinct'):
            audit.collect(self.root)

    def test_unmatched_source_manifest_rejected(self):
        self.edit_training(lambda t: t.update(source_hashes={}))
        with self.assertRaisesRegex(audit.AuditError, 'source/protocol mismatch'):
            audit.collect(self.root)

    def test_invalid_topology_rejected(self):
        self.edit_training(lambda t: t.update(input_blocks=0))
        with self.assertRaisesRegex(audit.AuditError, 'topology mismatch'):
            audit.collect(self.root)

    def test_symlink_checkpoint_rejected(self):
        p = self.root/'runs/whole_s1/cpu_900.pt'
        p.unlink()
        p.symlink_to('matched_step.pt')
        with self.assertRaisesRegex(audit.AuditError, 'Symlink'):
            audit.collect(self.root)


if __name__ == '__main__':
    unittest.main()
