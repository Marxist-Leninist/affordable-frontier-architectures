"""Synthetic JSON/byte fixtures only. No model imports, training, or inference."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import report_topology as report

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(root):
    root.mkdir()
    protocol = dict(report.EXPECTED_PROTOCOL)
    report.dump(root/'protocol.json', protocol)
    for name in report.REQUIRED_SOURCES - {'protocol.json'}:
        (root/name).write_text('SYNTHETIC FIXTURE, NOT EXPERIMENT SOURCE: '+name+'\n')
    frozen = {name: report.sha(root/name) for name in report.REQUIRED_SOURCES}
    report.dump(root/'source_hashes.json', frozen)
    audit = dict(all_six_fits_complete=True, training_only_checks={
        str(seed): dict(identical_initial_parameters=True, matched_step_data_hash_identical=True,
                       checkpoint_hashes_verified=True) for seed in report.SEEDS})
    report.dump(root/'completed_pair_checks.json', audit)
    for seed in report.SEEDS:
        for kind in report.KINDS:
            directory = root/'runs'/f'{kind}_s{seed}'
            directory.mkdir(parents=True)
            checkpoints = []
            for tag, step, cpu in (('matched_step', 2000, 860.0),
                    ('cpu_300', 700+seed+(kind == 'whole'), 300.2),
                    ('cpu_900', 2100+seed+(kind == 'whole'), 900.2)):
                path = directory/(tag+'.pt')
                path.write_bytes(f'FAKE CHECKPOINT BYTES: {kind} {seed} {tag}'.encode())
                checkpoints.append(dict(tag=tag, step=step, cpu=cpu, wall=cpu+.1,
                    batch_sha256=digest(f'data seed{seed} step{step}'), checkpoint_sha256=report.sha(path)))
            tr = dict(kind=kind, seed=seed, status='training_complete', parameters=764304,
                initial_hash=digest(f'initial{seed}'), source_hashes=frozen, schedule=report.schedule(kind),
                input_blocks=8, output_blocks=8, applied_layers=56, recurrent_applications=40,
                protocol=protocol, steps=checkpoints[-1]['step'], cpu=900.2, wall=checkpoints[-1]['wall'],
                checkpoints=checkpoints, peak_rss_kib=12345)
            report.dump(directory/'training.json', tr)
            report.dump(directory/'result.json', dict(tr, status='completed',
                evaluated_checkpoints=3, evaluation_wall_seconds=100.0+seed))
            ev = []
            for ti, ck in enumerate(checkpoints):
                row = dict(ck, datasets={})
                for di, domain in enumerate(report.DOMAINS):
                    n = report.WINDOWS[domain]
                    base = 3+.1*di+.05*ti+.01*seed
                    loss = base+((seed-2)*.02-.005 if kind == 'split' else 0)
                    metrics = dict(loss_nats=loss, bpb=loss/math.log(2),
                        accuracy=(n*8+seed)/(n*32), exact_match=0.,
                        per_window_loss=[loss]*n, groups=[i % 10 for i in range(n)])
                    manifest = dict(evaluated_windows=n, documents=10,
                        selection_sha256=digest('synthetic selection '+domain),
                        source='DO_NOT_EXPORT_UNRECOGNIZED_MANIFEST_FIELDS')
                    row['datasets'][domain] = dict(metrics=metrics, manifest=manifest)
                ev.append(row)
            report.dump(directory/'evaluation.json', ev)


class ReporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='_topology_test_', dir=HERE)
        cls.base = Path(cls.temporary.name)
        cls.root = cls.base/'input'
        fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @contextmanager
    def change(self, relative, change):
        path = self.root/relative
        original = path.read_bytes()
        value = json.loads(original)
        change(value)
        report.dump(path, value)
        try:
            yield
        finally:
            path.write_bytes(original)

    def collect(self):
        return report.collect(self.root)

    def test_valid_full_fixture_and_paired_statistics(self):
        components, timing, run_timing, provenance = self.collect()
        self.assertEqual((len(components),len(timing),len(run_timing)), (54,18,6))
        self.assertEqual(len(provenance['checkpoint_sha256']), 18)
        summary, pairs, timing_summary = report.aggregate(components, timing)
        self.assertEqual((len(summary),len(pairs),len(timing_summary)), (27,81,11))
        bpb = summary[0]
        self.assertAlmostEqual(bpb['difference_mean'], -.005/math.log(2), places=12)
        self.assertAlmostEqual(bpb['difference_seed_sd'], .02/math.log(2), places=12)
        self.assertAlmostEqual(bpb['difference_minimum'], -.025/math.log(2), places=12)
        self.assertAlmostEqual(bpb['difference_maximum'], .015/math.log(2), places=12)
        self.assertEqual(summary[1]['favorable_difference'], 'positive')

    def test_original_readme_not_required_or_used(self):
        path = self.root/'README.md'
        self.assertFalse(path.exists())
        before = self.collect()
        self.assertNotIn('README.md', before[3]['input_file_sha256'])
        try:
            path.write_text('Unused prepared README; must not affect reporting.\n')
            self.assertEqual(before, self.collect())
            path.write_text('Replacement public README; must not affect reporting.\n')
            self.assertEqual(before, self.collect())
        finally:
            path.unlink()

    def test_incomplete_gate_opens_no_scores_and_writes_nothing(self):
        out = self.base/'blocked_output'
        original_read = report.read
        opened = []
        def tracked(path):
            opened.append(path.name)
            return original_read(path)
        with self.change('runs/whole_s3/result.json', lambda v:v.update(status='training_complete')):
            with mock.patch.object(report,'read',side_effect=tracked):
                with self.assertRaisesRegex(report.ReportError,'Full comparison unavailable'):
                    report.generate(self.root,out)
        self.assertNotIn('evaluation.json',opened)
        self.assertFalse(out.exists())

    def test_result_requires_all_three_checkpoints(self):
        with self.change('runs/whole_s3/result.json', lambda v:v.update(evaluated_checkpoints=2)):
            with self.assertRaisesRegex(report.ReportError,'three evaluated checkpoints'):
                self.collect()

    def test_missing_checkpoint_is_rejected(self):
        path = self.root/'runs/whole_s3/cpu_900.pt'
        data = path.read_bytes()
        path.unlink()
        try:
            with self.assertRaisesRegex(report.ReportError,'checkpoint file missing'):
                self.collect()
        finally:
            path.write_bytes(data)

    def test_checkpoint_byte_corruption_is_rejected(self):
        path = self.root/'runs/split_s1/matched_step.pt'
        data = path.read_bytes()
        path.write_bytes(data+b'changed')
        try:
            with self.assertRaisesRegex(report.ReportError,'checkpoint bytes changed'):
                self.collect()
        finally:
            path.write_bytes(data)

    def test_unequal_initialization_is_rejected(self):
        mutate = lambda v:v.update(initial_hash=digest('changed'))
        with self.change('runs/whole_s1/training.json',mutate), self.change('runs/whole_s1/result.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'unequal initialization'):
                self.collect()

    def test_unequal_matched_step_data_is_rejected(self):
        def mutate(v):
            v['checkpoints'][0]['batch_sha256']=digest('different data')
        with self.change('runs/whole_s1/training.json',mutate),self.change('runs/whole_s1/result.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'unequal equal-step data'):
                self.collect()

    def test_completion_receipt_identity_must_match(self):
        with self.change('runs/whole_s1/result.json',lambda v:v.update(seed=3)):
            with self.assertRaisesRegex(report.ReportError,'differs from training identity'):
                self.collect()

    def test_evaluation_checkpoint_identity_must_match(self):
        with self.change('runs/whole_s1/evaluation.json',lambda v:v[0].update(checkpoint_sha256=digest('changed'))):
            with self.assertRaisesRegex(report.ReportError,'evaluation checkpoint identity mismatch'):
                self.collect()

    def test_partial_evaluation_is_rejected(self):
        with self.change('runs/whole_s3/evaluation.json',lambda v:v.pop()):
            with self.assertRaisesRegex(report.ReportError,'all three evaluated checkpoints'):
                self.collect()

    def test_duplicate_checkpoint_tag_is_rejected(self):
        with self.change('runs/whole_s3/evaluation.json',lambda v:v[1].update(tag=v[0]['tag'])):
            with self.assertRaisesRegex(report.ReportError,'all three evaluated checkpoints'):
                self.collect()

    def test_selection_hash_mismatch_is_rejected(self):
        def mutate(v):
            v[0]['datasets']['test']['manifest']['selection_sha256']=digest('changed')
        with self.change('runs/whole_s1/evaluation.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'unequal evaluation manifest'):
                self.collect()

    def test_document_order_mismatch_is_rejected(self):
        def mutate(v):
            v[0]['datasets']['test']['metrics']['groups'].reverse()
        with self.change('runs/whole_s1/evaluation.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'unequal evaluation manifest or document order'):
                self.collect()

    def test_missing_windows_are_rejected(self):
        def mutate(v):
            v[0]['datasets']['test']['metrics']['per_window_loss'].pop()
        with self.change('runs/whole_s1/evaluation.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'invalid per-window losses'):
                self.collect()

    def test_miscomputed_bpb_is_rejected(self):
        def mutate(v):
            v[0]['datasets']['test']['metrics']['bpb'] += .1
        with self.change('runs/whole_s1/evaluation.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'aggregate loss does not match'):
                self.collect()

    def test_impossible_accuracy_is_rejected(self):
        def mutate(v):
            v[0]['datasets']['test']['metrics']['accuracy'] = 1.2
        with self.change('runs/whole_s1/evaluation.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'invalid metrics'):
                self.collect()

    def test_impossible_denominator_is_rejected(self):
        def mutate(v):
            v[0]['datasets']['test']['metrics']['exact_match'] = .000001
        with self.change('runs/whole_s1/evaluation.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'impossible accuracy denominator'):
                self.collect()

    def test_source_tampering_is_rejected(self):
        path = self.root/'model.py'
        data = path.read_bytes()
        path.write_bytes(data+b'changed')
        try:
            with self.assertRaisesRegex(report.ReportError,'Frozen source changed'):
                self.collect()
        finally:
            path.write_bytes(data)

    def test_wrong_topology_is_rejected(self):
        mutate = lambda v:v.update(schedule=report.schedule('split'))
        with self.change('runs/whole_s1/training.json',mutate), self.change('runs/whole_s1/result.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'topology identity mismatch'):
                self.collect()

    def test_nonfinite_json_is_rejected(self):
        path = self.root/'runs/whole_s1/evaluation.json'
        data = path.read_bytes()
        path.write_bytes(data.replace(b'"loss_nats":', b'"bad": NaN, "loss_nats":', 1))
        try:
            with self.assertRaisesRegex(report.ReportError,'Nonfinite JSON constant'):
                self.collect()
        finally:
            path.write_bytes(data)

    def test_duplicate_json_keys_are_rejected(self):
        path = self.root/'runs/whole_s1/result.json'
        data = path.read_bytes()
        path.write_bytes(data.replace(b'"status":', b'"status": "completed", "status":', 1))
        try:
            with self.assertRaisesRegex(report.ReportError,'Duplicate JSON key'):
                self.collect()
        finally:
            path.write_bytes(data)

    def test_report_is_deterministic_and_svg_well_formed(self):
        first, second = self.base/'report_a', self.base/'report_b'
        report.generate(self.root,first)
        report.generate(self.root,second)
        self.assertEqual({p.name:report.sha(p) for p in first.iterdir()},
                         {p.name:report.sha(p) for p in second.iterdir()})
        figures = list(first.glob('*.svg'))
        self.assertEqual(len(figures),12)
        for figure in figures:
            tree=ET.parse(figure)
            self.assertEqual(tree.getroot().tag,'{http://www.w3.org/2000/svg}svg')
        text=(first/'REPORT.md').read_text()
        for phrase in ('A^r B^r','764,304','353,424','600B','0.0005','32 MASK','reused','no target-token feedback'):
            self.assertIn(phrase,text)
        status=report.read(first/'report_status.json')
        self.assertFalse(status['model_compute_performed'])
        self.assertFalse(any('DO_NOT_EXPORT_UNRECOGNIZED_MANIFEST_FIELDS' in p.read_text()
                             for p in first.iterdir()))
        with self.assertRaisesRegex(report.ReportError,'overwrite'):
            report.generate(self.root,first)

    def test_output_scope_is_enforced(self):
        with self.assertRaisesRegex(report.ReportError,'Output must be'):
            report.generate(self.root,HERE.parent/'not_allowed_output')


if __name__ == '__main__':
    unittest.main(verbosity=2)
