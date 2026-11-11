"""Synthetic receipt/metric fixtures only. Never trains, infers or loads corpora."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import report_bounded_sweep as report

HERE = Path(__file__).resolve().parent
SENTINEL = 'DO_NOT_EXPORT_PRIVATE_FIXTURE_CONTENT'


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(root):
    root.mkdir()
    src,cal=root/'bounded_sweep_results',root/'sweep_results'
    src.mkdir();cal.mkdir()
    for name in report.ORIGINAL_SOURCES:
        (root/name).write_text('SYNTHETIC ORIGINAL SOURCE: '+name+'\n')
    original_sources={name:report.sha(root/name) for name in report.ORIGINAL_SOURCES}
    report.dump(root/'source_hashes.json',original_sources)
    protocol=dict(report.EXPECTED_PROTOCOL)
    for name in report.AMENDED_SOURCES-{'source_hashes.json','bounded_sweep_protocol.json'}:
        (root/name).write_text('SYNTHETIC AMENDMENT SOURCE: '+name+'\n')
    report.dump(root/'bounded_sweep_protocol.json',protocol)
    amended_sources={name:report.sha(root/name) for name in report.AMENDED_SOURCES}
    report.dump(root/'bounded_source_hashes.json',amended_sources)
    checkpoints={j:digest('checkpoint '+j) for j in report.JOBS}
    envelopes={j:{'a':.03125,'b':.046875} for j in report.JOBS}
    prior=81.06850944599864
    raw=dict(synthetic_input='arange(8*96) modulo256',synthetic_target='arange(8*32) modulo256',
        batch=8,counts=[0,1,2,8,32,100],repetitions=2,warmup_loop_count=2,
        warmups=[dict(job=j,status='completed') for j in report.JOBS],
        samples=[dict(job=j,loops=c,repeat=r,seconds=.03125+.046875*c,status='completed')
                 for j in report.JOBS for c in (0,1,2,8,32,100) for r in (0,1)],
        private_text=SENTINEL)
    candidates=[]
    for n in (32,16,8):
        times={j:(n/4)*(101*e['a']+5050*e['b'])+32*(7*e['a']+147*e['b']) for j,e in envelopes.items()}
        bound=prior+600+1.5*sum(times.values())
        candidates.append(dict(windows_per_domain=n,predicted_seconds_per_checkpoint=times,
            global_bound_seconds=bound,fits=bound<=7200 and all(1.5*t<=1050 for t in times.values())))
    original=dict(status='blocked_no_fit',windows_per_domain=None,envelopes=envelopes,candidates=candidates,
        elapsed_debited_seconds=prior,calibration_elapsed_seconds=80.,checkpoint_hashes=checkpoints)
    blocker=dict(status='blocked_no_fit',heldout_loaded=False)
    for name,obj in zip(report.ORIGINAL_RECEIPTS,(raw,original,blocker)):report.dump(cal/name,obj)
    selection_times={j:2*(101*e['a']+5050*e['b'])+16*(3*e['a']+102*e['b']) for j,e in envelopes.items()}
    selection=dict(status='selected',windows_per_domain=8,selected_windows_per_domain=64,
        selected_counts=[0,2,100],predicted_seconds_per_checkpoint=selection_times,
        global_bound_seconds=prior+1+600+1.5*sum(selection_times.values()),
        prior_elapsed_debited_seconds=prior,new_setup_debited_seconds=1.,checkpoint_hashes=checkpoints,envelopes=envelopes)
    amendment=dict(protocol=protocol,original_source_hashes=original_sources,amended_source_hashes=amended_sources,
        calibration_receipt_manifest=dict(original_source_manifest_sha256=report.sha(root/'source_hashes.json'),
            receipts_sha256={name:report.sha(cal/name) for name in report.ORIGINAL_RECEIPTS}),
        standard_evaluations_verified=[dict(kind=k,seed=s,training_receipt_sha256=digest('training '+k+str(s)),
            standard_result_sha256=digest('standard result '+k+str(s)),standard_evaluation_sha256=digest('evaluation '+k+str(s)),
            private_text=SENTINEL) for k in report.KINDS for s in report.SEEDS],heldout_loaded=False,
        reason='Synthetic runtime-only scope reduction')
    cohorts={}
    for phase in report.PHASES:
        n=report.WINDOWS[phase];cohorts[phase]={}
        for domain in report.DOMAINS:
            total=10059 if domain=='test' else 2134
            cohorts[phase][domain]=dict(windows=n,selection_sha256=digest(phase+domain),
                indices=report.linspace_indices(total,n),
                source={'evaluated_windows':total,'private_text':SENTINEL})
    manifest=dict(protocol=protocol,source_hashes=original_sources,amended_source_hashes=amended_sources,
        selection=selection,cohorts=cohorts)
    report.dump(src/'amendment_receipt.json',amendment)
    report.dump(src/'manifest.json',manifest);report.dump(src/'cohort_selection.json',selection)
    for kind in report.KINDS:
        for seed in report.SEEDS:
            rows=[]
            for phase,counts in report.PHASES.items():
                for loops in counts:
                    datasets={};n=report.WINDOWS[phase]
                    for di,domain in enumerate(report.DOMAINS):
                        loss=3+.1*seed+.02*di+.001*loops+(.01 if kind=='split' else 0)
                        datasets[domain]=dict(status='completed',evaluated_windows=n,suffix_bpb=loss/math.log(2),
                            byte_accuracy=.25,exact_32_byte_match=0.,forward_cpu_seconds=.1+loops*.01,
                            forward_wall_seconds=.2+loops*.02,per_window_loss_nats=[loss]*n,groups=[i%4 for i in range(n)])
                    rows.append(dict(phase=phase,loops=loops,trained_loops=2,applied_layers=16+20*loops,
                        recurrent_applications=20*loops,unique_parameters=764304,
                        invoked_module_parameters=353424 if loops==0 else 764304,datasets=datasets))
            report.dump(src/f'{kind}_s{seed}.json',dict(kind=kind,seed=seed,checkpoint_sha256=checkpoints[f'{kind}_{seed}'],
                checkpoint_meta=dict(tag='matched_step',step=2000,batch_sha256=digest('seed data '+str(seed))),
                elapsed=100.,rows=rows))
    report.dump(src/'status.json',dict(runs=[dict(kind=k,seed=s,completed_cells=208,nonfinite_cells=0,
        missing_or_capped_cells=0,total_cells=208) for k in report.KINDS for s in report.SEEDS],
        prior_elapsed_debited_seconds=prior,new_elapsed_seconds=600.,combined_elapsed_seconds=prior+600.,global_capped=False))
    return src


class BoundedReporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='_bounded_report_test_',dir=HERE)
        cls.base=Path(cls.temp.name);cls.src=fixture(cls.base/'input');cls.root=cls.src.parent

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    @contextmanager
    def change(self,relative,modify):
        path=self.root/relative;before=path.read_bytes();value=json.loads(before);modify(value);report.dump(path,value)
        try:yield
        finally:path.write_bytes(before)

    def collect(self):return report.collect(self.src)

    def test_complete_counts_pairing_and_budget(self):
        rows,runtimes,provenance=self.collect();summary=report.aggregate(rows);pairs,ps=report.paired(rows)
        self.assertEqual((len(rows),len(summary),len(pairs),len(ps),len(runtimes)),(1248,416,624,208,6))
        self.assertEqual({r['loops'] for r in rows if r['phase']=='selected'},{0,2,100})
        self.assertTrue(all(r['cohort_windows']==(8 if r['phase']=='all101' else 64) for r in rows))
        self.assertTrue(all(r['invoked_module_parameters']==353424 and r['applied_layers']==16 for r in rows if r['loops']==0))
        self.assertAlmostEqual(ps[0]['suffix_bpb_difference_mean'],.01/math.log(2))
        self.assertAlmostEqual(provenance['budget']['prior_elapsed_debited_seconds'],81.06850944599864)

    def test_linspace_float_boundary_matches_saved_numpy_selection(self):
        self.assertEqual(report.linspace_indices(2134,64)[7],236)
        self.assertEqual(report.linspace_indices(2134,64)[-1],2133)
        self.assertEqual(len(set(report.linspace_indices(2134,64))),64)

    def test_missing_final_status_never_reads_scores(self):
        path=self.src/'status.json';data=path.read_bytes();path.unlink();opened=[];original_read=report.read
        def tracked(p):opened.append(p.name);return original_read(p)
        try:
            with mock.patch.object(report,'read',side_effect=tracked):
                with self.assertRaises(FileNotFoundError):report.generate(self.src,self.base/'blocked')
            self.assertEqual(opened,['status.json']);self.assertFalse((self.base/'blocked').exists())
        finally:path.write_bytes(data)

    def test_partial_status_is_rejected(self):
        with self.change('bounded_sweep_results/status.json',lambda v:v['runs'].pop()):
            with self.assertRaisesRegex(report.ReportError,'finalized six-run'):self.collect()

    def test_original_216_cell_status_is_rejected(self):
        with self.change('bounded_sweep_results/status.json',lambda v:v['runs'][0].update(total_cells=216)):
            with self.assertRaisesRegex(report.ReportError,'208 cells'):self.collect()

    def test_final_timing_required(self):
        with self.change('bounded_sweep_results/status.json',lambda v:v.pop('combined_elapsed_seconds')):
            with self.assertRaisesRegex(report.ReportError,'final timing'):self.collect()

    def test_retained_prior_debit_required(self):
        with self.change('bounded_sweep_results/status.json',lambda v:v.update(prior_elapsed_debited_seconds=0,combined_elapsed_seconds=600)):
            with self.assertRaisesRegex(report.ReportError,'runtime debit was not retained'):self.collect()

    def test_original_calibration_change_is_rejected(self):
        with self.change('sweep_results/blocker.json',lambda v:v.update(heldout_loaded=True)):
            with self.assertRaisesRegex(report.ReportError,'calibration receipt changed'):self.collect()

    def test_original_no_fit_is_verified_not_assumed(self):
        raw=report.read(self.root/'sweep_results/calibration_raw.json')
        original=report.read(self.root/'sweep_results/cohort_selection.json')
        blocker=report.read(self.root/'sweep_results/blocker.json');blocker['heldout_loaded']=True
        with self.assertRaisesRegex(report.ReportError,'before held-out'):report.verify_original(raw,original,blocker)

    def test_original_predictions_are_verified(self):
        raw=report.read(self.root/'sweep_results/calibration_raw.json');original=report.read(self.root/'sweep_results/cohort_selection.json')
        blocker=report.read(self.root/'sweep_results/blocker.json');original['candidates'][0]['global_bound_seconds']+=1
        with self.assertRaisesRegex(report.ReportError,'no-fit decision mismatch'):report.verify_original(raw,original,blocker)

    def test_source_changes_rejected(self):
        path=self.root/'model.py';data=path.read_bytes();path.write_bytes(data+b'changed')
        try:
            with self.assertRaisesRegex(report.ReportError,'Source changed'):self.collect()
        finally:path.write_bytes(data)

    def test_original_128_window_scope_is_rejected(self):
        def mutate(v):v['protocol']['selected_windows_per_domain']=128
        with self.change('bounded_sweep_results/manifest.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'Amendment protocol differs'):self.collect()

    def test_changed_fixed_cohort_rejected(self):
        def mutate(v):v['cohorts']['selected']['test']['windows']=128
        with self.change('bounded_sweep_results/manifest.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'fixed evenly-spaced'):self.collect()

    def test_changed_checkpoint_identity_rejected(self):
        with self.change('bounded_sweep_results/split_s1.json',lambda v:v.update(checkpoint_sha256=digest('wrong'))):
            with self.assertRaisesRegex(report.ReportError,'Checkpoint identity'):self.collect()

    def test_checkpoint_must_be_matched_update(self):
        with self.change('bounded_sweep_results/split_s1.json',lambda v:v['checkpoint_meta'].update(step=1999)):
            with self.assertRaisesRegex(report.ReportError,'matched-2000'):self.collect()

    def test_paired_training_stream_checked(self):
        with self.change('bounded_sweep_results/split_s1.json',lambda v:v['checkpoint_meta'].update(batch_sha256=digest('wrong'))):
            with self.assertRaisesRegex(report.ReportError,'training data stream'):self.collect()

    def test_duplicate_row_rejected(self):
        with self.change('bounded_sweep_results/split_s1.json',lambda v:v['rows'].append(v['rows'][0])):
            with self.assertRaisesRegex(report.ReportError,'Duplicate or unexpected'):self.collect()

    def test_wrong_outer_parameter_count_rejected(self):
        with self.change('bounded_sweep_results/split_s1.json',lambda v:v['rows'][0].update(invoked_module_parameters=24720)):
            with self.assertRaisesRegex(report.ReportError,'loop/parameter'):self.collect()

    def test_paired_group_order_rejected(self):
        def mutate(v):v['rows'][0]['datasets']['test']['groups'].reverse()
        with self.change('bounded_sweep_results/whole_s1.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'paired document order'):self.collect()

    def test_missing_nonfinite_capped_and_invalid_preserved(self):
        path=self.src/'whole_s3.json';data=path.read_bytes();path.unlink()
        def mutate(v):
            for i,status in enumerate(('nonfinite_logits','capped','missing_resource_cap')):
                v['rows'][i]['datasets']['test']={'status':status,'evaluated_windows':0}
            v['rows'][3]['datasets']['test']['suffix_bpb']=-1
            v['rows'][4]['datasets']['test']['status']='future_failure_kind'
            del v['rows'][5]
        try:
            with self.change('bounded_sweep_results/split_s1.json',mutate):
                rows,runtimes,p=self.collect();summary=report.aggregate(rows)
                for status in ('nonfinite_logits','capped','missing_resource_cap','invalid_completed_metrics','unrecognized_status','missing_cell','missing_file'):
                    self.assertTrue(any(r['status']==status for r in rows),status)
                self.assertTrue(any(r['source_status']=='future_failure_kind' for r in rows))
                self.assertTrue(all(r['suffix_bpb'] is None for r in rows if r['status']!='completed'))
                self.assertEqual(len(rows),1248);self.assertTrue(any(not s['complete'] for s in summary))
        finally:path.write_bytes(data)

    def test_false_final_ledger_is_rejected(self):
        def mutate(v):v['rows'][0]['datasets']['test']={'status':'capped'}
        with self.change('bounded_sweep_results/split_s1.json',mutate):
            with self.assertRaisesRegex(report.ReportError,'status does not match cell ledger'):self.collect()

    def test_nonfinite_json_rejected(self):
        path=self.src/'split_s1.json';data=path.read_bytes();path.write_bytes(data.replace(b'"elapsed":',b'"nonfinite": NaN, "elapsed":',1))
        try:
            with self.assertRaisesRegex(report.ReportError,'Nonfinite JSON'):self.collect()
        finally:path.write_bytes(data)

    def test_deterministic_safe_outputs_and_svg(self):
        a,b=self.base/'report_a',self.base/'report_b'
        report.generate(self.src,a);report.generate(self.src,b)
        self.assertEqual({p.name:report.sha(p) for p in a.iterdir()},{p.name:report.sha(p) for p in b.iterdir()})
        self.assertFalse(any(SENTINEL in p.read_text() for p in a.iterdir()))
        self.assertEqual(len(list(a.glob('*.svg'))),8)
        for p in a.glob('*.svg'):ET.parse(p)
        self.assertIn('64 windows/domain at r=0,2,100 only',(a/'selected_test_suffix_bpb.svg').read_text())
        self.assertIn('8 windows/domain, all 101 counts',(a/'all101_test_suffix_bpb.svg').read_text())
        text=(a/'REPORT.md').read_text()
        for phrase in ('1,200','7,200','8 fixed','64 fixed','did not run','353,424','764,304','96-byte','0.0005','600B','1248','416'):
            self.assertIn(phrase,text)
        self.assertNotIn('{{',text)
        status=report.read(a/'report_status.json')
        self.assertTrue(status['evaluation_finalized']);self.assertTrue(status['all_cells_completed'])
        self.assertFalse(status['model_compute_performed'])
        with self.assertRaisesRegex(report.ReportError,'overwrite'):report.generate(self.src,a)

    def test_output_scope_enforced(self):
        with self.assertRaisesRegex(report.ReportError,'under reporting'):report.generate(self.src,HERE.parent/'bad_output')


if __name__=='__main__':unittest.main(verbosity=2)
