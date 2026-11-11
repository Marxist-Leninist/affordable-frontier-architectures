"""Pure synthetic timing and AST tests; no torch/model/data execution."""
import ast,json,tempfile,unittest
from pathlib import Path
from runtime_plan import choose
from bounded_runtime_plan import plan,predict,verify_calibration,sha,validate_protocol
ROOT=Path(__file__).resolve().parent
IDS=[f'{k}_{s}' for s in (1,2,3) for k in ('split','whole')]
def fixtures(directory):
    samples=[dict(job=j,loops=r,repeat=k,status='completed',seconds=.04+.05*r) for j in IDS for r in [0,1,2,8,32,100] for k in range(2)]
    original=choose(samples,IDS,81.,80.);original['checkpoint_hashes']={j:'a'*64 for j in IDS}
    raw=dict(synthetic_input='arange(8*96) modulo256',synthetic_target='arange(8*32) modulo256',batch=8,counts=[0,1,2,8,32,100],repetitions=2,warmup_loop_count=2,samples=samples,warmups=[dict(job=j,status='completed') for j in IDS])
    for n,v in [('calibration_raw.json',raw),('cohort_selection.json',original),('blocker.json',dict(status='blocked_no_fit',heldout_loaded=False))]:(directory/n).write_text(json.dumps(v))
    return dict(original_source_manifest_sha256='s',receipts_sha256={n:sha(directory/n) for n in ['calibration_raw.json','cohort_selection.json','blocker.json']}),original
class Tests(unittest.TestCase):
    def test_formula(self):self.assertEqual(predict(2,3),2*(202+15150)+16*(6+306))
    def test_revised_fits_original_did_not(self):
        with tempfile.TemporaryDirectory() as d:
            m,old=fixtures(Path(d));self.assertEqual(old['status'],'blocked_no_fit');verified=verify_calibration(Path(d),m,'s');new=plan(verified,10)
            self.assertEqual(new['status'],'selected');self.assertTrue(all(1.5*t<=1050 for t in new['predicted_seconds_per_checkpoint'].values()));self.assertLessEqual(new['global_bound_seconds'],7200)
    def test_actual_elapsed_debit(self):
        with tempfile.TemporaryDirectory() as d:
            _,old=fixtures(Path(d));a=plan(old,0);b=plan(old,25);self.assertEqual(b['global_bound_seconds']-a['global_bound_seconds'],25)
            old['elapsed_debited_seconds']=7000;self.assertEqual(plan(old,1)['status'],'blocked_no_fit')
    def test_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m,_=fixtures(root);(root/'blocker.json').write_text('{}')
            with self.assertRaises(ValueError):verify_calibration(root,m,'s')
    def test_bad_envelope_even_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m,old=fixtures(root);old['envelopes'][IDS[0]]['b']=0;(root/'cohort_selection.json').write_text(json.dumps(old));m['receipts_sha256']['cohort_selection.json']=sha(root/'cohort_selection.json')
            with self.assertRaises(ValueError):verify_calibration(root,m,'s')
    def test_source_binding(self):
        with tempfile.TemporaryDirectory() as d:
            m,_=fixtures(Path(d))
            with self.assertRaises(ValueError):verify_calibration(Path(d),m,'different')
    def test_protocol(self):
        cfg=json.loads((ROOT/'bounded_sweep_protocol.json').read_text());validate_protocol(cfg);cfg['selected_counts']=[0,2,99]
        with self.assertRaises(ValueError):validate_protocol(cfg)
    def test_static_gates_and_caps(self):
        s=(ROOT/'bounded_sweep.py').read_text();ast.parse(s)
        self.assertLess(s.index("result['status']=='completed'"),s.index("sets={'test':"));self.assertLess(s.index("write(out/'amendment_receipt.json'"),s.index("sets={'test':"))
        self.assertIn("for names in (original_hashes,new_hashes)",s);self.assertIn("global_deadline=started+cfg['global_wall_cap_seconds']-prior",s)
        self.assertIn('started=PROCESS_STARTED',s);self.assertIn("remaining=1200-job['elapsed']",s)
        self.assertIn('order=jobs[ordinal%6:]+jobs[:ordinal%6]',s);self.assertNotIn('calibrate(',s)
        self.assertIn("[('all101',cfg['counts']),('selected',cfg['selected_counts'])]",s)
if __name__=='__main__':unittest.main()
