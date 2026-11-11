"""Pure mock tests: standard library only, no model/tensor/data execution."""
import unittest
from runtime_plan import COUNTS,choose,predict,validate_protocol
JOBS=[f'{kind}_{seed}' for seed in (1,2,3) for kind in ('split','whole')]
def fake(b):
    return [dict(job=j,loops=r,repeat=k,status='completed',seconds=.001+b*r) for j in JOBS for r in COUNTS for k in range(2)]
class PlanTests(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(predict(2,3,8),2*(202+15150)+32*(14+441))
    def test_largest_fit(self):
        for b,n in ((.001,32),(.02,16),(.03,8),(.06,None)):
            self.assertEqual(choose(fake(b),JOBS,10,9)['windows_per_domain'],n)
    def test_incomplete(self):
        with self.assertRaises(ValueError):choose(fake(.001)[:-1],JOBS,10,9)
    def test_nonfinite(self):
        rows=fake(.001);rows[0]['seconds']=float('nan')
        with self.assertRaises(ValueError):choose(rows,JOBS,10,9)
    def test_status(self):
        rows=fake(.001);rows[0]['status']='nonfinite_logits'
        with self.assertRaises(ValueError):choose(rows,JOBS,10,9)
    def test_caps_and_duplicates(self):
        with self.assertRaises(ValueError):choose(fake(.001),JOBS,200,181)
        with self.assertRaises(ValueError):choose(fake(.001),JOBS[:-1]+[JOBS[0]],10,9)
        self.assertIsNone(choose(fake(.001),JOBS,7190,9)['windows_per_domain'])
    def test_elapsed_includes_calibration(self):
        with self.assertRaises(ValueError):choose(fake(.001),JOBS,8,9)
    def test_protocol_drift(self):
        import json
        from pathlib import Path
        cfg=json.loads(Path(__file__).with_name('sweep_protocol.json').read_text())
        validate_protocol(cfg)
        cfg['batch']=4
        with self.assertRaises(ValueError):validate_protocol(cfg)
    def test_formula_equals_explicit_batch_work(self):
        for n in (8,16,32):
            expected=2*(n/8)*sum(.01+.02*r for r in range(101))+2*(128/8)*sum(.01+.02*r for r in (0,1,2,4,8,32,100))
            self.assertAlmostEqual(predict(.01,.02,n),expected)
    def test_slower_never_increases_cohort(self):
        values=[choose(fake(b),JOBS,10,9)['windows_per_domain'] or 0 for b in (.001,.01,.02,.03,.06)]
        self.assertEqual(values,sorted(values,reverse=True))
if __name__=='__main__':unittest.main()
