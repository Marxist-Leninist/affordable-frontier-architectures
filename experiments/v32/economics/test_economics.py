import unittest
from economics import calculate, report
class EconomicsTests(unittest.TestCase):
    def test_fee_once(self): self.assertAlmostEqual(calculate()['net'],500/1.2-15)
    def test_budget(self): self.assertAlmostEqual(calculate()['serving_allowance'],131.16666666666669)
    def test_burst_cost_invariant(self): self.assertEqual(calculate()['monthly_pool_cost'],calculate(burst=4)['monthly_pool_cost'])
    def test_burst_throughput(self): self.assertEqual(calculate(burst=4)['required_generated_decode_tps'],4*calculate()['required_generated_decode_tps'])
    def test_cache_decode_invariant(self): self.assertEqual(calculate(cached_fraction=.5)['required_generated_decode_tps'],calculate()['required_generated_decode_tps'])
    def test_boundaries_and_workload(self): self.assertEqual(len(report()['rows']),48)
if __name__=='__main__': unittest.main()
