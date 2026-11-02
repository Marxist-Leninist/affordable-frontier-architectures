import unittest, math
from experiment import *
class Tests(unittest.TestCase):
 def test_fixed_ledger(self):
  r=simulate(1,24);self.assertEqual((r['input_tokens_per_user'],r['visible_tokens_per_user'],r['hidden_tokens_per_user']),(10e6,2e6,2e6))
 def test_exact_request_count(self):self.assertEqual(len(list(arrivals(3,8))),6000)
 def test_monotonic_arrivals(self):
  a=list(arrivals(2,8));self.assertEqual(a,sorted(a));self.assertLess(a[-1],MONTH)
 def test_single_request_latency(self):self.assertAlmostEqual(simulate(1,24)['p95_first_visible_seconds'],50.725,places=6)
 def test_hidden_not_free(self):self.assertGreater(simulate(1,24)['p95_first_visible_seconds'],50)
 def test_pool_double_copy_cost(self):self.assertEqual(Pool().prefill_gpus+Pool().decode_gpus,64)
 def test_min_users_boundary(self):
  p=Pool();n=p.min_users_for_margin;self.assertLessEqual(p.hourly*720/n,p.allowance);self.assertGreater(p.hourly*720/(n-1),p.allowance)
 def test_capacity_hours(self):self.assertAlmostEqual(Pool().within_window_capacity(24),3*Pool().within_window_capacity(8))
 def test_impossible_reserve(self):self.assertIsNone(Pool(gross_usd=25).min_users_for_margin)
 def test_dedicated_fails(self):self.assertFalse(simulate(1,24)['margin_gate'])
 def test_drain_charged(self):
  r=simulate(30,24,Pool(decode_slots=1));self.assertGreater(r['drain_after_month_seconds'],0);self.assertGreater(r['serving_cost_per_user_usd'],Pool().hourly*720/30)
 def test_bad_inputs(self):
  for kw in ({'decode_slots':0},{'stream_tps':0},{'decode_slots':1.5},{'prefill_tps':float('nan')},{'target_margin':1},{'tax':float('nan')},{'other_usd':float('nan')},{'state_transfer_seconds':float('nan')},{'decode_slots':100.0}):
   with self.assertRaises(ValueError):Pool(**kw)
 def test_same_ledger_clustered(self):
  a=simulate(4,24);b=simulate(4,8);self.assertEqual(a['requests'],b['requests']);self.assertEqual(a['hidden_tokens_per_user'],b['hidden_tokens_per_user'])
 def test_deterministic(self):self.assertEqual(simulate(2,8),simulate(2,8))
if __name__=='__main__':unittest.main()
