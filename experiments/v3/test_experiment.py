import unittest
from speculative import *
from experiment import *
class Tests(unittest.TestCase):
 def test_exact_formula(self):
  for p,q in zip(TARGET,DRAFT):
   for x,y in zip(p,corrected_distribution(p,q)):self.assertAlmostEqual(x,y,places=14)
 def test_disjoint_support(self):self.assertEqual(corrected_distribution((1,0),(0,1)),(1,0))
 def test_identical(self):self.assertEqual(corrected_distribution((.5,.5),(.5,.5)),(.5,.5))
 def test_zero_draft_support(self):self.assertEqual(corrected_distribution((.5,.5),(1,0)),(.5,.5))
 def test_output_length(self):
  for n in (0,1,3,19,100):self.assertEqual(len(speculative(TARGET,DRAFT,n)[0]),n)
 def test_deterministic(self):self.assertEqual(speculative(TARGET,DRAFT,100),speculative(TARGET,DRAFT,100))
 def test_identical_draft_no_rejection(self):self.assertEqual(speculative(TARGET,TARGET,100)[1]['rejections'],0)
 def test_transition_distribution(self):self.assertLess(probe(100000)['max_row_total_variation'],.015)
 def test_no_acceptance(self):self.assertEqual(expected_tokens(0,4),1)
 def test_full_acceptance(self):self.assertEqual(expected_tokens(1,4),5)
 def test_scored_not_free(self):self.assertGreater(point(.5,8,.005,.2)['target_scored_positions_per_emitted'],4)
 def test_worse_draft_slower(self):self.assertLess(point(.5,8,.01,1)['hypothetical_single_stream_speedup'],1)
 def test_prefill_still_blocks(self):self.assertFalse(point(.98,16,.001,.05)['passes_8h_capacity_and_margin_necessary_gate'])
 def test_zero_extra_gpu_still_blocks(self):self.assertFalse(point(.98,16,.001,.05,draft_gpus=0)['passes_8h_capacity_and_margin_necessary_gate'])
 def test_more_resources_cost_more(self):self.assertGreater(point(.9,4,.005,.2)['monthly_pool_usd'],point(.9,4,.005,.2,draft_gpus=0)['monthly_pool_usd'])
 def test_empty_matrix(self):
  with self.assertRaises(ValueError):speculative([],[],10)
 def test_overflow(self):
  with self.assertRaises(ValueError):normalized((1e308,1e308))
 def test_invalid(self):
  for a,k in ((-1,1),(2,1),(.5,0),(.5,1.5),(float('nan'),1)):
   with self.assertRaises(ValueError):expected_tokens(a,k)
  with self.assertRaises(ValueError):point(.5,4,-1,.2)
  with self.assertRaises(ValueError):corrected_distribution((1,0),(1,))
if __name__=='__main__':unittest.main()
