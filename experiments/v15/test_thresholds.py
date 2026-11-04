import unittest
from thresholds import output_threshold,mixed_threshold
class Tests(unittest.TestCase):
 def test_reference(self): self.assertAlmostEqual(output_threshold(1),3968.253968253968)
 def test_utilization(self): self.assertAlmostEqual(output_threshold(1,.5),2*output_threshold(1))
 def test_hidden(self):
  self.assertAlmostEqual(output_threshold(1,hidden_per_visible=1),output_threshold(1))
  self.assertAlmostEqual(output_threshold(1,hidden_per_visible=1,hidden_billable=False),2*output_threshold(1))
 def test_other_exhausts_budget(self): self.assertIsNone(output_threshold(1,other_per_million=.07))
 def test_mixed_conservation(self):
  r=mixed_threshold(1,.6,80000,10e6,5e6,2e6,2e6)
  self.assertAlmostEqual(r['revenue'],.905)
  self.assertAlmostEqual(r['prefill_cost']+4e6/r['required_busy_generated_tps']/2160,r['total_cost_budget'])
 def test_prefill_exhaustion(self): self.assertIsNone(mixed_threshold(32*3.99*1.2,.6,8000,10e6,5e6,2e6,2e6)['required_busy_generated_tps'])
 def test_invalid(self):
  for kwargs in [dict(utilization=0),dict(margin=1),dict(hourly=float('nan'))]:
   with self.assertRaises(ValueError): output_threshold(**dict(hourly=1,**{k:v for k,v in kwargs.items() if k!='hourly'})) if 'hourly' not in kwargs else output_threshold(kwargs['hourly'])
if __name__=='__main__': unittest.main()
