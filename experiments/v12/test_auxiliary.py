import unittest
from economics import *
from ngram import Ngram
class Tests(unittest.TestCase):
 def test_hidden_and_cache(self):
  self.assertAlmostEqual(retail(WORKLOAD),.905)
  w=dict(WORKLOAD);w['hidden_output']=0;self.assertAlmostEqual(retail(WORKLOAD)-retail(w),.2)
 def test_budget(self):
  self.assertEqual(serving_budget(500,0,.3,{'other':150}),200)
  self.assertAlmostEqual(serving_budget(500,.2,.3,{'other':150}),141.6666666667)
  self.assertAlmostEqual(output_tps_required(1,.6,.1),4629.62962963)
 def test_ngram_mass_and_counts(self):
  m=Ngram();m.add(b'ababa');self.assertEqual(m.totals[0][b''],4)
  self.assertEqual(m.counts[1][b'a'][ord('b')],2)
  for c in [b'',b'a',b'xyz',b'abab']:
   self.assertAlmostEqual(sum(m.prob(c,y) for y in range(256)),1)
 def test_ngram_evaluation_boundaries(self):
  m=Ngram();m.add(b'ababa');raw=bytes(range(256))*2
  r=m.eval(raw,4);self.assertEqual(r['targets'],384);self.assertEqual(r['unscored_trailing_targets'],127)
if __name__=='__main__':unittest.main(verbosity=2)
