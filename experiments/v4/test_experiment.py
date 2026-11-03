import unittest
from experiment import replay,trace,PREFIX,INPUT
class Tests(unittest.TestCase):
 def test_empty(self): self.assertEqual(replay([],0)['requests'],0)
 def test_no_cache(self): self.assertEqual(replay([1]*10,0)['hits'],0)
 def test_cold(self): self.assertEqual(replay([1]*10,1)['hits'],9)
 def test_lru(self): self.assertEqual(replay([1,2,1,3,2],2)['hits'],1)
 def test_tenant(self): self.assertEqual(replay([(1,0,0),(2,0,0)],2)['hits'],0)
 def test_revision(self): self.assertEqual(replay([(1,0,0),(1,0,1)],2)['hits'],0)
 def test_full_work(self): self.assertEqual(len(list(trace(3,7,4))),21)
 def test_order_same_work(self): self.assertEqual(sorted(trace(3,9,1)),sorted(trace(3,9,4)))
 def test_conservation(self):
  r=replay([1]*10,1);self.assertEqual(r['computed_input_tokens']+r['hits']*PREFIX,10*INPUT)
 def test_max_fraction(self): self.assertLessEqual(replay([1]*100,1)['cached_input_fraction'],PREFIX/INPUT)
 def test_invalid(self):
  for n in (-1,1.5):
   with self.assertRaises(ValueError): replay([],n)
 def test_not_economics_pass(self): self.assertFalse(replay([1]*100,2)['all_in_feasible'])
if __name__=='__main__':unittest.main()
