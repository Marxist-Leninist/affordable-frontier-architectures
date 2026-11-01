import unittest
from model import Serving,budget,weights_bytes,kv_bytes,metered_cost
class Tests(unittest.TestCase):
    def test_weights(self):
        self.assertEqual(weights_bytes(bits=16),6e12)
        self.assertEqual(weights_bytes(bits=4),1.5e12)
        self.assertEqual(weights_bytes(bits=1),375e9)
    def test_budgets(self):
        self.assertAlmostEqual(budget(20),4)
        self.assertAlmostEqual(budget(200),49)
    def test_linear_cost(self):
        s=Serving(); self.assertAlmostEqual(s.cost(2e6,4e5),2*s.cost(1e6,2e5))
        a,b=s.rates; self.assertAlmostEqual(s.cost(1e6,2e6),a+2*b)
    def test_utilization(self):
        self.assertAlmostEqual(Serving(utilization=.3).cost(1000,1000),2*Serving(utilization=.6).cost(1000,1000))
    def test_capacity_conservation(self):
        s=Serving(); self.assertAlmostEqual(s.cost(1e6,2e5)*s.subscribers_capacity(1e6,2e5),s.hourly*s.hours_month)
    def test_kv(self): self.assertEqual(kv_bytes(32768),4*kv_bytes(8192))
    def test_hidden_and_cache(self):
        s=Serving()
        self.assertEqual(metered_cost(s,1000,100,100),s.cost(1000,200))
        self.assertEqual(metered_cost(s,1000,100,0,1,0),s.cost(0,100))
        self.assertEqual(metered_cost(s,1000,100,0,1,1),s.cost(1000,100))
    def test_invalid(self):
        for u in (0,-1,2):
            with self.assertRaises(ValueError): Serving(utilization=u)
        with self.assertRaises(ValueError): Serving().cost(-1,0)
if __name__=='__main__':unittest.main()
