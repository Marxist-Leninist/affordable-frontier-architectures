import unittest
import baseline as base
import numpy as np
from benchmark import Incremental
class Tests(unittest.TestCase):
    def test_prefix_parity_and_reset(self):
        for kind in ('attention','recurrent'):
            for seed in (1,2,3):
                m=base.LM(kind,4,seed); dec=Incremental(m,65)
                for tokens in (np.arange(65)%base.V,np.zeros(65,dtype=int)):
                    dec.reset(); actual=np.array([dec.step(t) for t in tokens]); expected=m.loss(tokens[None],None,False)[0]
                    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
                    with self.assertRaises(ValueError): dec.step(1)
    def test_resident_state(self):
        for kind in ('attention','recurrent'):
            m=base.LM(kind,4,1)
            for n in (32,128):
                dec=Incremental(m,n); expected=32+(2*n*4*8 if kind=='attention' else 0)
                self.assertEqual(dec.bytes(),expected)
    def test_sliding_window_is_not_same_semantics(self):
        for kind in ('attention','recurrent'):
            m=base.LM(kind,4,1); tokens=np.arange(64)%base.V
            full=m.loss(tokens[None],None,False)[0,-1]
            sliding=m.loss(tokens[None,-2:],None,False)[0,-1]
            self.assertGreater(float(np.max(abs(full-sliding))),1e-10)
if __name__=='__main__': unittest.main()
