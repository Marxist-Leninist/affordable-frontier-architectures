import unittest
import numpy as np
from benchmark import LM,V,TRAIN,VALID,check
class Tests(unittest.TestCase):
 def test_gradient(self):self.assertLess(max(check().values()),1e-7)
 def test_disjoint_sentences(self):self.assertFalse(set(TRAIN.splitlines())&set(VALID.splitlines()))
 def test_causal(self):
  for kind in ['attention','recurrent']:
   m=LM(kind,4,3);x=np.array([[0,1,2,3,4]]);p=m.loss(x,None,False);x[0,3:]=[5,6];q=m.loss(x,None,False);np.testing.assert_array_equal(p[:,:3],q[:,:3])
 def test_normalized(self):
  for kind in ['attention','recurrent']:
   m=LM(kind,4,3);p=m.loss(np.array([[0,1,2]]),None,False);np.testing.assert_allclose(p.sum(-1),1)
 def test_parameter_match(self):
  a=LM('attention',16,0).count();r=min(LM('recurrent',d,0).count() for d in range(1,100) if LM('recurrent',d,0).count()>=a);self.assertLess((r-a)/a,.05)
if __name__=='__main__':unittest.main()
