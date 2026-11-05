import unittest, numpy as np
from bounded_cache import BoundedCache
from copy_cache import CopyCache
class Tests(unittest.TestCase):
 def test_reference(self):
  rng=np.random.default_rng(27)
  for raw in [b'abc abc abd '*150,bytes(rng.integers(0,8,1600).tolist()),bytes(range(256))*6]:
   ref=CopyCache();cs={s:BoundedCache(s) for s in (32,64,512,1024)}
   for i,b in enumerate(raw):
    ref.feed(b)
    for s,c in cs.items():
     c.feed(b)
     if i%13==0:
      p,n,k=c.distribution();q,nn,kk,_=ref.distribution(s)
      np.testing.assert_array_equal(p,q);self.assertEqual((n,k),(nn,kk))
     self.assertTrue(all(len(e)<=s-k for k,e in c.events.items()))
 def test_normalized(self):
  c=BoundedCache(64)
  for b in b'a'*1000:
   c.feed(b);q,_,_=c.distribution();self.assertAlmostEqual(q.sum(),1);self.assertTrue((q>0).all())
 def test_forgetting(self):
  c=BoundedCache(64)
  for b in b'abcdefghX'*10+b'z'*100:c.feed(b)
  self.assertTrue(all(set(t)<= {b'z'*k} for k,t in c.tables.items()))
 def test_prefix_only(self):
  a=BoundedCache(64);b=BoundedCache(64)
  for x in b'abc'*50:a.feed(x);b.feed(x)
  np.testing.assert_array_equal(a.distribution()[0],b.distribution()[0])
 def test_invalid(self):
  for n in (0,16,-1):
   with self.assertRaises(ValueError):BoundedCache(n)
if __name__=='__main__':unittest.main()
