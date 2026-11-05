import unittest
import numpy as np
import torch
from run import *
class TestV23(unittest.TestCase):
 def test_ledger(self):
  d,_=data();s=np.load(P/'train_starts.npy');self.assertEqual(s.shape,(250,));self.assertEqual(250*2048,512000)
  for start in s:
   ix=torch.arange(2048)[None,:]+int(start);x=d[ix];y=d[ix+1]
   self.assertTrue(torch.equal(x.flatten(),x.reshape(4,512).flatten()));self.assertTrue(torch.equal(y.flatten(),y.reshape(4,512).flatten()));self.assertTrue(torch.equal(x[:,1:],y[:,:-1]))
 def test_initial_identity(self):
  for seed in (1,2,3):
   a=load('transformer',seed);b=load('transformer',seed)
   self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
 def test_causal(self):
  torch.manual_seed(23);m=load('transformer',1).eval();x=torch.randint(0,256,(1,64));y=x.clone();y[:,33:]=(y[:,33:]+1)%256
  with torch.no_grad():self.assertLess(float((m(x)[:,:33]-m(y)[:,:33]).abs().max()),1e-5)
 def test_optimized_scoring(self):
  _,items=data();raw=torch.tensor([list(items[0]['raw'])]);m=load('transformer',1).eval()
  with torch.no_grad():
   for length in (512,2048):
    x=raw[:,:-1][:,-length:];a=m(x)[:,-128:];b,_=prefill(m,x,None,128)
    self.assertLess(float((a-b).abs().max()),1e-4)
 def test_nested_targets(self):
  _,items=data()
  for i in items:
   r=i['raw'];self.assertEqual(r[-513:][-128:],r[-128:]);self.assertEqual(len(r),2049)
 def test_protocol(self):
  p=json.loads((P/'protocol.json').read_text())
  for f,h in p['source_sha256'].items():self.assertEqual(sha(P/f),h)
if __name__=='__main__':unittest.main()
