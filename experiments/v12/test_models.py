import unittest,torch,tempfile
from models import LM,count,state_bytes

torch.set_num_threads(1)
class Tests(unittest.TestCase):
 def test_parameters(self):
  a,b=[count(LM(k)) for k in ('transformer','gru')];self.assertLess(abs(a-b)/a,.01)
 def test_causality_and_streaming(self):
  for k in ('transformer','gru'):
   torch.manual_seed(13);m=LM(k).eval();x=torch.randint(256,(2,33));other=x.clone();other[:,17:]=torch.randint(256,(2,16))
   with torch.no_grad():
    a=m(x);self.assertTrue(torch.allclose(a[:,:17],m(other)[:,:17],atol=2e-6))
    s=None;parts=[]
    for start,end in [(0,8),(8,9),(9,21),(21,33)]:z,s=m(x[:,start:end],s,True);parts.append(z)
    self.assertTrue(torch.allclose(a,torch.cat(parts,1),atol=2e-6));self.assertGreater(state_bytes(s),0)
    self.assertTrue(torch.allclose(a[:1],m(x[:1]),atol=2e-6))
 def test_gradients_reload(self):
  for k in ('transformer','gru'):
   m=LM(k);x=torch.randint(256,(2,16));y=torch.randint(256,(2,16));torch.nn.functional.cross_entropy(m(x).flatten(0,1),y.flatten()).backward()
   self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters()))
   with tempfile.NamedTemporaryFile() as f:
    torch.save(m.state_dict(),f.name);n=LM(k);n.load_state_dict(torch.load(f.name,weights_only=True));self.assertTrue(torch.equal(m(x),n(x)))
if __name__=='__main__':unittest.main(verbosity=2)
