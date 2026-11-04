"""Independent synthetic checks for output-tail-only and chunked prefill."""
import unittest,hashlib,json
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from models import LM
from local_attention import localize,allocated_tensor_bytes
from prefill import prefill
P=Path(__file__).parent
torch.set_num_threads(1)

def tensors(s):
 if isinstance(s,torch.Tensor):return [s]
 return [t for x in s for t in tensors(x)]

class Tests(unittest.TestCase):
 def model(self,kind):
  torch.manual_seed(17);m=LM('gru' if kind=='gru' else 'transformer').eval()
  return localize(m).eval() if kind=='local' else m
 def test_tail_across_partial_chunks(self):
  for kind in ['transformer','local','gru']:
   m=self.model(kind);x=torch.randint(256,(2,301))
   with torch.inference_mode():
    full=m(x)
    for tail in [1,128,301]:
     for chunk in [None,37,128,500]:
      z,_=prefill(m,x,chunk,tail)
      self.assertEqual(tuple(z.shape),(2,tail,256));torch.testing.assert_close(z,full[:,-tail:],rtol=2e-5,atol=2e-5)
 def test_states_and_continuation(self):
  for kind in ['transformer','local','gru']:
   m=self.model(kind);x=torch.randint(256,(1,333));future=torch.randint(256,(1,33))
   with torch.inference_mode():
    _,ref=m(x,cache=True);_,s=prefill(m,x,128,1)
    for a,b in zip(tensors(ref),tensors(s)):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5)
    a,sa=m(future,ref,True);b,sb=m(future,s,True);torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5)
    if kind=='local':
     self.assertEqual(allocated_tensor_bytes(s),130064)
     self.assertTrue(all(int(layer[2])==333 for layer in s))
     self.assertTrue(all(int(layer[2])==366 for layer in sb))
 def test_single_head_call_and_weights_unchanged(self):
  for kind in ['transformer','local','gru']:
   m=self.model(kind);before={k:v.clone() for k,v in m.state_dict().items()};calls=[]
   hook=m.head.register_forward_pre_hook(lambda mod,args:calls.append(tuple(args[0].shape)))
   with torch.inference_mode():prefill(m,torch.randint(256,(2,301)),37,128)
   hook.remove();self.assertEqual(calls,[(2,128,64)])
   for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
 def test_mask_shape_bound(self):
  m=self.model('local');x=torch.randint(256,(1,2048));original=F.scaled_dot_product_attention
  def trace(chunk):
   shapes=[]
   def wrapper(*args,**kwargs):
    mask=kwargs['attn_mask'];shapes.append(tuple(mask.shape));self.assertEqual(mask.dtype,torch.bool)
    return original(*args,**kwargs)
   with torch.inference_mode(),patch('torch.nn.functional.scaled_dot_product_attention',wrapper):prefill(m,x,chunk,1)
   return shapes
  mono=trace(None);chunk=trace(128)
  self.assertEqual(mono,[(2048,2048),(2048,2048)]);self.assertEqual(len(chunk),32)
  self.assertEqual(max(a*b for a,b in chunk),128*255)
  self.assertEqual(chunk[:2],[(128,128),(128,128)])
 def test_tail_target_alignment(self):
  raw=torch.arange(513).remainder(256)[None,:];x=raw[:,:-1];y=raw[:,-128:]
  self.assertTrue(torch.equal(y,(x[:,-128:]+1)%256))
  raw=torch.arange(2049).remainder(256)[None,:]
  self.assertTrue(torch.equal(raw[:,-128:],(raw[:,:-1][:,-128:]+1)%256))
 def test_invalid(self):
  m=self.model('local');x=torch.ones((1,3),dtype=torch.long)
  for tail in [0,4,1.5]:
   with self.assertRaises(ValueError):prefill(m,x,output_tail=tail)
  for chunk in [0,-1,2.5]:
   with self.assertRaises(ValueError):prefill(m,x,chunk_size=chunk)
  with self.assertRaises(ValueError):prefill(m,x[:,:0])
 def test_snapshot_protocol(self):
  mf=json.loads((P/'data_manifest.json').read_text());raw=(P/'rows.json').read_bytes();j=json.loads(raw)
  self.assertEqual(hashlib.sha256(raw).hexdigest(),mf['response_sha256']);self.assertFalse(j['partial']);self.assertFalse(any(r['truncated_cells'] for r in j['rows']))
  self.assertEqual([r['row_idx'] for r in j['rows']],list(range(200,300)))
  p=json.loads((P/'protocol.json').read_text())
  for f,h in p['source_hashes'].items():self.assertEqual(hashlib.sha256((P/f).read_bytes()).hexdigest(),h)
if __name__=='__main__':unittest.main(verbosity=2)
