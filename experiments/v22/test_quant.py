import unittest
from run import *
class Checks(unittest.TestCase):
 def test_selection_keys(self):
  frozen=json.loads((P/'frozen_selection.json').read_text())
  for kind in KINDS:self.assertIn('transformer_full2048' if kind=='transformer' else kind,frozen)
 def test_manifest(self):
  items,docs=data();self.assertEqual(len(items),91);self.assertEqual(len(docs),8)
 def test_packed_and_fp32(self):
  for kind in KINDS:
   m=load(kind,1,True);info=tensor_inventory(m);self.assertTrue(info['modules']);self.assertEqual(m.emb.weight.dtype,torch.float32)
   self.assertTrue(all(w['dtype']=='torch.qint8' for w in info['weights'] if w['quantized']))
 def test_causal_suffix_and_normalization(self):
  raw=data()[0][0]['raw']
  for kind in KINDS:
   for quant in (False,True):
    m=load(kind,1,quant);p,s=predict(m,kind,raw);changed=raw[:2000]+bytes((b+1)%256 for b in raw[2000:]);q,_=predict(m,kind,changed)
    np.testing.assert_array_equal(p[:80],q[:80]);np.testing.assert_allclose(p.sum(-1),1,atol=1e-12);self.assertTrue(np.isfinite(p).all())
 def test_float_reference(self):
  # The causal FP32 route agrees with historical full-sequence scoring.
  old=np.load(OLD/'neural_probs.npz');raw=data()[0][0]['raw']
  for kind in ('local','gru','transformer_reset512'):
   p,_=predict(load(kind,1),kind,raw);target=np.frombuffer(raw[-128:],dtype=np.uint8);a=p[np.arange(128),target]
   np.testing.assert_allclose(a,old[f'{kind}_1'][0],atol=1e-6,rtol=1e-4)
if __name__=='__main__':unittest.main()
