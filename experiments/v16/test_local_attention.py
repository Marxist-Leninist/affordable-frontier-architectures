"""Independent bounded correctness checks for the v16 inference intervention."""
import unittest,torch,math,json,hashlib
from pathlib import Path
from models import LM,Attention
from local_attention import LocalAttention,localize,tensor_bytes,allocated_tensor_bytes
import run
P=Path(__file__).parent
torch.set_num_threads(1)

def reference_attention(m,x,window,offset=0):
 b,t,d=x.shape;q,k,v=m.qkv(x).chunk(3,-1)
 q,k,v=[a.reshape(b,t,m.h,d//m.h).transpose(1,2).double() for a in (q,k,v)]
 positions=torch.arange(offset,offset+t,dtype=torch.float64)
 freq=torch.exp(-math.log(10000.)*torch.arange(0,d//m.h,2,dtype=torch.float64)/(d//m.h))
 angles=positions[:,None]*freq[None,:]
 def rotate(a):
  out=torch.empty_like(a);out[...,0::2]=a[...,0::2]*angles.cos()-a[...,1::2]*angles.sin();out[...,1::2]=a[...,0::2]*angles.sin()+a[...,1::2]*angles.cos();return out
 q,k=rotate(q),rotate(k);scores=q@k.transpose(-2,-1)/math.sqrt(d//m.h)
 pos=torch.arange(t);valid=(pos[None,:]<=pos[:,None])&(pos[None,:]>=pos[:,None]-(window-1))
 scores=scores.masked_fill(~valid,float('-inf'));z=(scores.softmax(-1)@v).transpose(1,2).reshape(b,t,d)
 return torch.nn.functional.linear(z,m.out.weight.double())

class Tests(unittest.TestCase):
 def test_weights_and_short_context_parity(self):
  torch.manual_seed(16);base=LM('transformer').eval();local=localize(base).eval()
  self.assertEqual(set(base.state_dict()),set(local.state_dict()))
  self.assertTrue(all(torch.equal(v,local.state_dict()[k]) for k,v in base.state_dict().items()))
  with torch.no_grad():
   for n in [1,2,31,127,128]:
    x=torch.randint(256,(2,n));self.assertTrue(torch.allclose(base(x),local(x),atol=2e-5,rtol=2e-5))
 def test_independent_local_attention_reference(self):
  torch.manual_seed(17);m=LocalAttention(d=16,heads=2,window=8).eval();x=torch.randn(2,31,16)
  with torch.no_grad():
   actual,_=m(x);expected=reference_attention(m,x,8)
  self.assertLess(float((actual.double()-expected).abs().max()),5e-7)
 def test_single_layer_span_and_causality(self):
  torch.manual_seed(18);m=LocalAttention(d=16,heads=2,window=8).eval();x=torch.randn(1,31,16)
  with torch.no_grad():
   a,_=m(x);past=x.clone();past[:,:23]+=100;b,_=m(past)
   self.assertTrue(torch.equal(a[:,30],b[:,30]))
   future=x.clone();future[:,16:]+=100;c,_=m(future)
   self.assertTrue(torch.equal(a[:,:16],c[:,:16]))
 def test_full_chunked_and_trimmed_absolute_offsets(self):
  torch.manual_seed(19);m=localize(LM('transformer')).eval();x=torch.randint(256,(1,769))
  with torch.no_grad():
   full=m(x);state=None;parts=[];end=0
   for n in [64,64,1,127,256,257]:
    z,state=m(x[:,end:end+n],state,True);parts.append(z);end+=n
    for k,v,position in state:
     self.assertEqual(position.item(),end);self.assertEqual(k.shape[-2],min(end,127));self.assertEqual(v.shape[-2],min(end,127))
   self.assertEqual(end,769);self.assertTrue(torch.allclose(full,torch.cat(parts,1),atol=6e-5,rtol=2e-5))
 def test_cache_allocations_are_bounded_not_views(self):
  torch.manual_seed(20);m=localize(LM('transformer')).eval()
  with torch.no_grad():
   _,state=m(torch.randint(256,(2,1024)),cache=True)
   for k,v,offset in state:
    self.assertEqual(offset.item(),1024)
    for a in [k,v]:
     self.assertEqual(a.shape[-2],127);self.assertIsNone(a._base);self.assertEqual(a.storage_offset(),0)
     self.assertEqual(a.untyped_storage().nbytes(),a.numel()*a.element_size())
   expected=2*(2*2*4*127*16*4+8)
   self.assertEqual(tensor_bytes(state),expected);self.assertEqual(allocated_tensor_bytes(state),expected)
 def test_window_one_and_invalid(self):
  m=LocalAttention(d=16,heads=2,window=1).eval();x=torch.randn(1,7,16)
  with torch.no_grad():
   a,s=m(x,cache=True);parts=[];state=None
   for i in range(7):z,state=m(x[:,i:i+1],state,True);parts.append(z)
   self.assertTrue(torch.allclose(a,torch.cat(parts,1),atol=1e-6));self.assertEqual(s[0].numel(),0);self.assertEqual(s[0].untyped_storage().nbytes(),0);self.assertEqual(s[2].item(),7)
  for window in [0,-1,1.5]:
   with self.assertRaises(ValueError):LocalAttention(window=window)
 def test_target_windows_and_tail_alignment(self):
  docs=[bytes(i%256 for i in range(1025)),b'a'*512,b'b'*513];long=run.windows(docs,512);short=[(i,s,r[-129:]) for i,s,r in long]
  self.assertEqual([(i,s) for i,s,_ in long],[(0,0),(0,512),(2,0)])
  for (_,_,a),(_,_,b) in zip(short,long):self.assertEqual(a[1:],b[-128:])
 def test_statistical_target_and_reset_context(self):
  class Spy:
   def __init__(self):self.calls=[]
   def prob(self,c,y):self.calls.append((c,y));return .00390625
  docs=[bytes(i%256 for i in range(1025)),b'a'*513];items=run.windows(docs,512);m=Spy();r=run.statscore(m,items);expected=[]
  for _,_,raw in items:
   for pos in range(len(raw)-129,len(raw)-1):expected.append((raw[max(0,pos-3):pos+1],raw[pos+1]))
  self.assertEqual(m.calls,expected);self.assertTrue(all(x['bpb']==8 for x in r));self.assertEqual([x['doc'] for x in r],[100,100,101])
 def test_frozen_sources_and_fresh_snapshot(self):
  protocol=json.loads((P/'protocol.json').read_text());mf=json.loads((P/'data_manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  for f,h in protocol['source_hashes'].items():self.assertEqual(sha(P/f),h)
  self.assertEqual(sha(P/'data_manifest.json'),protocol['data_manifest_sha256']);self.assertEqual(sha(P/'rows.json'),mf['response_sha256'])
  rows=json.loads((P/'rows.json').read_text());self.assertFalse(rows['partial']);self.assertEqual([x['row_idx'] for x in rows['rows']],list(range(100,200)));self.assertFalse(any(x['truncated_cells'] for x in rows['rows']))

if __name__=='__main__':unittest.main(verbosity=2)
