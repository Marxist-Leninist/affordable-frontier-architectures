import unittest,json,hashlib,math
from pathlib import Path
import numpy as np
import torch
import run
from ngram import Ngram
P=Path(__file__).parent
class Oracle(torch.nn.Module):
 def __init__(self):super().__init__();self.calls=[]
 def forward(self,x):
  self.calls.append(x.clone());z=torch.zeros(*x.shape,256);z.scatter_(-1,((x+1)%256)[...,None],4.);return z
class Spy:
 order=4
 def __init__(self):self.calls=[]
 def prob(self,c,y):self.calls.append((c,y));return 1/256
class Tests(unittest.TestCase):
 def test_windows_document_boundaries(self):
  docs=[bytes(range(129)),b'a'*128,b'b'*257]
  w=run.windows(docs,128)
  self.assertEqual([(d,s,len(r)) for d,s,r in w],[(0,0,129),(2,0,129),(2,128,129)])
  self.assertEqual(w[-1][2],b'b'*129)
 def test_neural_alignment_tail_and_batch_boundary(self):
  docs=[bytes(i%256 for i in range(1025)) for _ in range(17)]
  long=run.windows(docs,512);short=[(d,s,r[-129:]) for d,s,r in long]
  a,b=Oracle(),Oracle();sa=run.neural_scores(a,short);lb=run.neural_scores(b,long)
  expected=math.log2(1+255*math.exp(-4))
  self.assertEqual(len(sa),34)
  for x,y in zip(sa,lb):
   self.assertAlmostEqual(x['bpb'],expected,places=5);self.assertAlmostEqual(x['bpb'],y['bpb'],places=6)
   self.assertEqual((x['doc'],x['start']),(y['doc'],y['start']))
  for x,y in zip(a.calls,b.calls):self.assertTrue(torch.equal(x,y[:,-128:]))
 def test_stat_alignment_context_reset(self):
  docs=[bytes(i%256 for i in range(1025)),b'z'*513];long=run.windows(docs,512);short=[(d,s,r[-129:]) for d,s,r in long]
  for items in [short,long]:
   m=Spy();scores=run.stat_scores(m,items);expected=[]
   for d,s,raw in items:
    for pos in range(len(raw)-129,len(raw)-1):expected.append((raw[max(0,pos-3):pos+1],raw[pos+1]))
   self.assertEqual(m.calls,expected);self.assertTrue(all(x['bpb']==8 for x in scores))
 def test_ngram_extra_context_only_first_three(self):
  m=Ngram();m.add(b'abracadabra '*100);raw=(b'abracadabra '*50)[:513];short=raw[-129:]
  for j in range(3,128):
   p=m.prob(short[j-3:j+1],short[j+1]);pos=384+j
   self.assertEqual(p,m.prob(raw[pos-3:pos+1],raw[pos+1]))
  before=[dict(x) for x in m.totals];run.stat_scores(m,[(0,0,raw)]);self.assertEqual(before,[dict(x) for x in m.totals])
  self.assertAlmostEqual(sum(m.prob(b'abra',i) for i in range(256)),1)
 def test_snapshot_and_source_identity(self):
  proto=json.loads((P/'protocol.json').read_text());mf=json.loads((P/'data_manifest.json').read_text())
  for f,h in proto['source_hashes'].items():self.assertEqual(hashlib.sha256((P/f).read_bytes()).hexdigest(),h)
  raw=(P/'rows.json').read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),mf['response_sha256']);j=json.loads(raw)
  self.assertFalse(j['partial']);self.assertEqual([x['row_idx'] for x in j['rows']],list(range(100)));self.assertFalse(any(x['truncated_cells'] for x in j['rows']))
 def test_pair_and_aggregation_weights(self):
  a=[dict(doc=0,start=0,bpb=3,targets=128),dict(doc=0,start=512,bpb=3,targets=128),dict(doc=1,start=0,bpb=3,targets=128)]
  b=[dict(x,bpb=x['bpb']+d) for x,d in zip(a,[-1,-1,2])];r=run.pair_summary(a,b)
  self.assertEqual(r['long_minus_short_bpb'],0);self.assertEqual(r['equal_document_mean_delta_bpb'],.5)
  self.assertAlmostEqual(run.aggregate(a)['bpb'],3)
  b[0]['start']=1
  with self.assertRaises(AssertionError):run.pair_summary(a,b)
if __name__=='__main__':unittest.main(verbosity=2)
