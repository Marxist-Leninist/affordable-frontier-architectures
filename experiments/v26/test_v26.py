import unittest
from run import *
class Tests(unittest.TestCase):
 def test_protocol(self):verify()
 def test_intervention(self):
  items=data();saved=json.loads((P/'items.json').read_text());self.assertEqual(len(items),len(saved))
  docs={r['row_idx']:r['row']['text'].encode()[:4097] for r in json.loads((P/'rows.json').read_text())['rows']}
  for i in items:
   self.assertNotEqual(i['doc'],i['donor_doc']);self.assertEqual(i['raw'][1408:],i['changed'][1408:]);self.assertEqual(i['raw'][-128:],i['changed'][-128:]);self.assertEqual(len(i['changed']),2049);self.assertEqual(i['changed'][:1408],docs[i['donor_doc']][:1408]);self.assertEqual(i['raw'][-513:-1],i['changed'][-513:-1])
 def test_causal_and_optimized(self):
  torch.manual_seed(25);x=torch.randint(256,(1,2048));alt=x.clone();alt[:,1985:]=(alt[:,1985:]+1)%256
  with torch.inference_mode():
   for arm in ARMS:
    m=load(arm,1);z=m(x);za=m(alt);self.assertLessEqual(float((z[:,:1985]-za[:,:1985]).abs().max()),1e-5)
    opt,_=local_prefill(m,x,128,prune=True) if arm=='local' else prefill(m,x,None,128)
    self.assertLessEqual(float((opt-z[:,-128:]).abs().max()),1e-4)
 def test_local_invariance_and_suffix_alignment(self):
  items=data()[:2]
  for i in items:self.assertEqual(i['raw'][-513:],i['changed'][-513:])
  m=load('local',1);a=score(m,'local',items,'original2048');b=score(m,'local',items,'donor2048');self.assertLessEqual(float(np.max(np.abs(a-b))),1e-10)
if __name__=='__main__':unittest.main()

class CopyTests(unittest.TestCase):
 def test_subgroup_oracle_and_target_independence(self):
  for item in data():
   raw=item['raw']
   for t in range(1921,2049):
    hist=raw[:t];key=hist[-8:];cut=t-512
    positions=[j for j in range(8,t) if hist[j-8:j]==key]
    expected=any(j<1408 and j<cut for j in positions) and not any(j-8>=cut for j in positions)
    self.assertEqual(subgroup(hist),expected)
    altered=raw[:t]+bytes([(raw[t]+1)%256])+raw[t+1:]
    self.assertEqual(subgroup(altered[:t]),expected)
 def test_compact_reference(self):
  rng=np.random.default_rng(26)
  for raw in [bytes(rng.integers(0,8,2200).tolist()),b'abcd1234X'*245,data()[0]['raw']]:
   a=CopyCache();b=CompactCache()
   for v in raw:
    a.feed(v);b.feed(v)
    qa,na,ka,_=a.distribution();qb,nb,kb=b.distribution()
    self.assertEqual((na,ka),(nb,kb));np.testing.assert_array_equal(qa,qb);self.assertAlmostEqual(qb.sum(),1)
