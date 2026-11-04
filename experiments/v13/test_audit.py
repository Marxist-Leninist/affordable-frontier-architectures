"""Independent v13 invariants; run with PYTHONPATH=/tmp/frontier-v10-packages."""
import unittest, json, math
import numpy as np
import torch
from torch.nn import functional as F
import run
from ngram import Ngram

class Audit(unittest.TestCase):
 def test_target_indexing_and_truncation(self):
  class Known(torch.nn.Module):
   def forward(self,x):
    logits=torch.zeros((*x.shape,256)); logits.scatter_(-1,((x+1)%256)[...,None],2.)
    return logits
  raw=bytes(range(256))+bytes(range(100))
  p=run.neural_probs(Known(),raw,batch=1)
  self.assertEqual(len(p),256)
  np.testing.assert_allclose(p,np.exp(2)/(np.exp(2)+255),rtol=1e-6)
 def test_statistical_context_reset_and_target_order(self):
  class Spy:
   def __init__(self):self.seen=[]
   def prob(self,c,y,o):self.seen.append((c,y,o));return .1
  m=Spy();raw=bytes(range(256))+b'x';p=run.stat_probs(m,raw)
  self.assertEqual(len(p),256)
  for i,(ctx,y,o) in enumerate(m.seen):
   start=(i//128)*128
   self.assertEqual(ctx,raw[max(start,i-3):i+1]);self.assertEqual(y,raw[i+1]);self.assertEqual(o,4)
  self.assertEqual(m.seen[128][0],raw[128:129])
 def test_ngram_frozen_normalized_and_reference_eval(self):
  m=Ngram();m.add(b'abracadabra'*20);raw=bytes(range(256))+b'end'
  before=repr(m.counts)+repr(m.totals)
  p=run.stat_probs(m,raw)
  self.assertAlmostEqual(float(-np.log2(p).mean()),m.eval(raw,4)['bpb'],places=11)
  for ctx in [b'',b'a',b'abra',b'unknown']:
   self.assertAlmostEqual(sum(m.prob(ctx,y,4) for y in range(256)),1.,places=12)
  self.assertEqual(before,repr(m.counts)+repr(m.totals))
 def test_probability_mixture_not_loss_average(self):
  p=np.array([.1,.9]);q=np.array([.9,.1])
  self.assertEqual(run.mix_bpb(p,q,.5),1.)
  self.assertAlmostEqual(run.mix_bpb(p,q,0),float(-np.log2(p).mean()))
  self.assertAlmostEqual(run.mix_bpb(p,q,1),float(-np.log2(q).mean()))
 def test_neural_reset_batch_and_cross_entropy_parity(self):
  raw=bytes(range(256))*2+b'end'
  x=torch.tensor(list(raw[:512])).reshape(4,128)
  y=torch.tensor(list(raw[1:513])).reshape(4,128)
  for kind in ['transformer','gru']:
   m=run.load(kind,1)
   p=run.neural_probs(m,raw,batch=1);pb=run.neural_probs(m,raw,batch=3)
   np.testing.assert_allclose(p,pb,rtol=2e-5,atol=1e-8)
   with torch.inference_mode():
    reference=F.cross_entropy(m(x).flatten(0,1),y.flatten()).item()/math.log(2)
    full=m(x[:1]);_,s=m(x[:1,:127],cache=True);last,_=m(x[:1,127:],s,True)
   self.assertAlmostEqual(float(-np.log2(p).mean()),reference,places=5)
   torch.testing.assert_close(last,full[:,-1:],rtol=2e-5,atol=2e-5)
 def test_frozen_source_data_ledger_checkpoints(self):
  protocol=json.loads((run.P/'protocol.json').read_text())
  for f,h in protocol['source_sha256'].items():self.assertEqual(run.sha(run.P/f),h)
  self.assertEqual(run.sha(run.V12/'train_starts.npy'),protocol['ledger_sha256'])
  self.assertEqual(run.sha(run.V12/'checkpoint_manifest.json'),protocol['v12_checkpoint_manifest_sha256'])
  for f in ['models.py','ngram.py','data_manifest.json']:
   self.assertEqual(run.sha(run.P/f),run.sha(run.V12/f))
  for k,info in protocol['dataset']['splits'].items():self.assertEqual(run.sha(run.D/(k+'.txt')),info['text_sha256'])
  checkpoints=json.loads((run.V12/'checkpoint_manifest.json').read_text())
  for kind in ['gru','transformer']:
   for seed in [101,1,2,3]:
    f=f'{kind}_seed{seed}_lr0.01.pt';self.assertEqual(run.sha(run.V12/'checkpoints'/f),checkpoints[f]['sha256'])
  self.assertEqual(json.loads((run.V12/'selection.json').read_text())['selected'],{'transformer':.01,'gru':.01})
  self.assertEqual(json.loads((run.V12/'ngram_selection.json').read_text())['selected_order'],4)
 def test_completed_result_and_prior_full_corpus_parity(self):
  results=json.loads((run.P/'results.json').read_text());self.assertEqual(len(results['rows']),6)
  prior={(r['kind'],r['seed']):r for r in json.loads((run.V12/'independent_audit_scores.json').read_text()) if 'seed' in r}
  ng=json.loads((run.V12/'ngram_results.json').read_text())
  for kind,selection in results['selection'].items():
   self.assertEqual(selection['selected_weight'],min(selection['grid'],key=lambda r:r['bpb'])['weight'])
   self.assertEqual(len(selection['grid']),11)
  for r in results['rows']:
   self.assertAlmostEqual(r['neural_test_bpb'],prior[r['kind'],r['seed']]['independent_test_bpb'],places=12)
   self.assertAlmostEqual(r['ngram_test_bpb'],ng['test']['bpb'],places=11)
   self.assertEqual(r['test_targets'],1291904)
   self.assertEqual(r['ngram_weight'],results['selection'][r['kind']]['selected_weight'])
   self.assertAlmostEqual(r['gain_vs_ngram_bpb'],r['ngram_test_bpb']-r['hybrid_test_bpb'])
   self.assertLess(r['hybrid_test_bpb'],min(r['neural_test_bpb'],r['ngram_test_bpb']))
   f=run.V12/'checkpoints'/f"{r['kind']}_seed{r['seed']}_lr0.01.pt"
   self.assertEqual(r['combined_serialized_bytes'],f.stat().st_size+(run.V12/'checkpoints/ngram.pkl').stat().st_size)
 def test_ngram_training_target_count(self):
  m=Ngram();m.add(bytes(range(129)));self.assertEqual(m.totals[0][b''],128)
  self.assertEqual(sum(m.totals[1].values()),128)
  self.assertEqual(sum(m.totals[4].values()),125)

if __name__=='__main__':unittest.main(verbosity=2)
