"""Independent bounded v14 audit checks; no fitting or full-corpus rescore."""
import ast,copy,hashlib,json,math,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import run
from ngram import Ngram

class KnownLogits(nn.Module):
    def __init__(self):
        super().__init__();self.calls=[]
    def forward(self,x):
        self.calls.append(x.clone())
        logits=torch.zeros(*x.shape,256)
        logits.scatter_(-1,((x+1)%256).unsqueeze(-1),2.)
        return logits

class NeuralSpy(nn.Module):
    def __init__(self):
        super().__init__();self.calls=[]
    def forward(self,x,states=None,cache=False):
        self.calls.append((x.clone(),None if states is None else int(states.item())))
        length=x.shape[1]+(0 if states is None else int(states.item()))
        return torch.zeros(1,x.shape[1],256),torch.tensor(length)

class StatisticalSpy:
    order=4
    def __init__(self):self.calls=[]
    def prob(self,context,y):
        if y==0:self.calls.append(context)
        return 1/256

class Audit(unittest.TestCase):
    def test_evaluation_target_alignment_and_truncation(self):
        # Exactly two windows; 13 trailing target bytes deliberately excluded.
        arr=torch.arange(270)%256; model=KnownLogits();result=run.evaluate(model,arr)
        expected=(math.log(math.exp(2)+255)-2)/math.log(2)
        self.assertAlmostEqual(result['bpb'],expected,places=6)
        self.assertEqual(result['targets'],256)
        self.assertEqual(result['unscored_trailing_targets'],13)
        self.assertEqual(len(result['block_bpb']),2)
        self.assertTrue(torch.equal(model.calls[0],arr[:256].reshape(2,128)))

    def test_evaluation_batch_boundary(self):
        arr=torch.arange(128*33+8)%256;model=KnownLogits();result=run.evaluate(model,arr)
        self.assertEqual([x.shape[0] for x in model.calls],[32,1])
        self.assertTrue(torch.equal(model.calls[1][0],arr[4096:4224]))
        self.assertEqual(result['targets'],4224)
        self.assertEqual(result['unscored_trailing_targets'],7)

    def test_ngram_eval_exact_context_and_targets(self):
        raw=bytes(range(256))+bytes(range(20));m=Ngram();observed=[]
        def spy(context,y,order=None):observed.append((context,y));return 1/256
        m.prob=spy;result=m.eval(raw,4)
        self.assertEqual(result['bpb'],8.)
        self.assertEqual(len(observed),256)
        for i,(context,y) in enumerate(observed):
            start=(i//128)*128
            self.assertEqual(context,raw[max(start,i-3):i+1])
            self.assertEqual(y,raw[i+1])
        self.assertEqual(observed[128][0],raw[128:129])
        self.assertEqual(result['unscored_trailing_targets'],19)

    def test_ngram_ledger_reset_and_normalization(self):
        raw=bytes(range(140));m=Ngram();starts=[0,1,0]
        for s in starts:m.add(raw[s:s+129])
        self.assertEqual(m.totals[0][b''],384)
        # Every window's first target has only one observed context byte.
        self.assertEqual(sum(m.totals[4].values()),3*125)
        before=copy.deepcopy((m.counts,m.totals))
        for context in [b'',b'abc',bytes(range(12)),b'\xff'*4]:
            self.assertAlmostEqual(sum(m.prob(context,y) for y in range(256)),1.,places=12)
        m.eval(raw,4)
        self.assertEqual((m.counts,m.totals),before)

    def test_timing_semantics_and_identical_sequences(self):
        raw=bytes(range(256))*2; neural=NeuralSpy();stat=StatisticalSpy()
        # Avoid relying on performance of fake model: inspect calls, not timing values.
        a=run.benchmark(neural,raw);b=run.ng_benchmark(stat,raw)
        self.assertEqual(len(neural.calls),8*5*33)
        self.assertEqual(len(stat.calls),8*5*33)
        for segment,offset in enumerate(np.linspace(0,len(raw)-161,8,dtype=int)):
            for rep in range(5):
                base=(segment*5+rep)*33
                prefix,state=neural.calls[base]
                self.assertIsNone(state)
                self.assertEqual(bytes(prefix.flatten().tolist()),raw[offset:offset+128])
                self.assertEqual(stat.calls[base],raw[offset+124:offset+128])
                for j in range(32):
                    token,state=neural.calls[base+j+1]
                    self.assertEqual(state,128+j)
                    self.assertEqual(token.item(),raw[offset+128+j])
                    self.assertEqual(stat.calls[base+j+1],raw[offset+125+j:offset+129+j])
        for row in [a,b]:
            self.assertEqual(row['contexts'],8)
            self.assertEqual(row['repetitions_after_warmup'],3)
            self.assertEqual(row['decode_steps'],32)
            self.assertGreater(row['end_to_end_p50_ms'],0)

    def test_actual_gate_selection_and_checkpoint_set(self):
        # Execute actual selection AST without fitting; this detects last/best-vs-first changes.
        source=ast.parse((Path(__file__).parent/'run.py').read_text())
        assignment=next(node for node in ast.walk(source) if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Constant)
                    and t.slice.value=='validation_gate_steps' for t in node.targets))
        row={'curve':[{'step':1000,'validation':{'bpb':2.35}},
                      {'step':2000,'validation':{'bpb':2.4}},
                      {'step':4000,'validation':{'bpb':2.25}}]}
        env={'row':row,'GATES':[2.5,2.35,2.25,2.0]}
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<actual-gate>','exec'),env)
        self.assertEqual(row['validation_gate_steps'],{'2.5':1000,'2.35':1000,'2.25':4000,'2.0':None})
        checkpoint_loop=next(node for node in ast.walk(source) if isinstance(node,ast.For)
            and isinstance(node.target,ast.Name) and node.target.id=='step'
            and isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name)
            and node.iter.func.id=='sorted')
        chosen=eval(compile(ast.Expression(body=checkpoint_loop.iter),'<actual-checkpoints>','eval'),
                    {'row':row,'STEPS':4000})
        self.assertEqual(chosen,[1000,4000])
        # Only pilot final validation determines the LR, irrespective of earlier loss.
        lr_assignment=next(node for node in ast.walk(source) if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name)
                    and t.value.id=='selected' for t in node.targets))
        env={'kind':'transformer','selected':{},'tuning':[
            {'kind':'transformer','lr':.003,'curve':[{'validation':{'bpb':1.}}, {'validation':{'bpb':3.}}]},
            {'kind':'transformer','lr':.01,'curve':[{'validation':{'bpb':4.}}, {'validation':{'bpb':2.}}]},
            {'kind':'gru','lr':.03,'curve':[{'validation':{'bpb':.1}}]}]}
        exec(compile(ast.Module(body=[lr_assignment],type_ignores=[]),'<actual-lr>','exec'),env)
        self.assertEqual(env['selected'],{'transformer':.01})

    def test_protocol_source_and_ledger_hashes(self):
        p=Path(__file__).parent;protocol=json.loads((p/'protocol.json').read_text())
        for filename,digest in protocol['source_hashes'].items():
            self.assertEqual(hashlib.sha256((p/filename).read_bytes()).hexdigest(),digest)
        manifest=json.loads((p/'data_manifest.json').read_text())
        # Independently regenerate complete target-position ledger.
        n=manifest['splits']['train']['utf8_bytes']
        expected=np.random.default_rng(14112026).integers(0,n-128,size=(4000,8),dtype=np.int64)
        actual=np.load(p/'train_starts.npy')
        self.assertTrue(np.array_equal(actual,expected))
        self.assertEqual(protocol['training_targets'],actual.size*128)
        self.assertEqual(protocol['training_targets'],4096000)
        self.assertEqual(hashlib.sha256((p/'train_starts.npy').read_bytes()).hexdigest(),protocol['ledger_sha256'])

if __name__=='__main__':unittest.main(verbosity=2)
