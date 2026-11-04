"""Independent frozen-checkpoint tests; run only outside scoring/timing workloads."""
import hashlib
import json
import math
import os
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from models import LM
from local_attention import localize, allocated_tensor_bytes
from prefill import prefill
from pruning import local_prefill

P = Path(__file__).parent
V = Path(os.environ.get('V14_DIR', P.parent / 'frontier_research_v14'))
torch.set_num_threads(1)


def frozen(seed):
    path = V / 'checkpoints' / f'transformer_seed{seed}_step4000.pt'
    manifest = json.loads((V / 'checkpoint_manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[path.name]
    model = LM('transformer').eval()
    model.load_state_dict(torch.load(path, weights_only=True))
    return localize(model).eval()


def sample():
    rows = json.loads((P / 'rows.json').read_text())['rows']
    docs = [r['row']['text'].encode()[:4097] for r in rows]
    docs = [d for d in docs if len(d) >= 2081]
    chosen = [docs[0], docs[len(docs)//2], docs[-1]]
    return torch.tensor([list(d[:2081]) for d in chosen])


def offset_prefill(model, tokens, chunk, tail, offset):
    # Independent construction of empty absolute-coordinate state.
    states = []
    for block in model.blocks:
        a = block.mix
        empty = model.emb.weight.new_empty((tokens.shape[0],a.h,0,a.d//a.h))
        states.append((empty, empty.clone(), torch.tensor(offset, dtype=torch.int64)))
    return prefill(model,tokens,chunk,tail,initial_states=states)


def tensors(state):
    if isinstance(state, torch.Tensor):
        return [state]
    return [t for layer in state for t in tensors(layer)]


class PruningTests(unittest.TestCase):
    def parity(self, a, b, targets):
        self.assertLessEqual(float((a-b).abs().max()), 1e-4)
        # Independent double-precision loss expression, not the scoring loop's CE.
        la = -a.double().log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1).mean(1)/math.log(2)
        lb = -b.double().log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1).mean(1)/math.log(2)
        self.assertLessEqual(float((la-lb).abs().max()), 2e-6)

    def test_indexing_and_receptive_field(self):
        n, span, depth, tail = 2048, 128, 2, 128
        receptive = 1 + depth*(span-1)
        retained = tail + receptive - 1
        self.assertEqual((receptive, retained, n-retained), (255,382,1666))
        # Propagate exact discrete dependency sets independently of tensor kernels.
        deps = [set([i]) for i in range(n)]
        for _ in range(depth):
            deps = [set().union(*deps[max(0,i-span+1):i+1]) for i in range(n)]
        combined = set().union(*deps[-tail:])
        self.assertEqual(combined, set(range(1666,2048)))
        raw = torch.arange(2049)[None, :]
        self.assertTrue(torch.equal(raw[:, :-1][:, -128:]+1, raw[:, -128:]))

    @torch.inference_mode()
    def test_real_weights_outputs_states_continuation(self):
        raw = sample(); x = raw[:, :2048]; y = raw[:,1921:2049]
        metrics = []
        for seed in (1,2,3):
            model = frozen(seed)
            before = {k:v.clone() for k,v in model.state_dict().items()}
            full, sf = prefill(model,x,128,128)
            pruned, sp = local_prefill(model,x,128,prune=True)
            self.parity(full,pruned,y)
            for a,b in zip(tensors(sf),tensors(sp)):
                torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-5)
            self.assertTrue(all(int(layer[2])==2048 for layer in sp))
            self.assertEqual(allocated_tensor_bytes(sf),allocated_tensor_bytes(sp))
            future = raw[:,2048:2080]
            af, nf = model(future,sf,cache=True)
            ap, np = model(future,sp,cache=True)
            self.parity(af,ap,raw[:,2049:2081])
            self.assertTrue(all(int(layer[2])==2080 for layer in np))
            for k,v in model.state_dict().items():
                self.assertTrue(torch.equal(v,before[k]))
            metrics.append(dict(seed=seed,max_logit_error=float((full-pruned).abs().max()),max_continuation_error=float((af-ap).abs().max())))
        print('FROZEN_WEIGHT_METRICS',json.dumps(metrics),flush=True)

    @torch.inference_mode()
    def test_independent_monolithic_and_prefix_poison(self):
        model = frozen(1); raw = sample()[:1]; x=raw[:,:2048]; targets=raw[:,1921:2049]
        ref = model(x)[:,-128:]
        for chunk in (None,37,128,511):
            actual,_ = offset_prefill(model,x[:,-382:],chunk,128,1666)
            self.parity(ref,actual,targets)
        poisoned=x.clone();poisoned[:,:1666]=(poisoned[:,:1666]+73)%256
        actual,_ = prefill(model,poisoned,128,128)
        self.parity(ref,actual,targets)

    @torch.inference_mode()
    def test_final_one_and_32_single_updates(self):
        raw=sample();x=raw[:,:2048]
        metrics=[]
        for seed in (1,2,3):
            model=frozen(seed)
            full,sf=local_prefill(model,x,1,prune=False)
            pruned,sp=local_prefill(model,x,1,prune=True)
            all_full=[full];all_pruned=[pruned]
            maxerr=float((full-pruned).abs().max())
            for t in range(32):
                a,sf=model(raw[:,2048+t:2049+t],sf,cache=True)
                b,sp=model(raw[:,2048+t:2049+t],sp,cache=True)
                all_full.append(a);all_pruned.append(b)
                maxerr=max(maxerr,float((a-b).abs().max()))
            self.assertTrue(all(int(layer[2])==2080 for layer in sp))
            a=torch.cat(all_full,1);b=torch.cat(all_pruned,1)
            self.parity(a,b,raw[:,2048:2081])
            targets=raw[:,2048:2081,None]
            la=-a.double().log_softmax(-1).gather(-1,targets).squeeze(-1)/math.log(2)
            lb=-b.double().log_softmax(-1).gather(-1,targets).squeeze(-1)/math.log(2)
            metrics.append(dict(seed=seed,max_final_one_and_updates_error=maxerr,max_33_target_window_bpb_error=float((la.mean(1)-lb.mean(1)).abs().max()),max_single_target_bpb_error=float((la-lb).abs().max())))
        print('FINAL_ONE_METRICS',json.dumps(metrics),flush=True)

    @torch.inference_mode()
    def test_rope_shift_is_numerical_not_bitwise(self):
        model=frozen(1);raw=sample();x=raw[:,:2048]
        original,_=prefill(model,x,128,128)
        reset,_=prefill(model,x[:,-382:],128,128)
        shifted,_=local_prefill(model,x,128,prune=True)
        a=float((original-reset).abs().max());b=float((original-shifted).abs().max())
        self.assertLessEqual(b,1e-4)
        print('ROPE_RESET_CONTROL',json.dumps(dict(reset_max_logit_error=a,preserved_max_logit_error=b)),flush=True)

if __name__=='__main__':
    unittest.main(verbosity=2)
