#!/usr/bin/env python3
"""Check that the 'tfplain' speed control computes the same function as the Transformer arm: same weights, forward and
token-by-token step outputs compared on random byte sequences. Usage: check_tfplain.py OUT_JSON"""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import om3bench  # noqa: F401
import om2bench as ob
torch.set_num_threads(1)
torch.manual_seed(0)
a = ob.build('tf'); b = ob.build('tfplain'); b.load_state_dict(a.state_dict())
x = torch.randint(0, 256, (2, 256))
with torch.no_grad():
    a.eval(); b.eval()
    fwd = float((a(x, hard=True) - b(x, hard=True)).abs().max())
    sa = sb = None; stp = 0.0
    for t in range(x.shape[1]):
        la, sa = a.step(x[:, t], sa, hard=True); lb, sb = b.step(x[:, t], sb, hard=True)
        stp = max(stp, float((la - lb).abs().max()))
out = dict(forward_max_abs_logit_diff=fwd, step_max_abs_logit_diff=stp, sequences=2, length=256, torch=torch.__version__,
           plain_attention_modules=sum(type(m).__name__ == 'PlainAttn' for m in b.modules()))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out))
