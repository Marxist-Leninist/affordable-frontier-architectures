#!/usr/bin/env python3
"""Export an OM2-study 'm2' run as a standard transformers Mamba-2 checkpoint.

The OM2 study trained 'm2' with its own single-group copy of the chunked kernel (om2bench.py: OM2Mixer with
signed=False, bcnorm=False), whose parameters map one for one onto transformers' Mamba2ForCausalLM. The export is
checked by comparing logits of (a) the study model, (b) the exported checkpoint reloaded from disk with stock
transformers, (c) the same reloaded checkpoint with patch_mamba2.
Usage: export_hf.py --study STUDY_DIR --run m2-5617 --out checkpoints/m2-seed5617
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch
from transformers import Mamba2Config, Mamba2ForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mamba2_cpu_ssd as fast  # noqa: E402

WINDOWS = 48


def config():
    return Mamba2Config(vocab_size=256, hidden_size=128, state_size=32, num_heads=8, head_dim=32, expand=2, n_groups=1,
                        conv_kernel=4, num_hidden_layers=4, chunk_size=32, use_cache=True, tie_word_embeddings=True,
                        pad_token_id=0, bos_token_id=0, eos_token_id=0)


def to_hf(sd, layers=4):
    out = {'backbone.embeddings.weight': sd['e.weight'], 'lm_head.weight': sd['e.weight'],
           'backbone.norm_f.weight': sd['n.w']}
    for i in range(layers):
        p = f'backbone.layers.{i}.'
        out[p + 'norm.weight'] = sd[f'norms.{i}.w']
        for ours, theirs in (('inp.weight', 'in_proj.weight'), ('conv.weight', 'conv1d.weight'),
                             ('conv.bias', 'conv1d.bias'), ('dt_bias', 'dt_bias'), ('A_log', 'A_log'),
                             ('Dskip', 'D'), ('gn.w', 'norm.weight'), ('out.weight', 'out_proj.weight')):
            out[p + 'mixer.' + theirs] = sd[f'mix.{i}.{ours}']
    assert len(out) == len(sd) + 1, (len(out), len(sd))
    return out


CARD = """---
library_name: transformers
tags: [mamba2, byte-level, research, cpu]
---
# Byte-level Mamba-2, {params:,} parameters (OM2 study, seed {seed})

A tiny research checkpoint in standard transformers `Mamba2ForCausalLM` format. Tokens are raw bytes: token id = byte
value, vocabulary 256, no tokenizer. Architecture and initialisation are exactly what transformers 4.57.6 builds for
this config: 4 Mamba-2 layers, hidden 128, 8 heads x 32, state 32, 1 group, conv 4, chunk 32, tied embeddings.

It was trained on one CPU thread by the OM2 study (`om2bench.py`, arm `m2`), which runs the same chunked SSD kernel as
`../../mamba2_cpu_ssd.py` in its own single-group copy. On {windows} held-out windows, this checkpoint's logits with
`patch_mamba2` are identical to the study model's (max abs difference {d_patch:.1e}), and stock transformers gives
the same logits to {d_stock:.1e}.

- Training: {steps:,} updates x {batch} sequences x {ctx} bytes ({bp:,} byte presentations) from the "P056" corpus
  (WikiText-2 encyclopedia articles, public-domain Project Gutenberg fiction, OpenAssistant oasst1 dialogue; see
  `../../THIRD_PARTY_NOTICES.md`); AdamW lr 3e-3, warmup 100, cosine to 10%, weight decay 0.01 on matrices, gradient
  clip 1.0.
- Held-out test (the corpus's 303 test windows; every scored byte sees 129-256 bytes of context):
  **{macro:.4f} bits/byte** macro average (article {article:.4f}, fiction {fiction:.4f}, dialogue {dialogue:.4f}).

```python
import torch
from transformers import Mamba2ForCausalLM
m = Mamba2ForCausalLM.from_pretrained("ScAGI/affordable-frontier-architectures",
                                      subfolder="experiments/2026-11-22/mamba2-cpu-ssd/checkpoints/{name}")
ids = torch.tensor([list("The history of ".encode())])
torch.manual_seed(0)
out = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=80, do_sample=True, top_k=20)
print(bytes(out[0].tolist()).decode("utf-8", "replace"))
```

Greedy decoding (`do_sample=False`) falls into loops such as "the state of the state of"; sampling reads better.
A half-million-parameter byte model learns spelling and local structure, not facts. For research comparison only.

Saving after loading: transformers 4.57.6 declares no tied weights for `Mamba2ForCausalLM`, so
`save_pretrained()` with safetensors refuses the shared embedding/head tensor. Give the head its own copy first
(`m.lm_head.weight = torch.nn.Parameter(m.backbone.embeddings.weight.detach().clone())`) or pass
`safe_serialization=False`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, help='OM2 study directory (contains om2bench.py, data/, runs/)')
    ap.add_argument('--run', required=True, help='run folder name, e.g. m2-5617')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.study)
    import om2bench as ob
    run = Path(a.study) / 'runs' / a.run
    res = json.loads((run / 'result.json').read_text())
    assert res['arm'] == 'm2' and res['status'] == 'ok', res.get('status')
    sd = torch.load(run / 'weights.pt', map_location='cpu')
    ours = ob.SSMLM([ob.OM2Mixer(signed=False, bcnorm=False) for _ in range(4)])
    ours.load_state_dict(sd, strict=True)
    hf = Mamba2ForCausalLM(config())
    hf.load_state_dict(to_hf(sd), strict=True)
    # transformers 4.57.6 declares no tied keys for Mamba2ForCausalLM, so safetensors refuses the shared
    # embedding/head tensor. Store the head as its own copy; the config still ties the two on load.
    hf.lm_head.weight = torch.nn.Parameter(hf.backbone.embeddings.weight.detach().clone())
    out = Path(a.out)
    hf.save_pretrained(out, safe_serialization=True)
    back = Mamba2ForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()
    patched = Mamba2ForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()
    fast.patch_mamba2(patched)
    _, _, test, _ = ob.load_all()
    idx = np.linspace(0, len(test) - 1, WINDOWS).round().astype(int)
    x = torch.from_numpy(np.stack([test[i]['data'][:256] for i in idx]))
    with torch.no_grad():
        ours.eval()
        l0, l1, l2 = ours(x), back(input_ids=x, use_cache=False).logits, patched(input_ids=x, use_cache=False).logits
    check = dict(study_vs_transformers_max_abs=float((l0 - l1).abs().max()),
                 study_vs_patched_max_abs=float((l0 - l2).abs().max()), windows=int(x.shape[0]))
    assert check['study_vs_transformers_max_abs'] < 1e-4 and check['study_vs_patched_max_abs'] < 1e-4, check
    t = res['test']
    (out / 'README.md').write_text(CARD.format(params=res['params'], seed=res['seed'], windows=check['windows'],
                                               d_patch=check['study_vs_patched_max_abs'],
                                               d_stock=check['study_vs_transformers_max_abs'], steps=res['steps'],
                                               batch=res['batch'], ctx=res['ctx'], bp=res['byte_presentations'],
                                               macro=t['macro'], article=t['article'], fiction=t['fiction'],
                                               dialogue=t['dialogue'], name=out.name))
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir())
             if f.is_file() and f.name != 'metrics.json'}
    meta = dict(source_run=a.run, seed=res['seed'], params=res['params'], steps=res['steps'], batch=res['batch'],
                ctx=res['ctx'], byte_presentations=res['byte_presentations'], train_stream_sha=res['train_stream_sha'],
                train_s=res['train_s'], test_bpb=t, export_check=check, files_sha256=files,
                trained_by='om2bench.py arm m2 (OM2Mixer signed=False, bcnorm=False), one CPU thread')
    (out / 'metrics.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(dict(out=str(out), **check, test=t)), flush=True)


if __name__ == '__main__':
    main()
