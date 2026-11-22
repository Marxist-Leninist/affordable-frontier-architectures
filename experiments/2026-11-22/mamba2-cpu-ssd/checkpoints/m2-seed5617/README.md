---
library_name: transformers
tags: [mamba2, byte-level, research, cpu]
---
# Byte-level Mamba-2, 471,008 parameters (OM2 study, seed 5617)

A tiny research checkpoint in standard transformers `Mamba2ForCausalLM` format. Tokens are raw bytes: token id = byte
value, vocabulary 256, no tokenizer. Architecture and initialisation are exactly what transformers 4.57.6 builds for
this config: 4 Mamba-2 layers, hidden 128, 8 heads x 32, state 32, 1 group, conv 4, chunk 32, tied embeddings.

It was trained on one CPU thread by the OM2 study (`om2bench.py`, arm `m2`), which runs the same chunked SSD kernel as
`../../mamba2_cpu_ssd.py` in its own single-group copy. On 48 held-out windows, this checkpoint's logits with
`patch_mamba2` are identical to the study model's (max abs difference 0.0e+00), and stock transformers gives
the same logits to 1.1e-05.

- Training: 2,000 updates x 8 sequences x 256 bytes (4,096,000 byte presentations) from the "P056" corpus
  (WikiText-2 encyclopedia articles, public-domain Project Gutenberg fiction, OpenAssistant oasst1 dialogue; see
  `../../THIRD_PARTY_NOTICES.md`); AdamW lr 3e-3, warmup 100, cosine to 10%, weight decay 0.01 on matrices, gradient
  clip 1.0.
- Held-out test (the corpus's 303 test windows; every scored byte sees 129-256 bytes of context):
  **2.1828 bits/byte** macro average (article 2.1721, fiction 2.1908, dialogue 2.1854).

```python
import torch
from transformers import Mamba2ForCausalLM
m = Mamba2ForCausalLM.from_pretrained("ScAGI/affordable-frontier-architectures",
                                      subfolder="experiments/2026-11-22/mamba2-cpu-ssd/checkpoints/m2-seed5617")
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
