---
license: mit
library_name: pytorch
base_model: Counfield/Diogenes-0.1-20M
datasets:
- Salesforce/wikitext
tags:
- byte-level
- recurrent
- research
---

# Diogenes 0.1 20M: independent continuation pilot

Experimental, not an instruction model or deployment recommendation. This
is an independently trained continuation of the exact original publicly
released [Counfield Diogenes-0.1-20M checkpoint](https://huggingface.co/Counfield/Diogenes-0.1-20M/tree/f39659c9805f098b581483e7bf46a529e905e303),
with author license reproduced in LICENSE. Original source author
attributed; reconstructed code was written independently from the public
architecture operations and does not import the upstream model file. No
binary quantization or Dory/Inkling layer sharing is claimed: this recurrent
Transformer feeds the decoder's hidden state from the preceding byte,
runs a causal encoder and has a local four-position decoder attention
plus a causal cross-attention memory. Its byte tokenizer is UTF-8 (0-255).

The immutable original 20,297,472 FP32 parameters load strictly by exact
names from the pinned 81,192,824-byte safetensors with SHA-256
5c274c53dbfbbc2f57ca34a8e8634c7e8ea56db9b48b0799a792c352b05f6190 .
At original initialization the new batched encoder and recurrent
decoder match a separately written manual one-token oracle derived from
the author's published forward equations (largest absolute logit difference
9.54e-6 float32; argmax exact on this fixture). No claim of recovering
the author's full pretraining corpus, optimizer, RNG, or scheduler.

Independent full-weight AdamW update in each of two 16-step arms, 512
true next-byte targets per arm, choosing LR on validation before scoring
held-out test. Official
[Salesforce WikiText2](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3)
pinned to its official train, validation, and test partitions, released
CC BY-SA 4.0 per publisher. No excerpts are included. Test disjoint from
adaptation; original model's possible WikiText exposure is unknown.
One seed, 24 test windows (768 targets), no wider quality or safety claims.

## Verified result
The actual continuation repeats the same fixed exposure from the original
using the frozen windows obtained over Hugging Face HTTPS with a fresh
datasets streaming=True dataset session; no invented or synthetic target.
The on-disk checkpoints for cached/public-stream replicas use different
metadata serialization ordering but all 35 loaded tensors were bit-exact.
The live-stream run and its file SHA are the publication candidate, not
the local-data replica.

| Configuration | Adaptation-heldout bits/byte ↓ | CPU training seconds | Combined ↑ |
| --- | ---: | ---: | ---: |
| 1e-4, selected on validation | 2.331191 | 9.94 | 101.87 |
| 3e-5 | 2.349104 | 10.79 | 100.85 |
| Unchanged original | 2.363507 | not applicable | 100.00 |

Quality/time index for all models evaluated under the same CPU eval protocol:
100 × 2^(0.9 × (original BPB − arm BPB)) × (original eval time / arm eval time)^0.1.
Timed eval is a small noisy CPU measurement, not evidence of throughput gain.

The 1e-4 continuation file for this exact measured run is 81,192,904 bytes,
SHA-256 a0f7e744b1ed6f0a2ec90f679fc1b5ea9629066d0841d1d09d66691000dd0a5e.
New separate process reloaded that file, the second LR file, and the
original, preserving all 72 scored window losses bit-for-bit. This package
contains reproducible source and measurement receipts, not model weights,
raw WikiText excerpts, or upstream optimizer state.
