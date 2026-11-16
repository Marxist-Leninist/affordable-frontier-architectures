# lctx: a learned context length for the AGILLM 888B (16 Nov 2026)

The idea: let the model decide, in training and at inference, how far back each attention head looks.

## Mechanism

Each layer and head gets the Forgetting Transformer's forget gate (Lin et al., ICLR 2025), with a learned per-head
bias:

    a_t = min(softplus(r_h + w_h . unit(v_t)), amax)            (v_t = the head's value at token t)
    logit_ij = s * q_i . k_j  -  (a_{j+1} + ... + a_i)           causal, j <= i

- `1 / softplus(r_h)` is head h's context length in tokens, learned on a log scale while it is long.
- `w_h` lets any single token raise the rate up to `amax`. A head can therefore keep a long context inside a document
  and still forget everything at a document break. The model decides this at inference, token by token.
- With `r = -inf` this is the original attention. The trainer installs it at rate 1e-6, which is the model as it was:
  the oldest of 2048 keys loses at most 0.002 nats.

**Exact, with no T x T mask.** The decay equals `kb_j - kb_i`, where `kb` is a running sum of `a`. Softmax ignores the
per-row term, so one extra head dimension carries the key bias: `q' = [q, 1/s, 0..]`, `k' = [k, kb, 0..]`, with hd
going from 128 to 136. The fused SDPA kernels work unchanged.

**One autograd Function.** Gate and attention run in a single Function. Its forward computes everything without a
graph and saves `q, k, v, r, w` only once it has succeeded. A failure therefore saves nothing, and the caller can
fall back or retry without breaking non-reentrant checkpoint recompute. The backward redoes each head group's forward
under its own small graph. The gate's gradients are analytic and never flow into `v`.

**Inference.** `kb` never decreases. Once `kb_i - kb_j > theta`, key j stays below `e^(2*T_att - theta)` of the
weight for that query and every later one. It can leave that head's KV cache for good.

## Toy experiment (`tests/toy_lctx.py`, results in `results/*.json`)

Setup:
- A small model with the 888B's attention: cosine q/k, RoPE, logit temperature sqrt(hd), ReLU^2 MLP.
- d=192, L=4, H=4, byte-level tiny-shakespeare, trained at context 256.
- 1500 steps without the gate. Then, from that checkpoint, as on the 888B, 1500 more steps three ways:
  - **base**: unchanged.
  - **lctx**: the gate, with the loss alone deciding the spans.
  - **lctx_c**: the gate plus a price on context (span_cost 0.02).

Validation loss (nats per byte):

| context | base | lctx | lctx_c |
|---|---|---|---|
| 256 (trained length) | 1.6535 | 1.6548 | 1.6753 |
| 1024 (4x) | **2.1009** | **1.6166** | 1.6443 |
| 2048 (8x) | **2.5695** | **1.6120** | |

- **The loss alone chose short contexts.** 14 of the 16 heads ended between 41 and 105 bytes; two kept 500 and 780.
  At the trained length the loss was level with base.
- **The gated model holds up beyond its trained length.** It is slightly better at 4x and 8x than at 1x, while base
  degrades by +0.45 and +0.92 nats.
- **The span price was harmful.** span_cost 0.02 collapsed every head below one token and cost 0.02 nats.
  The live default is 0.

KV cache a decoder keeps when it evicts keys more than theta behind (lctx model, share of a full cache at the last
position):

| theta | 1024 tokens: kept / extra loss | 2048 tokens: kept / extra loss |
|---|---|---|
| 4 | 31.5% / +0.0001 | 21.4% / +0.0003 |
| 8 | 50.5% / 0.0000 | 31.5% / 0.0000 |
| 12 | 69.5% / 0.0000 | 41.0% / 0.0000 |
| 16 | 82.4% / 0.0000 | 50.5% / 0.0000 |
| 40 (provably safe) | 100% / 0.0000 | 90.8% / 0.0000 |

So at 8x context the cache can shrink to about a third with no measurable loss.

Caveat: this is one seed of a 1.8M-parameter byte model. What the 888B learns is being measured on its own run.

## Code

| file | contents |
|---|---|
| `src/lctx_core.py` | LctxAttn (gate + biased attention, one Function), LctxState (per-layer Adam), references, decode cache |
| `src/lctx_hotmod_glue.py` + `tools/build_hotmod.py` | the trainer hotmod (details below) |
| `src/lctx_serve.py` + `tools/export_lctx.py` | pack files, whole-prompt / chunked / decode attention, eviction; export bound to the pack's own record |
| `tests/` | 16 core, 10 hotmod (fake trainer), 4 serving, plus an e2e gate on the real trainer source (`results/e2e_gate_v1.1.txt`) |

The trainer hotmod:
- wraps `_sdpa` and `_rs_status` (a `_prev` kwarg, like the other hooks);
- changes on/off state only at step boundaries;
- keeps a two-strike crash marker, a kill switch and an explicit `state.json` record on every launch;
- pins the gate of every paused or exported step (sha256 in the record);
- rewinds by the checkpoint's own update count.

## On the 888B

v1 went live on trainer 58892275 at step 7969 (16 Nov 2026, 13:43 UTC) with no restart. v1.1 followed at 15:15 UTC,
in-process, with fixes from an adversarial review.
- Causal passes only (AR, SAT, held-out AR: about 78% of steps). NAT and local-denoise passes are unchanged.
- The gate learns from the training loss alone.

Serving is not live yet. Packs cut after the spans move need `lctx_serve.py` in the serving release.
