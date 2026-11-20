# Same-forward attention training: full gradient vs recompute vs prefix gradient cutoff

The earlier Sakana-style experiments compared changes in objectives and architecture, including local noisy denoisers and independent side channels. This test isolates a different question: with identical full causal attention at inference and identical forward logits before training, can old-prefix gradient paths be dropped to save training memory? It does not change model or attention reach, introduce diffusion, or touch AGILLM production.

One identical ordinary 81,376-parameter two-layer byte Transformer, width48, head count4, exact same initialization/data/no dropout/AdamW per paired seed17/29/43. Each lane: 400 updates × batch6 × context192 = 460,800 next-byte target presentations per seed. All use the verified P044 524,288 source training bytes, frozen 8,192-byte WikiText dev, and 12,096 scored targets across three non-overlapping held-out within-domain windows sets per seed (21 windows/domain from WikiText official test, Shakespeare and Marlowe). These are prior diagnostic excerpts and do not support broad generalization. Dev measured every100; all final fixed 400th-step checkpoints used, no held-out selection.

- Full: standard exact causal backprop.
- Exact recompute: non-reentrant autograd checkpointing at each attention/MLP block, same forward and exact gradients, recomputing blocks on backward.
- Detached prefix: no-grad prepass caches exact K,V for each layer; chunks of48 are processed in order, attending to detached old K,V plus current trainable K,V. Four gradient accumulations per original batch and one optimizer step. Every byte contributes its own loss, but later loss cannot backpropagate into older cached keys/values or their hidden states on that gradient path. The cache is frozen only within a step and recalculated at current parameters each step. The *same* full-attention model is used for all held-out evaluations.

| Diagnostic order | Arm | Held-out BPB ↓ | Top-1 | Quality index | Saved activation + cache proxy | Fresh CPU p50/update |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Detached prefix | 4.83807 | 23.752% | 100.479 | 3.829 MB | 28.91 ms |
| 2= | Full | 4.84447 | 23.630% | 100.000 | 7.917 MB | 22.99 ms |
| 2= | Exact recompute | 4.84447 | 23.630% | 100.000 | 2.092 MB | 29.26 ms |

The index is 100 times geometric mean of relative top-1 and relative inverse perplexity vs full. Detached attention is not a statistically supported quality winner: delta −.00639 BPB on these paired seeds/excerpts, 95% within-cohort resample [−.01488,+.00337]; its late-half BPB 4.90693 vs full4.91087. Recompute final weights were **tensor-for-tensor bit identical** to full in all three seeds. Detached first-batch forward matched common logits exactly and passed causal future perturbation; it changes the gradient (seed17 relative L2 17.4%) by design. All nine weights passed strict reload and bit-identical fixture logits.

Activation proxy = nonparameter storage of forward autograd saved tensors, deduplicated by storage pointer within one active graph/chunk, plus separately counted resident detached K,V (884,736 bytes in the detached lane). This is **not** peak allocation, leaves out forward/backward temporaries, optimizer, allocator, parameter state, and may overcount overlapping lifetimes. Fresh-process 12 warm +52 measured steps per arm, same real training samples, sequential one thread: RSS peaks full325.1 MiB, recompute324.3 MiB, detached318.5 MiB; interpreter/runtime dominates. Relative to full, proxy drops 73.6% recompute and 51.6% detached; median update wall increases 27% and 26%. Separate training-loop totals retained in analysis.json; no measured CUDA peak, throughput or economics.

**Decision:** If activation memory is the pressure, profile exact recomputation before dropping gradient paths. It preserves model updates and did better on this storage proxy at practically the same extra CPU time as truncation. Validate actual allocator/peak GPU memory, wall-clock and meaningful contexts on isolated capacity before AGILLM application. Do not deploy either based on this tiny offline test.

Reproduce using CPU PyTorch+NumPy and one thread: bench.py per arm, memory_profile.py per arm, analyze.py. P044 data are referenced by pinned local layout, not bundled. Same parent's ordinary Transformer components retained; no buy/spend, live change or contested accelerator use.

Public package includes code, provenance, scores, and hashes. The full checkpoint archive is preserved privately. The pinned corpus is intentionally not redistributed.
