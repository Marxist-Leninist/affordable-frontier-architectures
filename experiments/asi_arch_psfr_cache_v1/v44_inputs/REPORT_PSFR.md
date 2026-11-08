# FusionGatedFIRNet (#879): empirical small-scale test

## Result

FusionGatedFIRNet improved heldout language-model quality in all three seeds at essentially equal parameter count and exactly equal training-token budget. It was slower in this unfused CPU implementation, and its exported incremental cache is incorrect. This is promising architecture evidence, not a production deployment recommendation or reproduction of the paper’s reported benchmark gain.

| Model | Parameters | WikiText BPB, 512-input | Fresh FineWeb BPB, 512-input |
|---|---:|---:|---:|
| Modern Transformer | 190,848 | 2.734220 | 3.173170 |
| FusionGatedFIRNet / PSFR | 190,788 | 2.486615 | 2.972478 |

Lower bits per byte (BPB) is better. PSFR improved by 0.247605 BPB on WikiText and 0.200692 BPB on FineWeb, approximately9.1% and6.3% lower BPB, respectively. These percentages describe BPB, not general intelligence or downstream accuracy.

The98.75% paired document-cluster confidence intervals for PSFR minus Transformer are:
- WikiText: [-0.269252, -0.225950] BPB. Per-seed differences: -0.264068, -0.208489, -0.270259.
- FineWeb: [-0.223160, -0.178026] BPB. Per-seed differences: -0.185400, -0.176432, -0.240245.

Both pass the prespecified practical quality gate (upper interval bound and each seed difference ≤+0.02 BPB). Intervals use6000 document-cluster resamples after averaging seeds and retain the original four-comparison Bonferroni correction, even though this priority report covers only PSFR’s two contrasts. They condition on these three training seeds, not the full distribution of training runs or all research choices.

## Exactly what was tested

Source: GAIR-NLP/ASI-Arch, record879, `delta_net_psfr`, named FusionGatedFIRNet. Pinned commit3113c51b04a1d4a7520a3b0238c6a08d3194d58f, export `data/106.json`, blob e403531e663cf24c800002720d293cccf77f94df, Apache2.0. Original code remains byte-for-byte unchanged in `original/`.

- Independent sigmoid fusion per token and head across short FIR, long FIR, delta memory and direct value. This is additive sigmoid fusion, not softmax routing. Epsilon floor0.02, learnable per-head temperature, fusion MLP multiplier2.
- FIR kernels3 and63, source Dirac-plus-noise initialization.
- Source chunk32 delta rule with learned per-head retention minimum0.6, actual initialization0.99996 due source clipping. Retention is applied at chunk boundaries, not each token: after within-chunk updates, state subtracts(1−lambda) times the chunk’s incoming state.
- Three full PSFR mixer blocks, width64, two heads, expand_k=expand_v=1, short-convolution kernel4, no optional output gate. RMSNorm, residuals and SwiGLU; byte vocabulary256, untied embedding/output.
- Transformer: three full attention blocks, width64, four attention heads, RoPE, CPU scaled-dot-product attention, otherwise the same outer LM design.
- Total active parameters differ by only60, or0.0314%. Parameter matching reallocates FFN widths: Transformer[189,188,188], PSFR[108,107,107]. This compares whole architectures at a fixed parameter budget, not an isolated mixer-only substitution. No dummy parameters were added.

Each arm used seeds1,2,3 and800 steps × batch4 × context512 =1,638,400 byte-prediction targets/run. Identical sampled token windows for paired seeds, same fixed AdamW0.003 warmup/cosine recipe, clipping1, final checkpoint only, no test-based checkpoint/hyperparameter selection. Equal token budget does not mean equal FLOPs or wall time. Training uses the first two million bytes of line-deduplicated WikiText train; FineWeb is heldout evaluation, not training.

CPU adapters replace fused FLA RMSNorm, ShortConvolution and l2norm; torch.compile is disabled at import. Their numerical/initialization parity with the original pinned FLA environment has not been established. The custom PSFR kernel itself was checked against an independent token-loop reference. These are tiny CPU architecture proxies, not the original model size, tokenizer, dataset scale or optimized kernels. No full original DeltaNet baseline was included because its pinned baseline implementation was not present in the verified source package.

## Honest speed measurements

After every PSFR training/evaluation job finished, CSM starts remained blocked and the parent confirmed other managed-host heavy work was paused. Measurements used one CPU thread, FP32, sequential randomized paired calls, three seeds and six fresh FineWeb documents. Five paired repeats per document/context followed three warmups. Both models recomputed the full prefix. No cached speed is attributed to PSFR.

| Full-prefix input | Transformer median / p95 | PSFR median / p95 | Paired median PSFR/Transformer |
|---|---:|---:|---:|
|128 bytes, batch1|2.630 /4.618 ms|12.808 /17.350 ms|4.64× slower|
|512 bytes, batch1|10.651 /14.227 ms|24.888 /30.866 ms|2.33× slower|

A separate sequential matched training-step microbenchmark at batch4/context512 gave Transformer119.983 ms median versus PSFR399.124 ms, approximately3.33× slower. It included backward, clipping and AdamW step; two warmups and eight measured steps per seed, with zero learning rate and no checkpoint mutation. Raw measurements are included. These CPU results do not predict an optimized GPU implementation, production serving throughput, dollars per million tokens, margins or subscription viability. Transformer’s working cached decode is not represented by this full-prefix-only comparison.

## Cache and correctness findings

- Original-source CPU adapters passed causal-prefix, independent-batch, finite output and finite-gradient tests.
- Independent FP64 chunk-kernel reference matched outputs/state within1.2e-15 and gradients within2.2e-14. This checks kernel consistency, not fused FLA parity.
- The source writes a recurrent cache but never reads it into the delta kernel. Both FIR paths restart each call. Source cache/full-prefix equivalence fails.
- This persists after training: first-layer cached versus full-prefix max absolute differences1.5367,1.6414,2.0412 for seeds1–3. No valid incremental decoding latency, cache-memory efficiency or production claim follows from this export.
- Packed/masked sequences are explicitly unsupported in the adapter because FIR/delta branches ignore sequence boundaries. Training uses independent unpadded rows. No silent cache or model repair was applied.

## Evaluation and audit

Final128 next-byte targets are scored. The512-input setting provides385–512 preceding bytes across those targets; the128-input setting provides1–128. It does not give exactly512 or128 history bytes to every scored target.

WikiText:183 new-coordinate windows across50 previously used test articles. Four windows overlapping v42/v43 target coordinates were excluded. Fresh FineWeb:99 documents from rows2500–2599, first513 bytes per document; one short document excluded. Observed Hugging Face revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 verified and response hash frozen. Complete-document hashes were checked against2000 historical hashes, including v42/v43. Exact64-byte target halves were excluded against full deduplicated Wiki train/validation. These checks do not guarantee semantic deduplication or absence of every earlier research exposure. Some historical originals are unavailable, so their recorded hashes cannot all be independently rederived.

All six final checkpoints, parameter counts, complete loss histories and paired token-stream hashes were verified. Every heldout evaluation was rerun and matched deterministically. Independent static review found no critical source/protocol blocker; its report and hash attestations are included.

The paper’s Table1 (ASI-Arch v2, https://arxiv.org/html/2507.18074v2#S4) reports FusionGatedFIRNet52.01%, Mamba251.38% and DeltaNet51.04%, nominal1.3B scale trained on15B FineWeb-Edu tokens. That is an arithmetic mean of14 accuracy metrics (8 search/development and6 unseen-during-search generalization benchmarks), excluding two perplexities: +0.97 percentage points over DeltaNet, +0.63 over Mamba2. The paper does not report finalist multi-seed means, confidence intervals or error bars, and common training scale does not establish exact parameter/FLOP equality. The verified source export record879 has a separate earlier average0.401 and466.90M parameter metadata; those are not Table1’s final run. Neither external benchmark result is reproduced by our independent tiny three-seed BPB screening test. Source permalink: https://github.com/GAIR-NLP/ASI-Arch/blob/3113c51b04a1d4a7520a3b0238c6a08d3194d58f/data/106.json#L739 .

## Private deliverable and remaining work

This priority package includes six trained checkpoints, source, exact protocol, per-seed results, all timing samples, hashes, data audit, independent static review and restoration scripts. It excludes raw corpus text, evaluation arrays and credentials. Public sources can be restored at pinned hashes; files were not published to GitHub. CSM was intentionally paused to finish this requested FusionGatedFIRNet result first and is not required to interpret it.

An isolated ZIP extraction verified every member hash, rebuilt evaluation arrays from separately supplied hash-pinned public bytes, repeated the independent data audit, and matched frozen source/protocol/evaluation hashes. This portability check did not perform a fresh dependency installation or cold-network corpus download. Read README_PSFR.md for restoration and verification. No GPU purchase/rental, production restart or service modification was performed.
