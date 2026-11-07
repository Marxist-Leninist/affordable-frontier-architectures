# Pretrained DiffusionBlocks conversion, controlled retry

**Outcome: direct selected-block training works, but this conversion still does not preserve language quality. Distillation reduces the measured quality loss without closing it.**

This is a bounded, CPU-only research test of independently trainable causal denoising blocks initialized from **EleutherAI/Pythia-14M**. It is not a 512B experiment, a production change, or a reproduction of unpublished language-model code from the paper.

The question is whether selected-block training can avoid earlier-layer execution and weight loading **while retaining useful pretrained language quality**. Structural savings and language quality are separate outcomes. The paper identifies pretrained conversion as future work; its main language results concern training from scratch. This pilot does not disprove those results. Our held-out decoder NLL also differs from the paper's generated-text MAUVE and external-model perplexity metrics.

## Results

Held-out nats per token, lower is better. Means and sample standard deviations are over three training seeds; all individual values are retained below and in the JSON records.

| Method | Endpoint | Seed 11 | Seed 22 | Seed 33 | Mean ± SD | Training CPU s | CPU s including offline teacher computation |
|---|---|---:|---:|---:|---:|---:|---:|
| Untouched | No adaptation | 5.615 | same model | same model | 5.615 | 0 | 0 |
| Ordinary CE | equal data | 5.511 | 5.529 | 5.531 | 5.523 ± 0.011 | 52.50 | 52.50 |
| Ordinary CE + teacher KL | equal data | 5.526 | 5.425 | 5.520 | 5.490 ± 0.057 | 69.08 | 122.31 |
| Direct blocks CE | equal data | 8.316 | 9.841 | 8.489 | 8.882 ± 0.835 | 42.36 | 42.36 |
| Direct blocks CE | equal cpu | 8.965 | 8.991 | 9.027 | 8.994 ± 0.031 | 52.52 | 52.52 |
| Direct blocks CE + teacher KL | equal data | 8.111 | 8.560 | 8.039 | 8.237 ± 0.282 | 58.52 | 111.74 |
| Direct blocks CE + teacher KL | equal cpu | 8.342 | 8.667 | 8.310 | 8.440 ± 0.198 | 69.14 | 122.36 |

At equal data, direct blocks lose **3.358 nats/token** against ordinary CE; adding teacher KL reduces that gap to **2.746 nats/token** against the paired ordinary-KL control. The corresponding paired window-bootstrap 95% intervals are [3.050, 3.678] and [2.467, 3.030], conditional on these three seeds and 64 test windows, not population confidence intervals over training runs or datasets. Teacher KL improves the converted decoder by 0.645 nats/token, but does not preserve quality.

Extra CPU-matched training did not close the gap. Direct-CE endpoints occurred at 996, 942, and 946 updates; direct-KL endpoints at 915, 910, and 892. Measured CPU overshoot was at most 0.131%. No best seed or checkpoint was selected.

Equal-data training CPU fell **19.3%** for CE and **15.3%** for KL. Including each recipe's computation for the full teacher cache reduces the paired KL saving to **8.6%**. KD is not a free improvement: its reported computation subtotal costs substantially more than the CE-only control.

### Measured selected-block costs

| Quantity | Ordinary body | One converted group |
|---|---:|---:|
| Logical body bytes read, shared maps resident | 4,767,632 | 1,796,744 |
| Logical bytes read, including shared maps | 56,279,120 | 53,308,232 |
| Updated trainable-weight bytes written | 4,782,456 | 1,803,123 |
| Saved nonparameter tensor storage, bytes | 16,065,540 | 15,017,092 |
| Separate-process peak RSS, bytes | 524,914,688 | 547,074,048 |
| Profiled matrix-operator FLOPs per CE step | 2,120,631,042 | 1,932,253,452 |

The selected trainable body read is **62.3% smaller**, proving that earlier/later group weight files need not be read for this update. With the 51.51 MB shared vocabulary maps included, logical read savings are only **5.3%**. Reading all three converted bodies would require 5.39 MB, versus 4.77 MB for one ordinary full-body read; a selected update and a complete group sweep are different units of work.

Saved nonparameter storage fell **6.5%**, not threefold. Observed process peak RSS was **4.2% higher**, including loading/construction peaks; this one-process measurement is not a robust hardware-wide memory estimate. The normalized codebook adds memory. Supported matrix FLOPs fell **8.9%**. Actual initial `/proc/self/io` physical reads were 11.49 MB versus 57.77 MB, reflecting uncontrolled cache/readahead behavior, so this test does **not** demonstrate a physical-disk bandwidth advantage.

Full-prefix inference took 0.0292 CPU seconds for ordinary CE, 0.1211 for one converted particle (**4.14×**), and 0.5014 for four particles (**17.15×**). These are warmed batch-2/context-32 measurements with clean-context preparation charged every time, not incremental token generation.

Logical K/V size is identical at 393,216 bytes across all six layers. The explicit converted cache retains 786,432 backing-storage bytes versus 393,216 in the native implementation because some values are tensor views. Cache reuse is therefore **not cache compression**; a more careful buffer layout remains an engineering opportunity.

### Validation-only decoder diagnostics

| Seed-11 method | Primary normalized rollout | Unnormalized rollout | High-noise one-pass |
|---|---:|---:|---:|
| Direct blocks CE | 8.756 | 7.617 | 6.741 |
| Direct blocks CE + teacher KL | 8.397 | 7.419 | 6.725 |

The one-pass head is much better than this iterative decoder on validation, showing a substantial decoder/transport problem. It remains worse than the seed-11 ordinary controls (5.480 and 5.488). These diagnostics did not select a replacement test decoder. The two short sample continuations are in `samples.json`; they are not adequate evidence of generation quality.

### Decision

**Do not transfer this recipe to a live 512B trainer.** The direct-input and selective-weight-loading contract passed, but the quality, memory, and inference tradeoff did not. No production or paid-resource action was taken.

Useful next experiments, not completed claims:

1. Preserve more of the full pretrained function before partitioning. Test nonlinear context adapters and progressive representation/logit distillation against matched controls, rather than assuming a two-layer partition can reproduce the six-layer function.
2. Cache the teacher's 128-dimensional final hidden states on training text, then reconstruct full-vocabulary targets through the resident frozen output head. At this shape the cache would be about twice the current top-32 cache size; earlier layers would still be unnecessary during student updates. This trades additional head computation/memory for exact teacher targets and has not been tested here.
3. Improve sampling/transport using validation-only tuning and an untouched new final evaluation. Include particle sensitivity and generated-text metrics comparable to the paper.
4. Only after quality preservation is demonstrated, measure actual GPU/offloaded end-to-end training and protected incremental-generation parity on a larger model.

All 18 checkpoint integrity checks and four exact held-out score replays passed. The causal, gradient-isolation, optimizer-isolation, fixed-prefix cache-parity, and descending-Euler oracle tests passed.

## Sources and licenses

- [DiffusionBlocks paper, arXiv:2506.14202](https://arxiv.org/abs/2506.14202), examined as v4. [Official SakanaAI code](https://github.com/SakanaAI/DiffusionBlocks) provides vision implementations, not the complete language-model implementation.
- [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m), Apache-2.0; pinned revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`.
- [WikiText-2 raw v1](https://huggingface.co/datasets/Salesforce/wikitext), pinned revision `b08601e04326c79dfdd32d625aee71d232d685c3`; source text is CC-BY-SA-3.0/GFDL. No dataset or model weights are included here. See the manifests for exact file hashes, provenance, and tokenization.
- [Previous pilot](../pretrained_diffusion_v1). This retry uses new nonoverlapping scored contexts, so the old 12.08 score is not an apples-to-apples baseline for the new numbers.

## Design

Pythia-14M has six transformer layers with hidden width 128. Every method freezes the original input embedding and output vocabulary matrices. Ordinary continued training updates all six layers and the final normalization. Conversion divides the six pretrained layers into **three independent two-layer groups**, each with a copied final normalization, zero-initialized residual context adapter, zero-initialized noise adapter, and noise-time MLP with zero output initialization.

A block's clean input comes directly from past-token embeddings, not earlier blocks. The noisy query is anchored at the last clean past-token embedding and includes a corrupted normalized target embedding during training. Clean and query streams use **only clean causal keys/values**. Query position i predicts token i+1, can read clean positions j <= i, and cannot read any other noisy target or clean future token. At evaluation no target embedding is supplied: the decoder starts from independent Gaussian noise.

The clean stream is independent of noise level. Each distinct block computes its own clean K/V once for a fixed prefix, reusing it across denoising steps and particles. This does not share K/V between different blocks, compress the cache, or establish incremental autoregressive continuation parity.

A separate full-depth neutral-adapter construction reproduces the original logits exactly. **Partitioning that construction into independent two-layer blocks is not function-preserving.** The full-depth sanity check must not be interpreted as preservation of the partitioned model's language quality.

### Training controls

- Three seeds: 11, 22, 33. Each model starts independently from the same original checkpoint.
- Four methods: ordinary CE, ordinary CE plus teacher KL, direct-block CE, direct-block CE plus teacher KL. Untouched original is evaluated separately.
- Equal-data endpoint: 768 updates, batch 2, context 32, **49,152 token exposures**. Direct groups cycle 0, 1, 2, so each group receives one-third of the updates. Equal token exposures do not mean equal updates per parameter.
- Additional direct endpoint: first update reaching the paired ordinary method's measured training CPU time, capped at 1,536 updates. Training CPU includes forward, backward, clipping, and AdamW. It excludes model loading, warmup, data/noise preparation, and evaluation. These are CPU-matched endpoints, not simultaneously equal-data, equal-wall-time, or equal-FLOP endpoints.
- AdamW: learning rate 1e-4, 48-step warmup, betas (0.9, 0.95), weight decay 0.01, gradient clipping 1.0. The protocol is frozen in `protocol.json` with primary-source hashes.
- Noise bands divide truncated lognormal probability equally: log sigma ~ Normal(-1.2, 1.2), sigma in [0.002, 80]. Unlike v1, denoising CE is unweighted.
- Teacher KL uses the teacher's top 32 token categories plus one aggregate tail category. It is categorical KL on that **coarsening**, with a 1e-7 numerical floor on tail masses, not full-vocabulary KL. Its multiplier is sigma²/(sigma² + 1/128), with the same paired schedule for ordinary and converted KD methods.
- Teacher targets are generated offline from the untouched full model on training windows only. No full teacher executes during student updates. Each seed's cache covers the maximum 1,536 windows and is 25,166,080 bytes. Its measured computation over all 1,536 cache windows is charged to each KD recipe, although the cache is physically generated only once per seed. The cache timer excludes teacher loading, initial array allocation, and final file serialization. The reported recipe CPU is a computation subtotal, not complete end-to-end elapsed cost. Cache creation still requires the original full model.

### Evaluation

Official train/validation/test splits are retained. New evaluation contexts are disjoint from all v1 scored contexts: 32 validation and 64 test windows of 33 tokens, with the last 16 targets scored per window (512 validation and 1,024 test targets). Source checks and window hashes are included. Original pretraining overlap with WikiText is unknown; this is not a decontaminated benchmark.

Primary converted inference is fixed before outcomes: four independent, reproducible noise particles; normalized posterior-mean embedding transport; one negative-delta Euler step per noise band; then one final low-noise readout from the last group. Each particle uses four denoiser calls. Per prefix this needs six clean layer passes plus eight query layer passes per particle.

The reported converted score is the negative log probability of the **explicit finite four-particle mixture decoder**, not the exact likelihood or perplexity of an implicit diffusion model. Low particle counts, transport, and calibration can materially affect it. Ordinary scores are standard next-token NLL at context length at most 32. Lower is better.

All seeds and both direct endpoints are reported. No test-based checkpoint or decoder selection is performed. Prespecified high-noise one-pass and unnormalized-transport diagnostics are validation-only on seed 11. Two fixed short greedy generations are illustrative, not a generation benchmark.

## Structural, leakage, and integrity tests

`checks.py` and `extra_checks.py` cover:

- Exact pretrained-logit parity of the full-depth neutral-adapter construction.
- Future clean-token exclusion, other noisy-target exclusion, and own-noise sensitivity with the noise path actively enabled, avoiding a vacuous zero-adapter test.
- Selected middle group executes only its own layers; earlier and later groups receive no forward calls or gradients.
- One optimizer update changes only the active group, with shared vocabulary maps frozen.
- Exact cached/uncached denoiser parity across all groups and three noise levels; fixed-prefix cached tensors remain unchanged by repeated noise queries.
- Finite gradients and decoder outputs, valid noise bands, teacher-KL identity, and a constant-denoiser oracle that checks the actual rollout contracts noise with the correct Euler sign.

`validate.py` checks source/data/window hashes, unchanged original files, all 18 saved adapted endpoints, finite checkpoint tensors, exact trainable key sets, four exact seed-11 held-out score replays, and the selected-block weight-loading trace. Checkpoints remain local and are not included in the release.

## I/O and cost methodology

Primary adaptation runs hold weights in RAM. `accounting.py` separately executes an isolated offload proof using actual scratch weight files, not an analytic estimate:

1. One-time export loads the full models and writes shared frozen vocabulary maps, the ordinary full trainable body, and each converted group separately. Export cost and hashes are recorded.
2. A fresh ordinary process loads shared maps plus the six-layer body. A fresh converted process constructs only a two-layer model and loads shared maps plus the selected middle group. It never loads earlier or later group weight files. Exact source-logit hashes verify that the loaded subsets reproduce the intended models.
3. Both run real CE forward/backward/clipping/AdamW steps. The probe records warmed CPU/wall time, supported matrix-operator FLOPs, saved nonparameter tensor storage, and separate-process peak RSS, including construction/loading peaks rather than only the warmed training state.
4. Actual logical file bytes and `/proc/self/io` counters are recorded for reading/deserializing files, rereading a body with shared maps already resident, and writing/fsyncing updated trainable weights. Optimizer-state transfers are excluded.

“Cold component” means shared maps are loaded too, **not that OS page caches were flushed**. Physical read bytes may be zero. File sizes and logical reads do not establish SSD bandwidth, end-to-end offloaded training throughput, or 512B performance. Sharding itself is not free. The full sweep of all three converted groups includes adapter overhead and should not be confused with one selected-block update. Shared vocabulary maps dominate this small model; the converted normalized codebook also consumes RAM.

`inference_accounting.py` measures actual full-prefix inference, including clean-context preparation, with one and four particles. This is not incremental next-token cache continuation or a long-context benchmark. Logical K/V elements and their retained backing storage are both reported.

## Reproduction

The recorded runs used Python 3.12.14 and the pinned dependencies in `requirements.txt`. Install CPU Torch from its official wheel index as needed. Run `bash reproduce.sh` in this directory. `prepare.py` fetches pinned public model/data files, checks hashes, and uses safe tensors with `trust_remote_code=False`. No hub Python code is executed.

The released protocol and window manifest are already frozen. The script does not regenerate or silently replace them. Equal-data endpoints are deterministic under the tested CPU environment; exact timing-matched update counts depend on hardware/runtime load. Report the resulting counts and measured CPU rather than expecting identical wall times or matched-CPU checkpoint hashes on another machine.

The tested process configuration is one intra-op and inter-op thread, autograd multithreading disabled, FP32 with process-local denormal flushing, and deterministic-algorithm enforcement. The first longer launch was killed before any adapted checkpoint; its cause was not conclusively established. Bounded diagnostics found severe backward timing variability under initial defaults. The runtime-only amendment was made before any adapted v2 outcome, with exact tested logit/gradient parity, unchanged untouched NLL, and identical regenerated teacher-cache hashes. Original source/protocol and untouched result are retained under `execution_history/`. No model, objective, data, or quality-selection change was made during that amendment.

This release contains code, manifests, scores, integrity results, and execution history only. It excludes model weights, checkpoints, raw text, token arrays, and teacher caches. No live trainer or service was changed and no compute was rented.
