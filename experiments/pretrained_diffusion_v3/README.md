# Practical local-objective comparison on a pretrained causal model

This study asks which local-training mechanism helps in practice, rather than which is most paper-faithful. It compares **common native causal inference** after ordinary continued training, AGILLM-inspired masked local training, causal-direct local denoising, and combinations. No production model or trainer is changed.

This is a dense Pythia mechanism study, **not a reproduction of the full current AGILLM runtime**. The distinction matters for every result below.

## Results and decision

**Ordinary continued training has the best mean language quality. Masked hybrids are the closest cost-quality tradeoff, but no candidate passes the conservative quality-preservation screen. Local-only training is much faster in some cases and loses substantial quality.**

### Equal input data: 648 updates per seed

Lower NLL is better. The scores below are means over three seeds; standard deviations reflect seed variation, not a population confidence interval. CPU measures training computation only.

| Method | Test NLL, mean ± SD | Gap versus ordinary | Training CPU seconds | CPU change |
|---|---:|---:|---:|---:|
| Untouched original | 5.610 | N/A | 0 | N/A |
| Ordinary AR | 5.458 ± 0.013 | +0.000 | 44.34 | +0.0% |
| Masked hybrid, parent group | 5.496 ± 0.033 | +0.038 | 41.56 | -6.3% |
| Masked hybrid, all groups | 5.489 ± 0.015 | +0.031 | 41.86 | -5.6% |
| Masked local only | 9.029 ± 0.035 | +3.572 | 20.86 | -53.0% |
| Causal-direct hybrid | 5.524 ± 0.027 | +0.066 | 42.85 | -3.4% |
| Causal-direct local only | 10.700 ± 0.500 | +5.243 | 36.29 | -18.1% |

The masked hybrids lose only 0.031–0.038 nats/token on average, roughly 3.2–3.9% higher perplexity under this common causal metric, while using 5.6–6.3% less measured training CPU. However, their conditional 95% upper gap limits are 0.062 and 0.065 nats, above the prespecified 0.05 margin. The apparent CPU advantage is modest and measured on one CPU environment with only three paired seeds. It is not a robust hardware-wide throughput guarantee.

The causal-direct hybrid has a larger 0.066-nat mean loss penalty and only 3.4% less measured CPU. The paper-inspired mechanism is therefore not the preferred candidate merely because it resembles the paper.

### Additional CPU-matched endpoints

| Method | Test NLL | Gap versus ordinary | Conditional 95% gap interval | Updates, seeds 101/202/303 |
|---|---:|---:|---:|---|
| Masked hybrid, parent group | 5.493 | +0.035 | [-0.004, +0.076] | 698 / 720 / 661 |
| Masked hybrid, all groups | 5.473 | +0.016 | [-0.029, +0.064] | 666 / 701 / 693 |
| Masked local only | 8.502 | +3.044 | [+2.837, +3.269] | 1375 / 1434 / 1391 |
| Causal-direct hybrid | 5.506 | +0.048 | [+0.007, +0.089] | 653 / 710 / 647 |
| Causal-direct local only | 11.114 | +5.657 | [+5.411, +5.899] | 810 / 812 / 747 |

The best mean among the CPU-matched alternatives is the all-group masked hybrid at 5.473 versus ordinary 5.458, but its uncertainty still crosses the 0.05-nat margin. No alternative establishes both cost reduction and conservative quality preservation in this small trial. Bootstrap intervals resample the 64 test windows conditional on the three trained seeds; they do not cover all possible training seeds, data, or models.

Extra local-only updates do not solve the problem. Masked local-only reaches 8.502 at matched CPU; causal-direct local-only reaches 11.114. In this tested setup, only mixtures retaining full-depth AR anchors stay close to the ordinary model. The local masked variants also predict fewer labels at equal input data; both counts are reported rather than calling the supervision identical. No local update exceeded the loss-35 skip threshold.

### What the actual weight-file probe establishes

| Loaded component | Ordinary full body | Masked selected group | Causal-direct selected group |
|---|---:|---:|---:|
| Trainable body bytes actually read | 4,768,000 | 1,592,304 | 1,797,088 |
| Bytes read including shared maps | 56,279,496 | 53,103,800 | 53,308,584 |
| Updated trainable weight bytes written | 4,782,840 | 1,597,762 | 1,803,507 |
| Saved nonparameter tensor bytes | 16,065,540 | 7,558,472 | 15,017,092 |
| Process peak RSS bytes | 554,496,000 | 519,135,232 | 573,001,728 |
| Profiled matrix-operator FLOPs | 2,120,631,170 | 981,657,858 | 1,932,253,580 |

Fresh probe processes loaded only the shared maps and their designated body file; exact source-logit hashes passed. A selected masked body is 66.6% smaller than the ordinary body, and a selected causal-direct body is 62.3% smaller. Thus earlier/later block weight files really are unnecessary for these selected updates. The shared frozen vocabulary maps are another 51.51 MB. Including them reduces per-load logical savings to 5.6% and 5.3%, respectively.

At an 8:1 anchor/local schedule, measured shard sizes imply only **7.4% fewer trainable-body read bytes per equal-data masked-hybrid run**, or **6.9% fewer for the direct hybrid**, assuming the shared maps remain resident and each needed body is reread every update. These are modeled schedule totals from actual files, not observed main-run disk traffic. At the CPU-matched endpoints, extra updates shrink the modeled read savings to approximately 0.9% for parent-masked, 1.8% for all-group-masked, and 3.7% for direct hybrid. Optimizer-state transfer and the production offload system are not modeled.

The masked probe scores 32 of 64 positions at sigma 0.5, while the other probes score all 64. Its lower local activation storage and FLOPs partly reflect that reduced target count. Other mask rates will differ. **A hybrid still needs the full ordinary anchor graph eight times out of nine, so these local measurements do not establish a lower hybrid peak-memory requirement.** A complete sweep of all three local groups also differs from a single selected update.

### Inference and integrity

All treatments keep ordinary causal inference. Measured warmed full-prefix forwards ranged from 0.023 to 0.028 CPU seconds at batch 2/context 32; there is no multi-particle inference penalty in this design. Each seed-101 treatment passed an actual native 16-token prefill plus 16-token continuation check with **zero logit difference** and identical argmax predictions. This is a short native-cache check, not a long-context or low-bit-KV benchmark, and does not validate diffusion-specific cross-block cache sharing.

All 33 checkpoint integrity checks and six exact held-out score replays passed. Initial function preservation, causal exclusion, removal of masked targets, selected-group execution/gradient/optimizer isolation, and source/data/checkpoint hashes passed. No checkpoint was deleted or uploaded.

### Practical recommendation

Keep full-depth anchor training as the safe reference. If pursuing another bounded cost-quality experiment, the masked hybrid is the stronger candidate here than pure local training or a causal-direct hybrid. More held-out contexts and seeds are needed to resolve a small acceptable quality margin, followed by a protected model-specific test before changing any native runtime.

This result does **not** authorize or justify disabling the current production multimode objectives, broadening its ancestor-restricted group rotation, or applying a new 512B recipe. The current source has ternary/MoE, learned SAT policy, expanded ancestry, and rotating writeback features absent from this proxy. No live trainer or service was changed.

## What is matched and what is different

All methods start from the same pinned [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m) checkpoint and use its unchanged six-layer causal inference graph. Original input/output vocabulary matrices stay frozen. Local auxiliary branches share the same native transformer layer parameters. Their conditioner/norm parameters do not participate in causal inference. Initial causal logits are bit-exact across treatments and the untouched original.

Six prespecified methods:

1. `normal`: ordinary causal next-token CE at every update.
2. `masked_hybrid_parent`: eight ordinary AR updates, then one masked local update restricted to the first two-layer group.
3. `masked_hybrid_all`: the same 8:1 mixture, cycling local updates across all three groups.
4. `masked_local_all`: only local masked updates, cycling groups.
5. `direct_hybrid_all`: eight ordinary AR updates, then one causal-direct denoising update, cycling groups.
6. `direct_local_all`: only causal-direct local updates, cycling groups.

The masked objective follows verified AGILLM local-step semantics: sample scalar sigma from a descending group-specific uniform band within [0.05, 0.95]; select tokens independently with probability sigma, forcing one if necessary; replace selected inputs with an unused reserved mask ID; apply sqrt(1-sigma²) times the masked-token embedding plus sigma times standard-normal noise; apply zero-initialized scalar-noise affine conditioning; execute only the selected group with **bidirectional** attention; reconstruct only selected original tokens. The probe uses Pythia LayerNorm instead of AGILLM RMS. ID 50303 is verified absent from every data split and is not an ordinary token or EOS.

The causal-direct objective reuses the tested v2 clean-past/noisy-query mechanism with independent group adapters and lognormal noise bands. Its target-corrupted query can see only clean past keys, not other noisy targets or clean future tokens. In v3 this is an **auxiliary training objective**: evaluation remains the full native causal model, not the v2 finite-particle diffusion decoder. These are practical modifications, not a paper-faithful diffusion generator.

## Relation to the current AGILLM implementation

Read-only source inspection verified the local mechanism and cadence in `agillm128b.py`, source SHA256 `1241dead5f786b7fa398c8f9859de182cc9c97341718ff9ffc8990f5e174942c`. The active parent-group override and launch evidence restrict local updates to the oldest 20 of 80 groups and recalibrate the sigma bands over those 20 groups. Full anchors still execute all 320 layers. Exact patch hashes were not captured and are not invented.

The six-layer proxy can provide three two-layer groups, so the parent-only analogue uses **one-third**, not exactly one-quarter, of the groups. This preserves the restricted-group mechanism but is not an exact scale ratio. Pythia's six layers are all pretrained, rather than newly expanded from an older ancestor. Results here cannot justify widening the production model's local group rotation.

For a controlled local-objective comparison, all hybrids deliberately use the same ordinary AR anchors. We do **not** port AGILLM's SAT-fixed, learned SAT-variable, and NAT anchor schedule, ternary weights, MoE routing, trainable shared table, rotating write-shard controls, or checkpoint/offload system. Consequently this study cannot establish whether the current full 512B configuration is better or worse than another full production configuration. Original source code, weights, and private operational logs are not redistributed.

## Frozen protocol

- Seeds 101, 202, 303; 648 equal-data updates, batch 2, context 32: 41,472 input-token exposures per method/seed.
- Additional endpoint at the first update reaching the same-seed ordinary control's training CPU budget; maximum 2,592 updates. Equal input data is not equal predicted-target count: masked steps score only selected tokens. The records include both counts and per-group updates.
- AdamW learning rate 1e-4 with 48-step warmup, betas (0.9, 0.95), weight decay 0.01, gradient clipping 1.0. Local loss above 35 is skipped and counted, reflecting the existing guard; nonfinite loss aborts the test.
- CPU accounting covers forward, loss, backward, clipping and optimizer. It excludes loading, warmup, input/noise preparation and evaluation. Wall time is recorded separately. Equal CPU is not simultaneous equal FLOPs, target count, data, or wall time.
- New 32 validation and 64 test windows exclude all v1/v2 scored contexts; the final 16 targets per window are scored: 512 validation / 1,024 test targets. All methods have the same causal evaluation, no target noise or future context.
- The metric is ordinary next-token NLL at context at most 32, **not** a finite diffusion-mixture score. Lower is better. Original pretraining overlap with WikiText remains unknown.
- All candidates and endpoints are reported. The prespecified practical screening margin is mean test degradation at most 0.05 nats/token versus ordinary training, together with conditional window-bootstrap uncertainty and a useful measured cost advantage. This small fixed benchmark is not a production noninferiority certificate.

`protocol.json` freezes the primary source and window hashes before adapted outcomes. V1/v2 are separate studies with different evaluation windows; do not compare their raw aggregate scores as if they were the same sample.

## Tests and cost evidence

`checks.py` verifies exact initial native-logit preservation, causal inference future exclusion, reserved-mask absence, removal of masked labels from inputs, intentional bidirectional dependence during masked reconstruction, causal-direct noisy-target/future exclusion, selected-group forward/gradient/optimizer isolation, frozen lexical maps, cadence, and parent-only versus all-group coverage.

`accounting.py` performs a separate actual file-loading test. It exports a shared vocabulary shard, ordinary body, masked local group, and causal-direct local group. Fresh probe processes construct six-layer ordinary or two-layer selected models, read only their designated files, and verify exact source logits before a real CE backward/optimizer step. Logical bytes, `/proc/self/io` counters, warmed time, supported matrix FLOPs, saved nonparameter storages, and process peak RSS are recorded. Weights remain local.

Primary runs keep weights resident. Hybrid weighted I/O figures are **modeled from measured shard sizes and the 8:1 schedule**, not observed offloaded main-run throughput. OS page caches are not flushed, physical reads depend on cache/readahead, optimizer-state transfer is excluded, and shared vocabulary maps dominate this small model. RSS includes loading/construction peaks. A selected local step is not a whole sweep of all groups.

`validate.py` checks every recorded checkpoint, exact trainable keys and finite tensors, unchanged pinned source/model/data, disjoint new evaluation contexts, six exact seed 101 held-out score replays, a native cached-prefix continuation check for each treatment, and selected-file loading parity.

## Reproduction and sources

Run `bash reproduce.sh` using Python 3.12 and the pinned requirements. `prepare.py` retrieves verified public safe-tensor weights with `trust_remote_code=False`, plus pinned [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext) text. Model revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`, Apache-2.0; dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`, CC-BY-SA-3.0/GFDL. Manifests record hashes. No weights, raw text, token arrays, or teacher caches are published.

The [DiffusionBlocks paper](https://arxiv.org/abs/2506.14202) and [official vision code](https://github.com/SakanaAI/DiffusionBlocks) motivate the causal-direct auxiliary. Pretrained conversion is a future direction in the paper, not a completed claim established by these tests. [V2](../pretrained_diffusion_v2) separately tests actual iterative finite-particle inference and shows its quality/cost limitations.

The runtime is FP32 on one CPU thread, with deterministic algorithms, autograd multithreading disabled, and process-local denormal flushing. The initial control launch and several later isolated attempts exited 137 before any adapted candidate endpoint. One ordinary control completed normally. Bounded stack traces showed long stalls inside parameter loading; the precise underlying cause is not established.

An explicit NumPy-before-Torch launcher now matches the completed v2 startup order. The primary scientific source files, data, objective, hyperparameters, and sampling stayed unchanged. Both the untouched evaluation and the already completed ordinary control replayed with exactly zero difference on every held-out window. Model loading then measured about 0.06 CPU seconds and full test evaluation about 1 second. Original protocol and control results are retained under `execution_history/`, and the launcher is separately hash-pinned. No manual FP32-loader diagnostic or temporary converted weight file is used in the final study.

No live process was changed, no checkpoint removed, and no compute rented.
