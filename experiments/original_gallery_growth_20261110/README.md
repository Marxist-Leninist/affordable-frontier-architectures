# Original gallery and parameter-growth CPU studies

Report-only release, 10 November 2026. This folder contains methods, derived scalar results and a provenance manifest. **It does not contain runnable study code, checkpoints, raw data or a self-contained reproduction package.** [METRICS.json](METRICS.json) retains per-seed results; [MANIFEST.json](MANIFEST.json) records provenance and file hashes. The two studies below are separate cohorts with different model sizes and CPU clocks.

**Limits first:** these are tiny byte-level CPU experiments on reused discovery heldouts, not fresh confirmation. Three seeds do not establish a universal winner, a twofold intelligence or time-to-quality gain, production serving economics, GPU scaling, a 600B/603B result, or a long-term growth guarantee. Monetary training and self-hosted serving costs were not measured.

## 1. v65: exact original Fusion, DLGM and a Transformer

The main comparison uses three paired seeds each for original #879 FusionGatedFIRNet (349,956 parameters), #458 DualScaleMemoryRouter (349,944) and the corrected-RoPE T96 Transformer (349,920). Each run stops at the first completed update crossing 120 training CPU seconds. Actual endpoints range from 120.014 to 120.210 seconds. This training clock covers minibatch construction through optimizer update and finite checks, but excludes checkpoints, hashing, I/O and evaluation; those costs are accounted for separately.

Original gallery program bytes were pinned to [ASI-Arch commit 3113c51](https://github.com/GAIR-NLP/ASI-Arch/blob/3113c51b04a1d4a7520a3b0238c6a08d3194d58f/data/106.json) and remain distinct from a repaired cache, #908 AdaptiveFusion or #1354 FusionGate-MS. The CPU loader removes top-level torch.compile decorators and uses narrow CPU FLA adapters. This is not a claim of official fused-kernel numerical parity. Models use BF16 CPU autocast with FP32 master weights and AdamW, CE-only training, learning rate 0.003, betas 0.9/0.999, weight decay 0.01 and gradient clipping at 1. The inherited common update-based learning-rate schedule and capacity matching are not FLOP matching or optimizer tuning for each architecture.

Evaluation uses fixed WikiText windows (140 windows from 38 articles), then 98 FineWeb-Edu documents, scoring the final 128 byte targets. Complete checkpoint state is restored between repeated evaluations. Full-sequence training and state-replay audits passed. Cached serving is a separate component: original gallery caches remain unsupported; T96 serving checks failed for seeds 1 and 2 and passed for seed 3. There is no valid comparative serving-performance result.

Mean final cross-entropy in nats per byte target, lower is better:

| Model | WikiText CE | FineWeb-Edu CE |
| --- | ---: | ---: |
| #879 FusionGatedFIRNet (original full forward) | 1.771213 | 2.124353 |
| #458 DualScaleMemoryRouter | 1.762221 | 2.103326 |
| T96 Transformer (corrected RoPE) | 1.829924 | 2.188071 |

Fusion is lower than T96 on both corpora in all three seeds. DLGM has lower mean loss than Fusion, but their ordering changes with seed: Fusion is slightly better in seed 1 on WikiText and seeds 1–2 on FineWeb-Edu, while DLGM's seed 3 advantage changes the mean. Conditional cluster intervals in the JSON are conditional on these seeds and reused clusters, not broad architecture guarantees. No main arm attained the frozen joint quality targets at two consecutive sampled checkpoints, so sampled time-to-quality remains right-censored and does not support a 2× speed claim.

Separate seed-1 screens tested gallery #1081, #750 AdaptiveSpanGateConv and #890. They raise the completed original-gallery coverage snapshot from 22 to **25 of 105**, not 105 trained models. Their metrics are separated in the JSON; the growth variants below are not new gallery candidates.

## 2. R4: preserving parameters while adding blocks

This separate study starts with a tiny DLGM-derived model: vocabulary 256, hidden width 32, two heads, FFN width 64, untied embeddings and one initial block. It grows from **31,032 → 45,649 → 60,266 parameters** at 32,768 and 98,304 completed byte targets (updates 64 and 192). Five arms use three paired seeds each:

- S: fixed small, one block.
- L: fixed large, three blocks with gates initially 1.
- Z: fixed large, three blocks with the added gates initially 0.
- G: add zero-gated blocks while retaining learned parameters and existing Adam state.
- R: the same growth schedule, but reset all model parameters and Adam state at each growth event while keeping data, learning-rate and global clocks running.

Training uses four sequences of 128 targets per update, next-byte CE, BF16 CPU autocast with FP32 masters, AdamW at 0.003, 32 warmup updates, then constant learning rate. New parameters start with zero moments at the current global learning rate. Data and heldouts are pinned real WikiText/FineWeb-Edu byte text, not synthetic language. The experiment has no 31k-parameter Transformer control. Its CPU endpoint is training-job process-tree CPU including frozen inline checks and checkpoint costs, excluding separately charged offline audits. **Do not compare this clock directly with v65.**

At the first committed state crossing 60 worker CPU seconds:

| Arm | WikiText CE | FineWeb-Edu CE | Mean worker CPU seconds |
| --- | ---: | ---: | ---: |
| S: fixed small | 2.07768 | 2.39845 | 60.05120 |
| L: fixed large | 2.11655 | 2.40173 | 60.09594 |
| Z: fixed large, zero gates | 2.15502 | 2.42014 | 60.08971 |
| G: preserving growth | 2.11123 | 2.40091 | 60.13370 |
| R: reset growth | 2.18733 | 2.43683 | 60.08975 |

G beats reset-growth R on both corpora in all three seeds. S beats G on WikiText in every seed. G versus L is seed-sensitive; the conditional bootstrap interval for the primary WikiText difference G−L is −0.01241 to +0.00108, including zero. This supports preserving learned state over destructive resets here, not universal superiority of growth over a small or large fixed model.

At exactly 131,072 training targets:

| Arm | WikiText CE | FineWeb-Edu CE | Mean worker CPU seconds |
| --- | ---: | ---: | ---: |
| S: fixed small | 2.29949 | 2.51961 | 21.79787 |
| L: fixed large | 2.21382 | 2.46373 | 44.81079 |
| Z: fixed large, zero gates | 2.25237 | 2.49009 | 44.56737 |
| G: preserving growth | 2.26273 | 2.50095 | 33.35581 |
| R: reset growth | 2.69923 | 2.81016 | 33.12442 |

G sits between S and L in mean quality and compute. Adding zero-gated blocks preserves immediate heldout CE exactly at both boundaries in all three G seeds; resetting increases immediate CE by about 2.92–3.43 nats. This boundary result is not a guarantee of lifelong retention, unchanged post-clipping gradients, or unchanged future optimizer updates.

### Descriptive fronts and resource accounting

The frozen quality_resource fields are empty because their eligibility partition selected no points: these arms have retention measurements and appear in a separate retention-available partition. This is not a bug in dominance arithmetic and does not mean every arm was dominated.

A separately labeled **post-hoc** calculation minimizes WikiText CE, FineWeb-Edu CE, worker CPU, active wall time and peak RSS across the five saved arm means. At the 60-CPU-second endpoint all five arms are nondominated. At 131,072 targets S, L, Z and G are nondominated, with S dominating R. These finite mean-row descriptions are not statistical dominance, practical equivalence, or a global optimum; tiny resource differences can determine membership.

All 31 study phases completed. The final inclusive guarded ledger is 1,841.247070624 CPU seconds and 1,766.296201303963 active-wall seconds, below the 3,600/5,400 caps. It includes recorded failed preparation, accounting, review and extraction costs. These are guarded accounting totals, not exact physical consumption and not monetary costs.

## Data, provenance and interpretation

[ASI-Arch source](https://github.com/GAIR-NLP/ASI-Arch/blob/3113c51b04a1d4a7520a3b0238c6a08d3194d58f/data/106.json) is Apache-2.0, at commit 3113c51b04a1d4a7520a3b0238c6a08d3194d58f, data/106.json Git blob e403531e663cf24c800002720d293cccf77f94df. See the upstream [AlphaGo Moment for Model Architecture Discovery paper](https://arxiv.org/abs/2507.18074). Paper-scale results and original source identity are context, not reproduced frontier-scale results in this report. No source license or repository permissions are changed by this report-only addition.

WikiText uses Salesforce/wikitext, wikitext-2-raw-v1, revision b08601e04326c79dfdd32d625aee71d232d685c3. FineWeb-Edu uses HuggingFaceFW/fineweb-edu, sample-10BT; the recorded response revision is 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. Heldouts are reused discovery data. Only 1,900 of 2,100 historical document hashes remain available, and missing historical coordinate ledgers prevent a complete independent exclusion rerun. There is no semantic-decontamination guarantee. The manifest binds the evidence used to derive this summary but does not make the underlying private study archive downloadable or reproducible from these three files alone.

## Economic limits, not an economic result

The last verified [Meta Contributor rates](https://dev.meta.ai/docs/pricing-rate-limits), as of 10 November 2026 at 14:19 UTC, were USD $0.10 input, $0.20 output and $0.002 cached input per million tokens. Targets of $0.05/$0.10/$0.001 with 30% net margin remain targets. No serving cost model or monetary training cost was measured. A ≤$500 subscription target and £200/£20 stretch goals remain unproved; currencies must not be combined. Contributor content-use and resale terms matter, and no resale authorization is established here. Tiny byte-level CPU training losses cannot establish frontier API economics.
