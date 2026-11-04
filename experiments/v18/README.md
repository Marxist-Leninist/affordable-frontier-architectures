# v18: pruning unreachable context, with a failed numerical control preserved

## Question and scope

v17 sped up long-prefix processing without establishing use of distant information. This run tests a consequence of the actual two-layer, span-128 local Transformer: each final prediction depends on at most **255 input bytes**, and the final 128 predictions depend on at most **382 input bytes**. Prefixes outside that receptive field can be skipped, but doing so does not create long-range capability. This is an inference implementation experiment, not a new architecture or frontier-quality result.

The matched comparator is v17's already optimized chunk-128 prefill with only the requested output-head positions computed, not v16's slower dense local mask. Both pruning and baseline use identical frozen weights, chunking and output-head optimization. Full-attention Transformer and parameter-matched gated GRU are additional diagnostic controls; they do not share this finite receptive-field guarantee. The statistical 5-gram control is rescored on exactly the same targets.

## Data and reproducibility

Fresh HuggingFaceFW/fineweb-edu `sample-10BT` training rows **300–399**, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, capped to first 4,097 UTF-8 bytes/document, no normalization or document joins. Nonoverlapping 2,048-input windows score identical final 128 next-byte targets. There are **98 windows / 12,544 targets**, shared across models/seeds. This deterministic viewer prefix is a bounded sample, not representative random sampling or a downstream task benchmark.

`data_manifest.json` pins revision, viewer response and per-document text hashes. `prepare_data.py` fails closed on changed content. Viewer URL is not immutable: if it changes, reproduction must recover the exact pinned snapshot rather than silently update hashes. Dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md . Dataset ODC-BY terms and underlying third-party content rights apply. No corpus text or checkpoints are published here.

Three frozen v14 checkpoint seeds; no training, tuning, model selection or paid compute. The local variant changes only attention connectivity, as in v16. Original modern two-block RoPE/RMSNorm/SwiGLU Transformer and parameter-matched residual gated GRU remain diagnostic controls. All compared implementations compute only needed output-head positions. Their training was short-context and small-scale: they are not strong frontier baselines. Prefix-pruning equivalence is within the same local model, not an equal-quality win over other architectures.

## Numerical failure and repair

The initial suffix-only implementation reset RoPE position to zero. Although relative RoPE shifts are algebraically invariant, float32 trigonometric evaluation is not bitwise invariant. **All three initial local runs failed the fixed strict gate** (max logit error <=1e-4 and every window loss difference <=2e-6 bits/byte). The exact failed scores, protocol and code are retained as `reset_position_*`.

The repair preserves absolute position counters in initially empty layer caches, skips only structurally unreachable tokens and keeps the original gate. It is a diagnostic repair on the same sample, not a new held-out confirmation. `pruning.py` derives required input count from each layer's window rather than hardcoding a cutoff. It is intended for this audited local model, not arbitrary models with global memory, pooling, routing or different positional rules.

## Quality results

Across three seeds, local full-prefix loss is **3.038811548 bits/byte**, versus **3.038811550** after pruning. All three repaired runs pass the original gate: maximum logit difference **2.52724e-5**, maximum per-window loss difference **2.82075e-7 bits/byte** across 294 seed/window comparisons. Sixty-seven documents contribute eligible windows. Original reset-position max logit error was **0.00189057**, well outside the gate.

Diagnostic full-attention Transformer changes **5.560003 → 5.134276**, and GRU **2.955912 → 2.955580**. Both fail strict equivalence, despite the GRU's small average loss change. These controls do not establish matched quality or that distant context is useless for recurrent models. The unchanged statistical control scores **3.379318 bits/byte**; its four-byte conditioning is already finite. Different architectures' losses are unequal, so no equal-quality cross-architecture speed claim is made.

Independent audit tests retained caches, boundary indexing, poisoned unreachable prefixes, alternate chunk boundaries, final-output pruning and 32-step continuation on real documents. The actual timing path (255-input final-one pruning plus 32 single-byte updates) passes across three documents/all three seeds: max logit difference 7.63e-6 and maximum 33-target mean difference 1.25e-7 bits/byte. Five model tests and seven economics tests pass. An additional, stronger per-single-token loss assertion initially failed at 2.63e-6; the declared gate concerns window-average loss, and output logits remain within tolerance. The failed extra assertion log is retained; this is not bitwise equality. Corrected scoring took 32.98 seconds, excluding failed control, statistical rebuild, audit and timing.

## Latency scope

Timing runs after scoring and independent CPU audits. One CPU thread, float32, batch one, same first eight eligible documents at 128/512/2048 input lengths, three seeds, six repetitions including two discarded warmups. Randomized variant order, seed18112026; paired comparisons share document, model seed and repetition. Every probe computes one prefix distribution and 32 distributions after observed byte updates. This is **teacher-forced CPU model work**, not sampled generation, concurrent serving, billing-token throughput or network/user latency. No CPU exclusivity is claimed. `timings.json` preserves every retained sample. Persistent-state bytes are measured; peak process RAM is not.

At 2,048 input bytes, baseline/pruned end-to-end p50 is **58.872  / 26.350 ms**; the median paired speed ratio is **2.146×**, with 95/96 pairs faster. Median paired prefill speed ratio is **8.589×**. End-to-end p95 falls 72.920→36.163 ms. Ratios of aggregate medians differ from median paired ratios; both are reported rather than conflated.

At 512 bytes, end p50 is 30.191→26.396 ms, median paired speed ratio 1.159× (81/96 faster). At 128 bytes, no prefix can be pruned:24.063→24.010 ms and 1.002× (49/96 faster), consistent with noise. The optimization is workload-dependent and its skip check has overhead.

All **576 retained samples  / 288 matched pairs** are preserved. Actual persistent cache storage stays **130,064 bytes**; this is skipped prefill work, not a new persistent-memory reduction. CPU: AMD EPYC 9V74, Python 3.12.14, PyTorch 2.6.0+cpu. Timings do not establish GPU speedups, peak RAM or user-perceived latency. The speedup is possible precisely because this model cannot use the discarded distant content.

## Economics: gates remain unproven

Official prices rechecked 2026-11-04 around 20:20 UTC: https://dev.meta.ai/docs/pricing-rate-limits lists Muse Spark Contributor input $0.10/M, cached $0.002/M and output $0.20/M. Contributor permits content use for training; it is not a privacy-equivalent substitute for Standard. The repository target remains half those rates: $0.05/$0.001/$0.10. Do not substitute Wonder's different product or presume equal task quality. https://www.runpod.io/pricing now says updated November4: H100 PCIe $2.89/h and SXM $3.99/h, storage extra; no inventory or deployment quote checked.

At **30% margin**, output-only full cost budget is **$0.07/M billable generated tokens**. Spending all of it on hardware requires **3,968.25 generated tokens/s per $1/h** at full utilization, **6,613.76 at 60%**. These are phase-only accounting floors; this experiment does not measure provider-billing tokens or full replica costs.

The same mixed illustrative ledger remains 10M uncached + 5M cached input, 2M visible + 2M hidden output; cached work assumed 10%. At $1/h, 60% utilization, 80k prefill tokens/s and $0.10 separate cost per whole ledger, required busy decode is **3,917.31 generated tokens/s** if hidden output is billed, **5,565.53** if not. At 8k prefill the budget is already exhausted. Input revenue and input work both count; an output-only floor is not necessarily a mixed-service lower bound. `economics_audit.py/json` preserve exact calculations and scenarios.

A $500 gross monthly plan at 30% margin and $150 assumed other allocation leaves **$200** serving budget, or **$141.67** with 20% included tax. Those are placeholders, not measured usage or profit. Include training/failed runs, hidden reasoning, tools, storage/network, payments, support, idle capacity and failover without double counting. Cheap prefix skipping is valid only if discarded information cannot affect the required task; this model's finite receptive field is exactly the capability limitation that makes it cheap.

No architecture has passed the frontier-quality, long-range-task or commercial cost gate. The next consequential test is a strong context-trained model on a task where distant information is necessary, with its quality and complete serving workload fixed. Further toy-prefix speedups alone would not close that gap.

## Reproduce

Install `requirements.txt` from the pinned reputable package sources into an isolated environment. Obtain the pinned dataset snapshots and reproduce v14 checkpoints/ledger as documented in v14, then:

```sh
python prepare_data.py
V14_DIR=../v14 python reset_position_control.py
V14_DIR=../v14 python run.py
V14_DIR=../v14 WIKITEXT_DATA=../v14 python statistical_control.py
V14_DIR=../v14 python -m unittest -v test_pruning test_thresholds
# Run alone, after other CPU work ends:
V14_DIR=../v14 python benchmark.py
python economics_audit.py > economics_audit.json
```

`artifact_manifest.json` hashes delivered artifacts. No production changes, rentals, purchases or external sharing occurred.
