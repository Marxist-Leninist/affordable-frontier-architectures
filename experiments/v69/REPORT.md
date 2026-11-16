# v69: Exact single-step specialization narrows but does not close decode gap

2026-11-16. Following v68's measured Gated DeltaNet decode overhead, replaced only its one-token recurrent update's Python loop/einsum with equivalent batched matmul and a rank-one update. No weights, training data, model state size or full-sequence prefill kernels were changed. This is a bounded independent CPU implementation optimization, not a new architecture.

| Model/path | Batch | median 32-step decode ms | median combined prefill+decode ms |
|---|---:|---:|---:|
| Transformer unchanged | 1 | 19.448 | 27.421 |
| Gated DeltaNet original | 1 | 39.093 | 50.979 |
| Gated DeltaNet specialized | 1 | 34.410 | 45.774 |
| Transformer unchanged | 4 | 34.224 | 64.122 |
| Gated DeltaNet original | 4 | 42.655 | 69.338 |
| Gated DeltaNet specialized | 4 | 39.899 | 65.885 |

Ratio of medians gives 12.0% / 6.5% lower DeltaNet decode time, batch1/4. Median paired speed ratios are 1.114x/1.102x; 12/15 paired trials favor specialization in each batch. These are descriptive shared-host measurements, not significance or production-tail claims. Specialized DeltaNet remains slower than the Transformer on this decode workload. Combined latency uses median sample sums, not sums of component medians. Prefill uses unchanged code; timing variation there is noise, not optimization credit.

## Provenance and controls
Same six step800 frozen checkpoints as v68: 3 seeds each, AAAwide Transformer 190,848 parameters and GGG Gated DeltaNet 165,996. Same first four v63 FineWeb-Edu windows, 480-byte prefill and 32 teacher-forced continuation inputs, batches1/4. Byte tokenizer, 256 IDs, FP32, single CPU intra/inter-op thread. Two warmup rounds and five measured rounds, 18 cells per round, randomized seed69 order. All90 measured cases passed predeclared full-forward equivalence tolerance atol/rtol1e-4; maximum logit discrepancy 7.75e-6. Checkpoint/model-state and protected source/data hashes unchanged. No new training or model selection. Full-forward reference does not invoke the specialized one-token path.

HuggingFaceFW/fineweb-edu sample-10BT revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, original source rows100000–100099, existing v63 retained windows. Dataset card rechecked https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu on2026-11-16: ODC-BY database licence, public English educational web corpus. Underlying source rights/limitations remain; no raw text redistributed. Exact sampling, window/data hashes, historical checkpoint training provenance and contamination checks are in protocol/source manifests and the preceding v63/v67/v68 reports. This cohort has been evaluated before, so not a fresh held-out quality replication. The numerical gate supports preservation here, not universal proof. Historical Transformer/GGG losses remain separate and do not establish matched frontier quality.

Existing torch2.6.0+cpu runtime. Runner bounds300CPU seconds/360wall/6GiB, no GPU/network/subprocess/production writes. Receipt10.201CPU seconds/10.255wall seconds includes validation/load. The initial command without `-B` was stopped by the write guard before an attempted Python bytecode-cache write outside scratch; rerun with `python -B run.py` completed. No guard removed. Parent notified before timings; shared host cannot guarantee absence of unrelated contention. Arithmetic audit verifies90unique keys. No independent peer audit.

## Economics: unchanged target, no commercial pass
Primary Meta pricing https://dev.meta.ai/docs/pricing-rate-limits rechecked2026-11-16, unchanged: Muse Spark1.3 Contributor $0.10 uncached input/$0.002 cached input/$0.20 output per million. Required targets remain$0.05/$0.001/$0.10 respectively. Contributor discount exchanges content-training permission; it is not underlying cost or permission to send user content. No API call/content transmission.

Gross mixed-token revenue=(.05U+.001C+.10O)/1e6. Deduct applicable taxes and payment fees once to define net revenue; all attributable cost must be <=70% of net for30%margin. The conditional output-only hardware floor remains3968.25*H billable output tokens/s per paid wall-clock hour for H$/hour; busy throughput must divide by utilization once. This run measures teacher-forced bytes, not billable BPE tokens or autonomous output. No rental price attributable to this scratch host and no useful-quality production throughput exists, so cost/token and gate gap are UNMEASURED, not zero.

Account for prefill/decode/cache creation/hits, weights/state/transients, hidden reasoning/tools, bandwidth/storage, idle/burst/reliability capacity, concurrency/latency, support, tax/fees and amortized training. The economic scenario ledger is retained in economics.md, with separate GBP200/GBP20 stretch tiers. No workload or quality reduction credited. No frontier/AGI/equal-quality cost win established.

## Publication and reproduction
The repository was initially found public while the scheduled task specified private-only publication. Scott subsequently explicitly authorized public upload of this tested code/report/results, so this release contains no secrets, raw dataset text or model binaries. No visibility or access change was made.

`python -B run.py` in a fresh `frontier_research_v69_single_step` directory beside the historical `frontier_research_v56_prediction_recovery` and `frontier_research_v63_fresh_context` prerequisites reproduces the test. Those authenticated runtime/frozen model/data prerequisites are not included in this release. This is not a standalone turnkey reproduction bundle. The exclusive attempt marker intentionally prevents overwriting existing results; use a fresh copy for a rerun. Raw timing JSON permits independent arithmetic checking without those dependencies.
