# v23: training at the evaluated context repairs much of the Transformer deficit

Bounded isolated CPU research,2026-11-05. Fresh held-out real-text cohort; no paid compute, production changes, corpus or checkpoint publication.

## Results

Means across all three fixed seeds, lower bits/byte is better. The512 and2048 columns score **identical target bytes from identical windows**. The512 path resets to the final512 inputs; it is not a separate short-document sample.

|Condition|Nested512 input|Full2048 input|
|---|---:|---:|
|initial|3.747093|4.357838|
|context2048|3.475771|3.563760|
|reset512|3.230214|4.572721|
|frozen_local|3.183280|3.183280|
|frozen_gru|3.101030|3.101269|

Full2048 training beats the equal-target reset512 training control at2048 input by **1.008961 bits/byte**; paired document-cluster95% bootstrap interval for context2048 minus reset512 is [-1.043653597213998, -0.9749187691457795]. All per-seed effects, per-window/target losses, training curves and completed conditions are retained. The intervention is context/reset structure, with exactly the same training targets, update count, optimizer and initial weights. It is **not equal FLOPs, time or evidence of convergence**.

At nested512 evaluation, the reset512 control is better (3.230214 versus 3.475771 bpb), an important length-specific tradeoff rather than a universal quality improvement.

The context2048 model's full2048-minus-nested512 difference is 0.087989 bpb, interval [0.07503094913588895, 0.10172045902242034]. Positive means longer context still hurts on these matched targets. Do not equate this with intrinsic Transformer failure or general long-distance reasoning. This study addresses the earlier evaluation-length mismatch, not whether tiny models have been adequately scaled/trained.

Frozen local/GRU references receive no new training. Their comparisons are **unequal-training descriptive references**, not an equal-budget architecture contest. They have different original training compute and context capabilities. No architecture winner, frontier-quality parity, distant-information reasoning success or production margin is established. Seeds share evaluation documents and are not independent dataset replications.

## Frozen experiment

- Three seeds1,2,3 begin from frozen v19 full Transformer adapt500 checkpoints. The same two-layer64-wide causal RMSNorm/RoPE/SwiGLU architecture is used in both continuation arms.
-250 updates each; one contiguous2048-input WikiText sample per update. Reset512 independently reshapes x and shifted y into4×512 segments, retaining boundary next-byte targets. Exactly512,000 new target presentations per arm/seed,3,072,000 across six runs.
- Fresh AdamW, peak LR.001, betas.9/.95, weight decay.01, gradient norm clip1, warmup25, cosine to.1×. Fixed ledger seed23112026 and protocol/source hashes written before tests/training. No evaluation-based tuning, checkpoint selection or early stopping selection.
- FineWeb-Edu fresh rows700–799, pinned revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, first4097 UTF-8 bytes/document, no joins/normalization or content selection. 105 nonoverlapping2048-input windows from 66 eligible documents; 13440 final next-byte targets per condition/seed. Nested512 effective prediction context385–512 bytes; full2048 effective context1921–2048.
- No copy hybrid fitted here: this isolates the neural context-training intervention. Existing memory/quantization results remain separate evidence.

## Resources and validation

All six continuation runs completed,1,500 optimizer updates. Training used 183.34 CPU seconds; complete pre/post evaluation and training process used 249.02 CPU seconds excluding startup, preparation and small audit/report work. Hard main-process limit820CPU seconds; the main budget bound includes import CPU, but auxiliary startup/shell CPU was not separately instrumented, so an exact inclusive CPU total is not claimed. Measured work is far below the900-second overall allowance. One CPU thread, float32 parameters, float64 loss normalization. Shared host, no exclusivity claim. No additional latency benchmark is reported: changing weights does not itself establish a new serving-speed result and quality comparability remains unresolved.

No failed training or scoring runs occurred. Six pre-training tests passed: ledger/target preservation, paired initial checkpoint identity, causal future-suffix invariance, optimized/full-head scoring agreement, nested target alignment and frozen protocol/source hashes. Post-run audit verifies all30 aggregate rows from stored per-target losses, all six final hashes and complete training counts. Final2048 causal future-suffix logit error 0; optimized/full-head logit error 0. Every target is predicted only from earlier/observed bytes by causal masking; teacher-forced evaluation is not sampled generation.

The prespecified summary averages seeds then performs5,000 paired document-cluster bootstrap resamples (seed23), preserving within-document windows. Intervals are conditional on this deterministic convenience cohort; they are not generalization guarantees. The inherited0.02-bpb descriptive screen is not formal statistical equivalence.

## Data, reproducibility and publication limits

See DATA_CARD.md, data_audit.json and retained pinned cards for provenance, license and restrictions. WikiText train revisionb08601e04326c79dfdd32d625aee71d232d685c3, UTF-8 byte vocabulary0–255. Full-document checks found no within-cohort duplicates, no exact matches to prior v15–v21 manifests, and no whole evaluation document occurring in WikiText train. Near-duplicate/passage contamination remains unmeasured. Fresh sample is bounded and not representative random sampling.

Prepare existing sibling frontier_research_v19/checkpoints, frontier_research_v14/data_manifest.json and frontier_research_v8/train.txt using their pinned instructions. The local dependency path is /tmp/frontier-v10-packages:/tmp/frontier-v8-packages. Install requirements in an isolated environment for other systems. Then:

```sh
python prepare_data.py
python run.py --freeze
python -m unittest -v test_v23
python run.py
python summarize.py
python audit_results.py
python check_data.py
python write_report.py
```

Do not replace historical runs. Corpus rows.json, checkpoints and Python caches are excluded from repository/archive delivery. train_starts.npy is an index ledger, not text; losses.npz contains scalar loss values, not corpus or weights. Reproduction requires separate lawful data access and original checkpoint creation; dependencies alone do not reproduce inherited training for free.

## Economic boundaries

Requested targets remainUS$0.05/M uncached input,US$0.001/M cached input,US$0.10/M billable output and30% margin on net revenue. A conditional output-only hardware budget isUS$0.07/M, requiring3,968.25 billable generated tokens/s perUS$1/h at100% productive utilization, divided by utilization otherwise, before other costs. Byte loss/CPU training here does not measure commercial-token throughput or cost per useful/billable production token.

The separate≤US$500/month subscription must preserve the same useful average/heavy full-month workloads and capability while total attributable cost≤70% of net revenue. GBP200/GBP20 tiers remain separate. Prefill/decode, cached work, hidden reasoning/tools, storage/network, idle/burst/failover capacity, tax/payment/support and training amortization remain real costs. Pricing recheck, full economic gates and private-repository archival are handled in the parent report. This experiment demonstrates a context-training repair at tiny scale, not AGI, frontier quality or sustainable profit.
