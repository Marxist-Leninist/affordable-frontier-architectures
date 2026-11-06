# Anti-attention v8: fresh-document context replication

## Useful result

The v7 tradeoff replicates on100 fresh FineWeb-Edu documents: mixed training improves128-input-window loss relative to long-only training but worsens512-window loss, for both architectures in every seed. Document-paired95% bootstrap intervals exclude zero for all four prespecified primary contrasts. This strengthens a narrow generalization finding, not a frontier-quality or equal-quality speed claim. Frozen checkpoints were evaluated without retraining or tuning.

Signed mixed gains0.00494 bits/byte at128 while losing0.01594 at512 versus signed long-only. Transformer gains0.00789 and loses0.03177 respectively. Intermediate256 windows show little mean difference between mixed and long-only, rather than a broad mixed-training advantage. Short-only models already degrade severely at256 and fail further at512.

## Fresh heldout scores

Three-seed mean bits/byte; lower is better:

|Architecture|Training arm|128 input|256 input|512 input|
|---|---|---:|---:|---:|
|attention|short|3.324066|4.092432|5.040334|
|attention|long|3.352830|3.334654|3.346287|
|attention|mixed|3.344939|3.337657|3.378055|
|signed|short|3.299590|4.055531|5.075702|
|signed|long|3.311858|3.291272|3.295472|
|signed|mixed|3.306919|3.292203|3.311414|

All conditions score exactly the same128 target bytes per document. “128/256/512 context” denotes total input-window length, not equal history at each scored target: actual histories range1–128,129–256 and385–512 bytes. Position/window initialization changes with input length. This does not measure distant reasoning or isolate prefix content from position effects.

## Paired uncertainty

Mixed-minus-long bpb (negative means mixed is better). Average three seeds per document first, then5000 paired resamples of100 documents, fixed RNG seed8008; equal128 targets per document. Intervals are conditional on this deterministic100-document cohort and fixed trained models, not full uncertainty across training runs or the web. No multiple-comparison adjustment. All21 primary/secondary contrasts, per-seed deltas and per-document differences are disclosed in summary.json.

|Primary contrast|Mean delta|95% document interval|Documents favoring mixed|
|---|---:|---|---:|
|attention:mixed-minus-long@128|-0.007891|[-0.009485, -0.006263]|84/100|
|attention:mixed-minus-long@512|+0.031768|[+0.028370, +0.034993]|4/100|
|signed:mixed-minus-long@128|-0.004939|[-0.006180, -0.003698]|78/100|
|signed:mixed-minus-long@512|+0.015942|[+0.014045, +0.017813]|4/100|

No prespecified practical noninferiority margin establishes quality preservation. Signed-vs-Transformer secondary comparisons use matched architecture/training target budgets and seeds, but not equal FLOPs or converged/frontier-scale models. Architecture claims remain limited to these131,904-parameter byte predictors and training recipes.

## Dataset and model provenance

HuggingFaceFW/fineweb-edu sample-10BT upstream train split, pinned revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, new rows1600–1699. Public official endpoint, no gated terms or private data; retrieval validated revision header and bounded2MB response. Response SHA25627e084092b0c8f9323364a5410acfb708248d20029e9cf8597c12386383d2fcc,693207 bytes. Dataset card/metadata identifies educational English web text filtered from FineWeb/Common Crawl and ODC-BY; underlying web-text rights remain applicable. Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu .

First513 UTF8 bytes per document, no normalization/joining or loss-based selection. All100 rows qualify; zero exclusions. Score final128 bytes,12,800 targets per model/context,54 model/context rows. No exact full-document duplicates against1600 distinct prior FineWeb documents or within this cohort. No exact128-byte scored targets in prior FineWeb, raw WikiText training, or the actual line-deduplicated first2,000,000 training bytes (hash77fc488d9c24735f44714b30f6c8507e61c16369d278810cffb0decb5df5f4af). All100 current targets are unique. Shorter/near-duplicate contamination remains unmeasured; this is not exhaustive decontamination.

The original training data is Salesforce/wikitext wikitext-2-raw-v1 revisionb08601e04326c79dfdd32d625aee71d232d685c3. Source hashes and the source card's CC-BY-SA3.0/GFDL metadata versus CC-BY-SA4.0 prose inconsistency remain inherited; no new license interpretation is asserted.

All18 checkpoints are frozen v7 short/long/mixed ×attention/signed ×seeds1–3. Their hashes match the manifest published before v8 in commit607d1237365e6dc4a8df9e42d410f290f13b453c. Initial scoring recorded and independently verified hashes; the loader was then hardened to check that existing manifest before loading. No weight selection after scores. Original U−A checkpoints are converted by negating signed output projections only for optimized inference. Eighteen full/optimized/cache/causal checks and exact target alignment pass. Numerical checks use two33-byte sequences in FP64, not exhaustive coverage of all lengths or the FP32 timing path. Dataset regeneration now fails closed if prior-cohort receipt lists or source hashes differ.

## Fresh inference probe

Seed2 mixed-trained weights; two CPU threads, batch4 first4 eligible documents;5 warmups+31 randomized interleaved repetitions. Full-logit prefill and one next-token decode after the stated prefix are separate. Decode excludes prefix construction; neither is end-to-end service latency. No retraining or new training-speed claim. v7's matched training measurements remain prior evidence, not rerun here.

|Prefix|Probe|Attention ms|Signed ms|Attention bytes/s|Signed bytes/s|
|---:|---|---:|---:|---:|---:|
|128|prefill|3.7032|3.8078|138257.7|134460.8|
|128|decode|0.9180|0.9388|4357.1|4260.8|
|512|prefill|18.0648|19.1035|113369.9|107205.3|
|512|decode|1.1354|1.1429|3522.9|3499.8|

## Economics and scope

New training targets:0. Quality evaluation scores691,200 target predictions (54×12,800) in14.82s measured evaluator wall time, including preparation but excluding external download, separate tests, timing and report work. Prior training/search and failed-pilot costs remain in v7 accounting, not amortized to zero. No new paid resources, production changes or external-agent messages. Actual invoice/hardware rate and all-in cost per useful commercial token remain unmeasured.

Retain$0.05/M uncached input,$0.001/M cached input,$0.10/M output and30% margin on net receipts. Conditional output-only hardware floor3968.254 billable output tokens/s per$1/hour at100% productive utilization is not a mixed-workload or byte-throughput gate. See economics.md for explicit tax/fee, utilization, prefill/decode/cache, memory, burst/reliability, non-billable reasoning/tools, training amortization and operations accounting. No billable-token conversion, frontier-quality match or full workload SLA supports a numerical commercial gap calculation.≤$500/month subscription keeps full workload and30% net margin; no profitable scaling claim follows.

## Reproduce

Python3.12, torch2.6.0 CPU, numpy; pyarrow for pinned WikiText download. Restore original WikiText with download_wikitext.py and set ANTI_DATA_DIR to its data directory. download_data.py restores/hash-checks the fresh snapshot plus16 required prior snapshots into the recorded sibling directory names (maximum34MB if all missing). It fails closed on source revision/response changes. Raw corpora are excluded from repository publication.

Restore frozen v7 checkpoints using v7's published reconstruction instructions, then set ANTI_V7_CHECKPOINT_DIR to that checkpoint directory. expected_checkpoint_hashes.json is the prior v7 manifest; exact serialization may depend on runtime, and hash failures are never silently refrozen. Run evaluate_fresh.py under an external300-second timeout, check_provenance.py, summarize_v8.py, profile_v8.py and validate_v8.py, then report_v8.py. Numerical results, per-document losses, bootstrap deltas, timings, receipts and tested code are included. New source hardening did not change data, model weights or recorded scores.
