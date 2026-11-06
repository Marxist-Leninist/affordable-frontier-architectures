# Anti-attention v6: optimizer-retention replication

## Useful result

Retaining Adam state improves all four heldout metrics in both architectures in each of three seeds. Seeds 2/3 are new experiments; seed 1 is transparently reused from v5. All four new 800-step reconstructions match historical weights exactly (max error 0). No new hyperparameter search or model selection was performed.

Three-seed means, bits per byte (lower is better):

|Architecture|Metric|Reset|Retained|Improvement|
|---|---|---:|---:|---:|
|attention|wiki_128|2.822252|2.800797|0.021455|
|attention|wiki_512|2.866416|2.792068|0.074347|
|attention|fineweb_128|3.387685|3.371280|0.016405|
|attention|fineweb_512|3.423462|3.364695|0.058767|
|signed|wiki_128|2.777715|2.768818|0.008897|
|signed|wiki_512|2.771184|2.751549|0.019635|
|signed|fineweb_128|3.351137|3.342705|0.008432|
|signed|fineweb_512|3.339175|3.323649|0.015526|

This strengthens the earlier single-seed optimizer result, not a frontier-quality claim. The retained attention baseline benefits more than signed suppression; comparing only against a reset Transformer would exaggerate signed's advantage. A retained-optimizer context128 control is still absent, so optimizer×context interactions remain untested. Three seeds and reused small heldouts cannot establish broad robustness.

## New performance probes

CPU torch2.6, two threads, seed2 retained-model weights converted exactly to the optimized implementation. Batch4; 5 warmups, 41 randomized interleaved repeats for each architecture. Prefill emits logits over the full input; decode is one next token after the stated prefix (cache preparation excluded). Training probe includes zero-grad, forward, CE, backward, clipping and fresh AdamW. Only disposable in-memory training copies change. Fresh optimizer timing is not a retained-optimizer throughput test. Inputs are real training text; no test-data gradient updates. These tiny CPU timings are not GPU or service benchmarks.

|Context|Probe|Attention ms|Signed ms|Signed/attention time|
|---:|---|---:|---:|---:|
|128|prefill|5.0734|5.4888|1.0819|
|128|decode|0.8293|0.8053|0.9710|
|128|training|14.7209|14.6487|0.9951|
|512|prefill|17.3463|17.7475|1.0231|
|512|decode|1.1967|1.2097|1.0109|
|512|training|63.0557|60.2044|0.9548|

Measured throughput equivalents (bytes/s, batch aggregate, not commercial tokens):

- Context 128, prefill: attention 100919.0; signed 93280.9.
- Context 128, decode: attention 4823.4; signed 4967.3.
- Context 128, training: attention 34780.5; signed 34951.8.
- Context 512, prefill: attention 118065.4; signed 115396.6.
- Context 512, decode: attention 3342.6; signed 3306.5.
- Context 512, training: attention 32479.2; signed 34017.4.

Do not infer an architecture-wide speed win from a marginal timing difference. Raw timing arrays are in profile_v6.json. Architecture quality differs, and commercial token conversion / full-workload service cost are unknown. See economics.md for the unchanged 30% net-revenue margin and output-only versus mixed-workload gates.

## Protocol and provenance

Both tiny 131,904-parameter architectures use identical seed initialization, training bytes, optimizer steps, explicit LR, clipping and weight decay. For each new seed, reconstruct 800 steps of original 128-context training (819,200 targets/model). Fork identical parameters into reset/retained AdamW continuations: 200 steps, batch4 × context512, 409,600 targets/arm, same batches. Retaining state changes moments and step counters together. New checkpoint files preserve model and optimizer; retained step counters are1000 and reset counters200. Original U−A projection convention is retained in files. If converting optimizer states to optimized A−U, also negate the output projection first moment.

Salesforce/wikitext, wikitext-2-raw-v1, pinned revision b08601e04326c79dfdd32d625aee71d232d685c3, CC-BY-SA4.0 card. UTF8 byte tokenizer, first2,000,000 line-deduplicated training bytes; loaders validate source hashes. WikiText test slice starts196608, 128 windows; final128 targets scored at contexts128/512 (16,384 targets). FineWeb-Edu sample-10BT train-split rows1500–1599, revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, ODC-BY dataset card with underlying web rights caveat, frozen first64 eligible windows (8,192 targets). FineWeb is out-of-training evaluation, despite its upstream split name. Exact line/block exclusions are checked, not exhaustive near-duplicate removal. Frozen evaluation cohorts were reused from v4/v5 and are exploratory rather than newly untouched data. Windows within documents are correlated. No test data enters quality training. Raw corpora are not redistributed.

## Reproduce

Python3.12, CPU torch2.6.0, numpy, pyarrow for download. Source hashes/revisions fail closed. download_wikitext.py and fineweb/prepare_data.py reproduce data. ANTI_DATA_DIR may point to matching WikiText text files; ANTI_FINEWEB_ROWS may point to matching rows.json. Run restart_control.py, profile_v6.py, validate_v6.py, summarize_v6.py, report_v6.py. Summarization includes historical_seed1.json explicitly as prior evidence. No production or paid-resource changes were made. This is the text-only repository edition prepared after publication authorization; original run artifacts were first preserved locally. Historical reports retain their original publication-status statements.

## Accounting

New quality training presents 6,553,600 target bytes; reconstruction wall time 78.53s; continuation optimizer steps 94.98s. Those times omit evaluation, checkpoint, process and other overhead. Performance warmup/timing executes an additional 235,520 target bytes on disposable copies; this is separate probe work, not part of quality training. Full invoice/hardware price was not observed. No dollar-cost or frontier economic pass is claimed.

## Text-only repository edition

Raw datasets, binary model/optimizer checkpoints and Python caches are intentionally excluded. Recorded numerical results, training histories, checks, provenance hashes and reproducible source are included. To reconstruct the original six selected checkpoints without a binary input, run the included baseline.py on the hash-verified WikiText data, then copy attention_{1,2,3}_0.006.pt and signed_{1,2,3}_0.006.pt into checkpoints/. baseline.py performs the full historical validation-only learning-rate search, including learned gating; it is more compute than the v6 replication and is not part of its reported timing. Then run restart_control.py and the v6 profile/validation/summary commands described above. The v6 reset/retain checkpoint files are regenerated with optimizer state. Historical seed1 numerical evidence is retained, not rerun by restart_control.py.

Repository edition's artifact_manifest.json hashes only included text files; checkpoint_hashes.json preserves expected SHA256 for the excluded original/reproduced binary artifacts. Exact byte serialization can vary across library environments; max-parameter reconstruction checks are the scientific parity gate. Original runtime: Python3.12, torch2.6.0+cpu.
