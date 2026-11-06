# Anti-attention v9: does earlier prefix content help?

## Useful result

On 95 fresh recipient/donor document pairs, replacing the first 256 bytes with unrelated real text changes neural predictions but yields no clear aggregate genuine-prefix advantage for any of the four context-trained neural configurations. All four primary 95% document-pair bootstrap intervals include zero. This is limited evidence, not proof that these models ignore earlier text or that its true effect is zero.

The fixed exact-copy reference does benefit from the original prefix by 0.306895 bits/byte [0.235807,0.377445], showing that exploitable repeated text exists in this cohort. Its total loss is much worse than the neural models, so it is a diagnostic reference, not a competitive replacement or evidence of frontier reasoning.

Removing the earlier prefix improves mean neural loss in every configuration on this cohort and reduces measured prefill work. That is a small-model, cohort-specific observation, not permission to prune useful user context or a demonstrated general equal-quality optimization. The training-context gains in v5–v8 do not alone establish useful distant-content processing.

## Frozen intervention and quality

New FineWeb-Edu recipient rows 1700–1799. Recipient row 1700+i receives the first 256 bytes from donor row 1600+i, without reuse. Replace only bytes 0–255; retain recipient bytes 256–512, input length, and target positions exactly. Original/donor512 and genuine suffix256 all score recipient bytes 385–512 (128 targets). At every scored prediction,129–256 immediately preceding recipient bytes remain unchanged between original and donor. The donor introduces an artificial incoherent boundary; the test does not isolate semantic retrieval or generalize across all donor choices.

Frozen v7 attention and signed U−A models, retained-Adam long/mixed training arms, three seeds. Same architecture parameter count 131,904 and historical target/update budgets; not equal FLOPs or converged models. No new training, tuning or loss-based selection. This extends earlier donor diagnostics to the newly context-trained signed models with matched Transformer, a disjoint fresh cohort, full-distribution sensitivity, and exact-copy reference.

Mean bits/byte over three seeds, lower is better:

|Model|Training arm|Original512|Donor512|Suffix256|
|---|---|---:|---:|---:|
|attention|long|3.322843|3.322607|3.314395|
|attention|mixed|3.353320|3.352133|3.316699|
|signed|long|3.273633|3.275489|3.270067|
|signed|mixed|3.289569|3.290934|3.270679|
|Exact copy|No neural training|6.902845|7.209740|7.337538|

Suffix truncation changes available history and, for signed models, the population contributing to the cumulative value mean. Pure RoPE is shift-invariant; merely reindexing a suffix is not an absolute-position confound. Original/donor is the length-preserving content intervention; original/suffix is a separate operational comparison.

## Conditional paired uncertainty

Negative original-minus-donor delta favors genuine earlier text. Average the three model seeds within each pair first, then 5000 resamples of 95 disjoint recipient/donor pairs, fixed seed 9009. All 10 primary/secondary contrasts are retained. Intervals are conditional on this deterministic cohort, fixed donors and checkpoints; they do not include training-run uncertainty. Four primary comparisons are unadjusted for multiplicity. No practical equivalence margin was specified, so an interval crossing zero does not establish equivalence.

|Contrast|Mean delta bpb|95% interval|
|---|---:|---|
|attention:long:original-minus-donor|+0.000236|[-0.002844, +0.003336]|
|attention:long:original-minus-suffix|+0.008448|[+0.005325, +0.011513]|
|attention:mixed:original-minus-donor|+0.001187|[-0.003720, +0.006213]|
|attention:mixed:original-minus-suffix|+0.036621|[+0.031438, +0.041829]|
|signed:long:original-minus-donor|-0.001856|[-0.004279, +0.000722]|
|signed:long:original-minus-suffix|+0.003566|[+0.000435, +0.006678]|
|signed:mixed:original-minus-donor|-0.001365|[-0.004550, +0.001929]|
|signed:mixed:original-minus-suffix|+0.018890|[+0.014718, +0.023044]|
|copy:original-minus-donor|-0.306895|[-0.377445, -0.235807]|
|copy:original-minus-suffix|-0.434693|[-0.505587, -0.368184]|

## Prediction sensitivity and copy reference

KL(original||donor) is computed over every 256-way next-byte distribution, not just the observed target. It measures sensitivity, not whether the change helps.

|Model|Training arm|Mean KL bits/target|Largest token KL bits|
|---|---|---:|---:|
|attention|long|0.023401|3.512628|
|attention|mixed|0.056170|7.054289|
|signed|long|0.015264|6.092083|
|signed|mixed|0.029107|5.820635|

Standalone copy predicts with the longest observed context among orders 4/8/16, empirical follower counts plus total uniform pseudocount 1 across 256 outcomes. It predicts before feeding each target. This is one normalized causal language model, not three separate models or a neural mixture. No mixture weight or memory scope is fitted here. Its original-versus-donor loss interval lies below zero, showing narrow exact-repeat usefulness; it does not prove learned neural retrieval or meaningful semantics.

## Dataset, contamination and reproducibility

HuggingFaceFW/fineweb-edu sample-10BT upstream train split, revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu . Educational English web text filtered from FineWeb/Common Crawl; ODC-BY dataset license and underlying web-text rights caveat. This is evaluation-only despite the upstream split name. Public endpoint, no gated terms accepted. Official HTTPS retrieval verified x-revision and 2MB bound. Recipient response hash ca75465c061b2fb469a17d01d03a2b9ee49e7a2434759231817a263d83146359;474343 bytes. Donor response hash 27e084092b0c8f9323364a5410acfb708248d20029e9cf8597c12386383d2fcc;693207 bytes.

UTF8 byte tokens, no normalization/joining; first 513 recipient bytes and first 256 donor bytes.95 pairs selected: four short recipients excluded and row 1767 excluded by exact scored-target overlap. No score-based exclusions. Selected whole documents/128-byte targets pass exact prior 1700-document, raw WikiText, actual deduplicated 2M training-stream and current-target checks. Shorter/semantic/near duplicates remain unchecked. All 95 donor prefixes are unique and differ from their assigned originals. Pair/window/target hashes and exclusions are recorded. Required prior snapshots are frozen by hash; an unexpected filesystem receipt change fails closed.

Training provenance remains Salesforce/wikitext wikitext-2-raw-v1 revision b08601e04326c79dfdd32d625aee71d232d685c3, actual training hash 77fc488d9c24735f44714b30f6c8507e61c16369d278810cffb0decb5df5f4af. Prior source card has CC-BY-SA3.0/GFDL metadata versus CC-BY-SA4.0 prose; obligations and that discrepancy are preserved. All 12 input checkpoints must match the v7 manifest published before this experiment, commit 607d1237365e6dc4a8df9e42d410f290f13b453c.

36 full-input-length FP64 original/optimized/cache/causality/tail-head checks pass on two documents per configuration/condition, at 256 and 512 inputs. 1,152 full-distribution copy-oracle checks pass. This is substantial targeted testing, not exhaustive numerical validation or a GPU test. Independent review also reconstructs recipient/donor hashes and scoring alignment.

## Inference timing and state

Two-thread CPU torch 2.6, seed 2, batch 4 first 4 eligible pairs, 5 warmups + 31 randomized interleaved measurements per path. Tail128 prefill runs all hidden layers over the input but emits only the 128 scored positions' logits for every condition, avoiding an extra-output-head-work artifact. Tail-head equivalence is tested. Next-token decode follows the whole prefix and excludes cache construction. Shared-CPU timings are probes, not end-to-end service or latency SLOs. Copy-reference timing/all-in system cost was not measured.

|Model/arm|Probe|Original512 ms|Donor512 ms|Suffix256 ms|
|---|---|---:|---:|---:|
|attention/long|tail128_prefill|15.9277|15.6391|7.5691|
|attention/long|next_token_decode|1.0554|1.0668|0.7564|
|attention/mixed|tail128_prefill|18.0183|16.8969|7.6001|
|attention/mixed|next_token_decode|1.0184|1.0809|0.8069|
|signed/long|tail128_prefill|16.3601|15.8537|7.8042|
|signed/long|next_token_decode|1.1751|1.0979|0.8245|
|signed/mixed|tail128_prefill|17.7340|17.2085|7.9366|
|signed/mixed|next_token_decode|1.0286|1.0191|0.8672|

Throughput equivalents use 512 scored positions per tail128 batch or 4 next-byte outputs per decode batch. They are not commercial billable tokens:

- attention/long/original512: 32145.2 scored positions/s; 3789.9 next bytes/s.
- attention/long/suffix256: 67643.9 scored positions/s; 5288.2 next bytes/s.
- attention/mixed/original512: 28415.6 scored positions/s; 3927.7 next bytes/s.
- attention/mixed/suffix256: 67367.6 scored positions/s; 4957.5 next bytes/s.
- signed/long/original512: 31295.6 scored positions/s; 3404.1 next bytes/s.
- signed/long/suffix256: 65605.6 scored positions/s; 4851.5 next bytes/s.
- signed/mixed/original512: 28871.1 scored positions/s; 3888.6 next bytes/s.
- signed/mixed/suffix256: 64511.4 scored positions/s; 4612.4 next bytes/s.

Measured batch 4 persistent cache tensor bytes (same for both training arms):

- attention/original512: 2,097,152 logical cache tensor bytes, 4,194,304 distinct underlying storage bytes retained by cache tensors, 527,616 parameter bytes, 524,288 warm positional-table bytes.
- attention/suffix256: 1,048,576 logical cache tensor bytes, 2,097,152 distinct underlying storage bytes retained by cache tensors, 527,616 parameter bytes, 524,288 warm positional-table bytes.
- signed/original512: 2,099,200 logical cache tensor bytes, 4,196,352 distinct underlying storage bytes retained by cache tensors, 527,616 parameter bytes, 524,288 warm positional-table bytes.
- signed/suffix256: 1,050,624 logical cache tensor bytes, 2,099,200 distinct underlying storage bytes retained by cache tensors, 527,616 parameter bytes, 524,288 warm positional-table bytes.

Logical tensor payload can undercount retained storage because V is a view into a larger QKV allocation. The additional storage figure deduplicates underlying cache storages and includes those retained allocations. Neither figure measures total process RSS, peak activations, allocator/Python/framework overhead or service concurrency. Prefix shortening saves state by removing information; it is not an exact-model-equivalence optimization.

## Compute and retained economic gates

No training targets added. Neural evaluation scores437,760 target predictions; copy scores36,480. Recorded neural evaluator wall time12.42s includes data preparation but excludes the subsequent copy reference, external retrieval, tests, timing and reporting. These are not an inclusive bill. Historical training/search costs remain counted in v7, not silently amortized away. No new paid resources, production changes or external-agent messages.

Rates remain$0.05/M uncached input,$0.001/M cached input,$0.10/M billable output, with 30% margin on net revenue. Conditional output-only hardware floor 3968.254 billable output tokens/s per $1/hour at 100% productive utilization excludes all other costs and is not a mixed-workload gate. See economics.md for separate input/output revenue and prefill/decode/cache, taxes/fees, memory, utilization, bursts/reliability, non-billable reasoning/tools, training amortization, storage/bandwidth and operations. No verified hardware rate, byte-to-billable-token mapping, full-workload SLA or frontier parity allows actual cost per useful token or commercial gap scoring for these models.≤$500/month subscription retains the full workload and 30% net margin. This experiment grants no economic pass.

## Reproduce

Python3.12, torch 2.6.0 CPU, numpy; pyarrow for WikiText download. download_wikitext.py restores pinned WikiText; set ANTI_DATA_DIR. download_data.py restores/hash-checks fresh recipient plus17 prior/donor snapshots in recorded sibling directories, maximum 36MB if all missing. Source revision/hash changes fail closed. Restore v7 checkpoints by its published instructions and set ANTI_V7_CHECKPOINT_DIR; ANTI_DONOR_ROWS optionally points to the exact donor snapshot. expected_checkpoint_hashes.json must match.

Run prefix_test.py with external 300-second cap, summarize_v9.py, profile_v9.py, validate_v9.py, measure_state.py, report_v9.py. External 120-second caps used for profiling/validation. Publish only text code, numeric results, provenance and tests; raw corpora and binary weights excluded. The next research gate should demonstrate useful earlier-content processing on a prespecified task/cohort, rather than treating context-length loss recovery as sufficient evidence.
