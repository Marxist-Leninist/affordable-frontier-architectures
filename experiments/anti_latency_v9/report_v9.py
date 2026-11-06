from pathlib import Path
import json
P=Path(__file__).resolve().parent;s=json.loads((P/'summary.json').read_text());d=json.loads((P/'data_manifest.json').read_text());r=json.loads((P/'results.json').read_text());copy=json.loads((P/'copy_results.json').read_text());prof=json.loads((P/'profile.json').read_text());memory=json.loads((P/'state_memory.json').read_text())
out='''# Anti-attention v9: does earlier prefix content help?

## Useful result

On95 fresh recipient/donor document pairs, replacing the first256 bytes with unrelated real text changes neural predictions but yields no clear aggregate genuine-prefix advantage for any of the four context-trained neural configurations. All four primary95% document-pair bootstrap intervals include zero. This is limited evidence, not proof that these models ignore earlier text or that its true effect is zero.

The fixed exact-copy reference does benefit from the original prefix by0.306895 bits/byte [0.235807,0.377445], showing that exploitable repeated text exists in this cohort. Its total loss is much worse than the neural models, so it is a diagnostic reference, not a competitive replacement or evidence of frontier reasoning.

Removing the earlier prefix improves mean neural loss in every configuration on this cohort and reduces measured prefill work. That is a small-model, cohort-specific observation, not permission to prune useful user context or a demonstrated general equal-quality optimization. The training-context gains in v5–v8 do not alone establish useful distant-content processing.

## Frozen intervention and quality

New FineWeb-Edu recipient rows1700–1799. Recipient row1700+i receives the first256 bytes from donor row1600+i, without reuse. Replace only bytes0–255; retain recipient bytes256–512, input length, and target positions exactly. Original/donor512 and genuine suffix256 all score recipient bytes385–512 (128 targets). At every scored prediction,129–256 immediately preceding recipient bytes remain unchanged between original and donor. The donor introduces an artificial incoherent boundary; the test does not isolate semantic retrieval or generalize across all donor choices.

Frozen v7 attention and signed U−A models, retained-Adam long/mixed training arms, three seeds. Same architecture parameter count131,904 and historical target/update budgets; not equal FLOPs or converged models. No new training, tuning or loss-based selection. This extends earlier donor diagnostics to the newly context-trained signed models with matched Transformer, a disjoint fresh cohort, full-distribution sensitivity, and exact-copy reference.

Mean bits/byte over three seeds, lower is better:

|Model|Training arm|Original512|Donor512|Suffix256|
|---|---|---:|---:|---:|
'''
for kind in ['attention','signed']:
 for arm in ['long','mixed']:
  vals=[next(x['bpb'] for x in s['means'] if x['kind']==kind and x['arm']==arm and x['condition']==c) for c in ['original512','donor512','suffix256']];out+=f"|{kind}|{arm}|{vals[0]:.6f}|{vals[1]:.6f}|{vals[2]:.6f}|\n"
vals={x['condition']:x['bpb'] for x in copy['rows']};out+=f"|Exact copy|No neural training|{vals['original512']:.6f}|{vals['donor512']:.6f}|{vals['suffix256']:.6f}|\n"
out+='''
Suffix truncation changes available history and, for signed models, the population contributing to the cumulative value mean. Pure RoPE is shift-invariant; merely reindexing a suffix is not an absolute-position confound. Original/donor is the length-preserving content intervention; original/suffix is a separate operational comparison.

## Conditional paired uncertainty

Negative original-minus-donor delta favors genuine earlier text. Average the three model seeds within each pair first, then5000 resamples of95 disjoint recipient/donor pairs, fixed seed9009. All10 primary/secondary contrasts are retained. Intervals are conditional on this deterministic cohort, fixed donors and checkpoints; they do not include training-run uncertainty. Four primary comparisons are unadjusted for multiplicity. No practical equivalence margin was specified, so an interval crossing zero does not establish equivalence.

|Contrast|Mean delta bpb|95% interval|
|---|---:|---|
'''
for x in s['contrasts']:out+=f"|{x['label']}|{x['mean_delta_bpb']:+.6f}|[{x['ci95'][0]:+.6f}, {x['ci95'][1]:+.6f}]|\n"
out+='''
## Prediction sensitivity and copy reference

KL(original||donor) is computed over every256-way next-byte distribution, not just the observed target. It measures sensitivity, not whether the change helps.

|Model|Training arm|Mean KL bits/target|Largest token KL bits|
|---|---|---:|---:|
'''
for x in s['kl']:out+=f"|{x['kind']}|{x['arm']}|{x['mean_kl_bits']:.6f}|{x['largest_token_kl_bits']:.6f}|\n"
out+='''
Standalone copy predicts with the longest observed context among orders4/8/16, empirical follower counts plus total uniform pseudocount1 across256 outcomes. It predicts before feeding each target. This is one normalized causal language model, not three separate models or a neural mixture. No mixture weight or memory scope is fitted here. Its original-versus-donor loss interval lies below zero, showing narrow exact-repeat usefulness; it does not prove learned neural retrieval or meaningful semantics.

## Dataset, contamination and reproducibility

HuggingFaceFW/fineweb-edu sample-10BT upstream train split, revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu . Educational English web text filtered from FineWeb/Common Crawl; ODC-BY dataset license and underlying web-text rights caveat. This is evaluation-only despite the upstream split name. Public endpoint, no gated terms accepted. Official HTTPS retrieval verified x-revision and2MB bound. Recipient response hashca75465c061b2fb469a17d01d03a2b9ee49e7a2434759231817a263d83146359;474343 bytes. Donor response hash27e084092b0c8f9323364a5410acfb708248d20029e9cf8597c12386383d2fcc;693207 bytes.

UTF8 byte tokens, no normalization/joining; first513 recipient bytes and first256 donor bytes.95 pairs selected: four short recipients excluded and row1767 excluded by exact scored-target overlap. No score-based exclusions. Selected whole documents/128-byte targets pass exact prior1700-document, raw WikiText, actual deduplicated2M training-stream and current-target checks. Shorter/semantic/near duplicates remain unchecked. All95 donor prefixes are unique and differ from their assigned originals. Pair/window/target hashes and exclusions are recorded. Required prior snapshots are frozen by hash; an unexpected filesystem receipt change fails closed.

Training provenance remains Salesforce/wikitext wikitext-2-raw-v1 revisionb08601e04326c79dfdd32d625aee71d232d685c3, actual training hash77fc488d9c24735f44714b30f6c8507e61c16369d278810cffb0decb5df5f4af. Prior source card has CC-BY-SA3.0/GFDL metadata versus CC-BY-SA4.0 prose; obligations and that discrepancy are preserved. All12 input checkpoints must match the v7 manifest published before this experiment, commit607d1237365e6dc4a8df9e42d410f290f13b453c.

36 full-input-length FP64 original/optimized/cache/causality/tail-head checks pass on two documents per configuration/condition, at256 and512 inputs.1,152 full-distribution copy-oracle checks pass. This is substantial targeted testing, not exhaustive numerical validation or a GPU test. Independent review also reconstructs recipient/donor hashes and scoring alignment.

## Inference timing and state

Two-thread CPU torch2.6, seed2, batch4 first4 eligible pairs,5 warmups+31 randomized interleaved measurements per path. Tail128 prefill runs all hidden layers over the input but emits only the128 scored positions' logits for every condition, avoiding an extra-output-head-work artifact. Tail-head equivalence is tested. Next-token decode follows the whole prefix and excludes cache construction. Shared-CPU timings are probes, not end-to-end service or latency SLOs. Copy-reference timing/all-in system cost was not measured.

|Model/arm|Probe|Original512 ms|Donor512 ms|Suffix256 ms|
|---|---|---:|---:|---:|
'''
for kind in ['attention','signed']:
 for arm in ['long','mixed']:
  for task in ['tail128_prefill','next_token_decode']:
   vals=[prof['median_seconds'][task+':'+kind+'_'+arm+':'+c]*1000 for c in ['original512','donor512','suffix256']];out+=f"|{kind}/{arm}|{task}|{vals[0]:.4f}|{vals[1]:.4f}|{vals[2]:.4f}|\n"
out+='\nThroughput equivalents use512 scored positions per tail128 batch or4 next-byte outputs per decode batch. They are not commercial billable tokens:\n\n'
for kind in ['attention','signed']:
 for arm in ['long','mixed']:
  for c in ['original512','suffix256']:
   t=prof['median_seconds']['tail128_prefill:'+kind+'_'+arm+':'+c];dt=prof['median_seconds']['next_token_decode:'+kind+'_'+arm+':'+c];out+=f'- {kind}/{arm}/{c}: {512/t:.1f} scored positions/s; {4/dt:.1f} next bytes/s.\n'
out+='\nMeasured batch4 persistent cache tensor bytes (same for both training arms):\n\n'
for kind in ['attention','signed']:
 for c in ['original512','suffix256']:
  a=next(x for x in memory if x['kind']==kind and x['arm']=='long' and x['condition']==c);out+=f"- {kind}/{c}: {a['persistent_cache_tensor_bytes']:,} logical cache tensor bytes, {a['distinct_retained_cache_storage_bytes']:,} distinct underlying storage bytes retained by cache tensors, {a['model_parameter_bytes']:,} parameter bytes, {a['warm_position_table_bytes']:,} warm positional-table bytes.\n"
out+='''
Logical tensor payload can undercount retained storage because V is a view into a larger QKV allocation. The additional storage figure deduplicates underlying cache storages and includes those retained allocations. Neither figure measures total process RSS, peak activations, allocator/Python/framework overhead or service concurrency. Prefix shortening saves state by removing information; it is not an exact-model-equivalence optimization.

## Compute and retained economic gates

'''
out+=f"No training targets added. Neural evaluation scores{36*d['targets_per_model_condition']:,} target predictions; copy scores{3*d['targets_per_model_condition']:,}. Recorded neural evaluator wall time{r['elapsed_wall_seconds']:.2f}s includes data preparation but excludes the subsequent copy reference, external retrieval, tests, timing and reporting. These are not an inclusive bill. Historical training/search costs remain counted in v7, not silently amortized away. No new paid resources, production changes or external-agent messages.\n"
out+='''
Rates remain$0.05/M uncached input,$0.001/M cached input,$0.10/M billable output, with30% margin on net revenue. Conditional output-only hardware floor3968.254 billable output tokens/s per$1/hour at100% productive utilization excludes all other costs and is not a mixed-workload gate. See economics.md for separate input/output revenue and prefill/decode/cache, taxes/fees, memory, utilization, bursts/reliability, non-billable reasoning/tools, training amortization, storage/bandwidth and operations. No verified hardware rate, byte-to-billable-token mapping, full-workload SLA or frontier parity allows actual cost per useful token or commercial gap scoring for these models.≤$500/month subscription retains the full workload and30% net margin. This experiment grants no economic pass.

## Reproduce

Python3.12, torch2.6.0 CPU, numpy; pyarrow for WikiText download. download_wikitext.py restores pinned WikiText; set ANTI_DATA_DIR. download_data.py restores/hash-checks fresh recipient plus17 prior/donor snapshots in recorded sibling directories, maximum36MB if all missing. Source revision/hash changes fail closed. Restore v7 checkpoints by its published instructions and set ANTI_V7_CHECKPOINT_DIR; ANTI_DONOR_ROWS optionally points to the exact donor snapshot. expected_checkpoint_hashes.json must match.

Run prefix_test.py with external300-second cap, summarize_v9.py, profile_v9.py, validate_v9.py, measure_state.py, report_v9.py. External120-second caps used for profiling/validation. Publish only text code, numeric results, provenance and tests; raw corpora and binary weights excluded. The next research gate should demonstrate useful earlier-content processing on a prespecified task/cohort, rather than treating context-length loss recovery as sufficient evidence.
'''
for old,new in [('On95','On 95'),('first256','first 256'),('first513','first 513'),('bytes0','bytes 0'),('bytes256','bytes 256'),('bytes385','bytes 385'),('all10','all 10'),('then5000','then 5000'),('of95','of 95'),('seed9009','seed 9009'),('counts131','counts 131'),('count131','count 131'),('four primary95','four primary 95'),('primary95','primary 95'),('orders4','orders 4'),('pseudocount1','pseudocount 1'),('across256','across 256'),('every256','every 256'),('rate3968','rate 3968'),('floor3968','floor 3968'),('with30','with 30'),('and30','and 30'),('17 prior','17 prior'),('maximum36','maximum 36'),('external300','external 300'),('External120','External 120'),('batch4','batch 4'),('first4','first 4'),('pairs,5','pairs, 5'),('warmups+31','warmups + 31'),('outputs per','outputs per'),('uses512','uses 512'),('use512','use 512'),('or4','or 4'),('All95','All 95'),('All12','All 12'),('95 pairs','95 pairs'),('row1767','row 1767'),('prior1700','prior 1700'),('actual deduplicated2M','actual deduplicated 2M'),('and512','and 512'),('at256','at 256'),('36 full-input','36 full-input'),('source:','source:')]:out=out.replace(old,new)
import re
for word in ['by','first','last','all','All','On','over','Over','only','total','among orders','orders','across','after','before','at','with','and','per','than','every','mean','seed','step','rows','row','batch','maximum','external','External','includes','counts','count','hash','revision','commit','length','input','targets','Byte','copy','model','torch']:
 out=re.sub(r'\b'+re.escape(word)+r'(?=[0-9$])',word+' ',out)
out=re.sub(r'(?<=\d)(?=bits|bytes|targets|pairs|documents|predictions|models|seconds|bpb|ms|s measured)', ' ',out)
out=out.replace('the128','the 128').replace('inputs.1,152','inputs. 1,152')
out=re.sub(r'\b(scores|hash|revision)(?=[0-9a-f]{4})',r'\1 ',out)
(P/'README.md').write_text(out)
