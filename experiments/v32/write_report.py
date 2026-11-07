"""Post-measurement report; no scientific selection or scoring."""
import json
from pathlib import Path
P=Path(__file__).parent
s=json.loads((P/'summary.json').read_text())
t=json.loads((P/'timing_summary.json').read_text())
a=json.loads((P/'selection.json').read_text())
r=json.loads((P/'results.json').read_text())
timing=json.loads((P/'timing.json').read_text())
quality='\n'.join('|'+arm+'|'+'|'.join(f'{s["models"][arm][v]:.6f}' for v in ('neural','historical','constant','by_order'))+'|' for arm in ('local','gru'))
latency='\n'.join('|'+key+'|'+f'{v["total_ms_p50_p95"][0]:.3f} / {v["total_ms_p50_p95"][1]:.3f}'+'|'+f'{v["index_bytes_median"]:.0f}'+'|'+str(v['state_bytes'])+'|' for key,v in t['groups'].items())
text=f'''# v32: match-length calibration helps one control, not a universal winner

Bounded isolated CPU research, 2026-11-07. No new neural training, paid resources or production changes. Text code, numeric results and provenance are retained privately; source text and weights are excluded.

## Main result

Choosing copy-memory strength from the longest **already observed** matching context improves the local Transformer against a constant freshly calibrated on the identical WikiText validation ledger: **2.715855 → 2.682434 bits/byte**, delta **−0.033420**, paired 95% interval **[−0.054759, −0.018114]**. All three seeds agree in direction.

The stronger historical constant remains an important control: local loss is **2.689486**, and the adaptive gate's **−0.007051** difference has interval **[−0.018988, +0.001045]**. A gain over that previous configuration is not established. GRU improves **2.652418 → 2.645736**, but its **−0.006683** interval **[−0.022546, +0.005627]** also spans zero. Failure to establish an improvement does not prove equivalence.

This is a calibration/domain-transfer result, not evidence that retrieval gates generally beat constant mixtures, that this is a new general architecture, or that the cost target is met. The fair retuned constant minimizes its validation grid but transfers worse than the historical local constant. Keep historical controls rather than interpreting one newly retuned baseline as the strongest possible comparator.

## Identical held-out targets

Mean over three frozen neural seeds; lower bits/byte is better.

|Neural model|Alone|Historical constant|Retuned constant|Match-order gate|
|---|---:|---:|---:|---:|
{quality}

Standalone copy loss is **{r['copy_only_bpb']:.6f} bits/byte**. It complements the neural predictor; it is not a competitive replacement. Local and GRU remain different-quality models, not an equal-quality architecture speed comparison.

FineWeb contributes **43 windows, 34 documents and 5,504 scored next-byte targets**. Every memory/gate variant receives identical checkpoints, neural input, complete copy history and targets. A 4,096-input window scores its final 128 targets, with 3,969–4,096 observed bytes available. No targets or future text enter cache counts before prediction.

## Frozen method and calibration

The inherited full-history exact-copy index stores observed context/follower counts for orders 4, 8 and 16. Query the longest matching context, add one total uniform pseudocount over 256 bytes, and normalize. Mix its full distribution with the neural distribution. With no match, use neural alone. The gate depends only on match length, not the next target, probability of that target, future text or a loss-selected test subgroup.

Same calibration ledger for both new methods: first 32 eligible nonoverlapping 4,096-input WikiText validation windows, final 128 targets, 4,096 distinct target positions, pooled equally over seeds 1–3. Constant chooses one weight per neural architecture on 0, .1, ..., 1; match-order gate chooses a separate weight for each observed order using the same grid and observations. This introduces three parameters versus one and more calibration flexibility, not an equal-capacity estimator. Its separable bucket objective is exactly equivalent to exhaustive joint optimization over this grid. Lower weight breaks ties; missing buckets would use the selected constant.

- Local: constant .2; order 4 / 8 / 16 weights **.2 / .4 / .5**.
- GRU: constant .2; order 4 / 8 / 16 weights **.1 / .4 / .4**.
- Historical weights retained without tuning: local .3 and GRU .2.
- Validation buckets: no-match 1,899; order4 1,597; order8 531; order16 only **69**. The high-order gate has little calibration data.
- Held-out buckets: no-match 2,328; order4 2,167; order8 785; order16 224.

Protocol/source/data/checkpoint hashes were recorded before calibration and scoring; selection was saved before FineWeb scoring. Those hashes support integrity and reviewed control flow, not an externally timestamped preregistration. No new neural fitting or evaluation-based parameter changes were made. WikiText validation has also supported earlier model selection, so this is not untouched validation. The fresh FineWeb cohort was not used for selection.

This changes model interpolation rather than repeating v28–v30 index representation optimizations. Cache interpolation has substantial prior art, including [continuous neural caches](https://arxiv.org/abs/1612.04426) and [information-weighted interpolation](https://arxiv.org/abs/1809.08826). This experiment uses a simpler exact-byte match-length gate, not a replication of those algorithms or a claimed invention.

## Baseline, training history and fair-comparison limits

Frozen v19 local span128 Transformer and gated residual GRU, seeds 1–3, about 132k dense parameters each. Modern Transformer components are two blocks, width64, four attention heads, RMSNorm, RoPE and SwiGLU. Local weights occupy 527,616 FP32 bytes (131,904 parameters); GRU 528,640 bytes (132,160 parameters). All parameters are active, no MoE or quantization. Neural training is inherited from v14 step4,000 plus v19's 500 updates/512,000 additional target presentations per run. The same adaptation ledger was used across architectures, but pretraining/search histories and compute differ. There is no matched-FLOPs or converged frontier-baseline claim.

The Transformer has a 255-input-byte receptive field from its two span128 layers. Its tested pruned prefill preserves absolute RoPE positions and is applied equally to every variant. GRU receives all 4,096 inputs. Full-history copy memory supplies exact repetitions beyond the local neural horizon, but this is not a test of distant semantic reasoning, robust retrieval or AGI. The within-checkpoint gate comparison is controlled; a strong pretrained full-context Transformer and real downstream tasks remain necessary for frontier claims.

## Complete-path timing and storage

|Path|Total p50 / p95 ms|Final copy-index bytes|Neural state bytes|
|---|---:|---:|---:|
{latency}

Median paired constant/gate total-latency ratios (>1 favors the gate): local **{t['paired']['local']['median_constant_over_by_order_total_ratio']:.4f}**, GRU **{t['paired']['gru']['median_constant_over_by_order_total_ratio']:.4f}**. Gate is faster in {t['paired']['local']['by_order_faster_pairs']}/72 local and {t['paired']['gru']['by_order_faster_pairs']}/72 GRU pairs. These shared-host probes do not establish a robust speed benefit, latency equivalence, or reduced service cost. Raw phase timings are retained.

One CPU thread, batch1, PyTorch 2.6.0+cpu, FP32 neural computation and float64 probability normalization/mixing. First eight eligible evaluation documents, three seeds, one warmup per variant and three randomized interleaved repetitions per document/seed/architecture (seed320032). Each probe includes construction of full copy memory from 4,096 bytes, neural prefill and 32 teacher-forced observed-byte updates, yielding 33 complete 256-way distributions. There are 432 measured rows, not independent customer requests. No task-owned correctness audit overlapped timing; shared host load is uncontrolled.

Index sizes are reachable CPython object allocation, not peak RSS/VRAM. Weights are additional to the table. Gate parameter dictionary/Python metadata, framework, allocator and transients are not included in index/state columns; three scalar weights are not meaningful memory savings. Copy history remains full-window and grows with context; no eviction/memory cap gain is claimed. No sampled generation, commercial tokenizer, GPU batching, arrival bursts or network latency was measured.

Calibration used **{a['cpu_seconds']:.3f} CPU seconds**, evaluation **{r['cpu_seconds']:.3f}**, final timing **{timing['cpu_seconds']:.3f}**. Preparation, startup, independent audit, tests and historical training/search are additional. These are incremental measured compute times, not an inclusive lifecycle bill or billed cloud expenses.

## Data provenance and rights

Evaluation: public **HuggingFaceFW/fineweb-edu**, **sample-10BT**, upstream **train** split used only as held-out evaluation here, revision **87f09149ef4734204d70ed1d046ddc9ca3f2b8f9**, viewer rows **1800–1899**. First 8,193 UTF8 bytes per document, byte vocabulary 0–255, no normalization or document concatenation. Total capped bytes 349,795. Raw response 609,761 bytes, SHA256 **3e576f275f934136822ea6b368babe911da9f57f19a757830b314d06a9b7b53d**. HTTPS viewer x-revision checked fail-closed; response, capped/full document, window and target hashes are retained. This is a deterministic minimum-length convenience sample, not a representative random web sample.

Training/calibration: **Salesforce/wikitext**, **wikitext-2-raw-v1**, revision **b08601e04326c79dfdd32d625aee71d232d685c3**, original LF-joined UTF8 rows. Raw train 36,718 rows/10,951,562 bytes, SHA256 **c58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8**; validation 1,148,007 bytes, SHA256 **714a24a5de7459876f7cfdc64837e551dd7cde2a51f95b763ee083fff273e307**. Exact training-target exclusion retained all first32 calibration windows.

No whole-document duplicates or complete128-byte target matches were found against training, validation, 1,800 unique prior evaluation documents or other current documents; no current window exclusions were required. Shorter overlaps, paraphrases, shared websites and near/semantic duplicates remain unmeasured. Full source text is never published.

Fresh pinned-card review preserves FineWeb-Edu **ODC-By1.0**, Common Crawl terms and underlying source rights. WikiText's metadata **CC-BY-SA3.0/GFDL** conflicts with **CC-BY-SA4.0** licensing prose; the discrepancy is preserved rather than silently resolved. Dataset card access is not legal certification or blanket commercial rights. See [current source review](pricing_sources.md).

## Tests and independent review

Four mixture/copy unit tests and six inherited economics tests pass. Independent review reconstructs **9,600 complete copy distributions** across validation/evaluation, calibration grid optima, all saved mixture losses and primary document-bootstrap summaries. All six checkpoint hashes verify. Bounded neural full/chunked/pruned/causal checks pass; maximum logit discrepancy **1.63e-5**, and future changes do not affect earlier predictions. These are targeted checks, not exhaustive correctness or frontier evaluation. See [independent audit](independent_audit.md) and its raw JSON/source.

Average the three seed losses first, then 5,000 paired document-cluster resamples, seed32, target-weighted ratio of sums. All windows from the same document move together. The two primary contrasts are both retained, unadjusted for multiple comparisons. Intervals are conditional on this small convenience cohort and fixed trained models, not independent model-training uncertainty. Full per-document sufficient statistics and all seed deltas are in summary.json. Secondary subgroup/historical comparisons are descriptive, not a search for a test-tuned winner.

## Economics and decision

[Primary prices](pricing_sources.md) remain numerically unchanged versus the November6 capture. Targets stay **$0.05/M uncached input, $0.001/M cached input, $0.10/M billable output**, and all-in cost must be **<=0.70×net revenue** for a 30% margin, not markup. Muse Spark Contributor remains .10/.002/.20 and requires training-use permission; no user content was submitted. Runpod Secure H100 PCIe/SXM list2.89/3.99 dollars/hour, not verified deployment offers.

Conditional output-only floor remains **3,968.254 billable output tokens/s per $1/hour**, at100% productive utilization and zero other costs; at H dollars/hour and productive utilization u it is3,968.254H/u. Mixed revenue/workload can change the appropriate gate and must be calculated independently. Retained10M uncached+5M cached+2M visible+2M hidden ledger permits **$0.478695** all-in cost when hidden is not billed, **$0.614495** when billed, after3% fees and30% margin. Hidden compute is always charged.

The [complete inherited economic ledger](economics/README.md) retains **<=$500/month**, the unchanged typical/average/heavy full-month workload, context/sessions, cache sensitivity, hidden/tool work, pooled versus dedicated capacity, subscriber counts, concurrency and 4x bursts. GBP200/GBP20 remain separate stretch tiers. Fees/tax are excluded from net revenue once. Prefill/decode, state/cache, memory/bandwidth/storage, idle/latency/reliability capacity, training amortization and support all remain costs. Six accounting tests were rerun; unchanged analytical scenarios are not new measured results.

**Every tested candidate remains unscored against the commercial gate, not a pass:** no actual all-in billable-token costs, commercial token mapping, frontier-quality parity, GPU sustained throughput or latency-constrained capacity were established. A small predictive improvement cannot justify a cheaper equivalent-capability product. The next useful step is independent fresh-cohort confirmation with better-supported calibration, not deployment or a declared universal gate advantage.

## Reproduce

Use Python3.12, requirements.txt and official CPU torch2.6.0 as appropriate. This directory was executed as frontier_research_v32 alongside prepared frontier_research_v8 (pinned train/validation files) and frontier_research_v19/checkpoints (six hash-matching adapted weights). In repository layout create equivalent sibling aliases rather than changing frozen paths. Reconstruct v19 through the [preserved v19–v22 archive](../../archives/20261105_v19_v22/README.md), with its v14/v8 dependencies. Weights and corpus are intentionally not bundled, and reproducing them is additional work; exact float results may vary by software/hardware.

Run prepare_data.py to retrieve and hash-check the bounded fresh snapshot. Retained items.json/provenance.json specify the scored windows and historical overlap checks. For a full provenance rerun restore the prior snapshots identified by provenance.json; prepare_ledger.py is a pre-freeze construction helper and must not overwrite published protocol inputs. It is not needed for scoring the published ledger.

Run python -m unittest discover -s . -v; python run.py --select; python run.py --evaluate; python summarize.py; python independent_audit.py; then python timing.py and python summarize.py after other heavy work ends. Preserve published selection/results separately before reproduction because these commands write their output files. Never refreeze historical source hashes to hide drift; --freeze is only for a new experiment in a clean directory. Independent review has additional historical-source dependencies disclosed in its source. Pure numeric loss arrays are regenerable and excluded alongside corpus/weights; summary sufficient statistics, raw timing, source/protocol/hashes and audits remain available.
'''
(P/'README.md').write_text(text)
