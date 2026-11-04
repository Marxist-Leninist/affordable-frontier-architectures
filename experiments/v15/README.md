# v15: frozen-model transfer and context extrapolation

2026-11-04. New second-corpus evaluation using real Hugging Face FineWeb-Edu data, with frozen v14 models. No training on the new corpus, paid compute, production changes, or frontier/AGI claim.

## New findings

The statistical model's WikiText advantage **does not transfer to this FineWeb-Edu sample**. At the original 128-byte scoring length:

| Frozen model | FineWeb-Edu evaluation bits/byte | Initialization sample SD | Prior WikiText test bits/byte |
|---|---:|---:|---:|
| Modern Transformer | 3.076653 | .010586 | 2.359696 |
| Gated GRU | 3.002345 | .023075 | 2.238784 |
| Statistical five-gram | 3.433956 | deterministic | 2.226092 |

Lower is better. The statistical model was narrowly best on WikiText but is worst here; the two neural models preserve more useful predictive information in this transfer test. This is an ordered 100-document convenience sample, not population-level evidence or a task-intelligence benchmark. Cross-corpus loss differences also reflect different text distributions and serialization/reset protocols.

A second finding is more severe: **giving these 128-byte-trained Transformer checkpoints 512 bytes of context sharply worsens prediction on identical targets**.

| Model | Paired tail loss with 128 input bytes | Same targets with 512 input bytes | Long minus short |
|---|---:|---:|---:|
| Transformer | 3.075228 | 5.614077 | +2.538848 |
| GRU | 3.000528 | 2.977672 | −.022856 |
| Statistical five-gram | 3.445420 | 3.433760 | −.011660 |

Every paired window worsens for each Transformer seed. Independent fresh scoring, full-versus-cached inference, causal perturbations, and an independently computed attention/RoPE reference agree: no target-alignment, mask or caching error was found that explains away this failure. It is a measured failure of **these short-context-trained checkpoints**, not a claim about Transformers generally or proof that recurrence scales better.

The statistical gain uses no distant memory: its long condition supplies the missing local suffix for only the first three scored targets; all later probabilities match. GRU's small gain is predictive loss, not reasoning or retrieval accuracy. Headline deltas weight windows equally. The separate document-bootstrap intervals in the results weight documents equally and are descriptive convenience-sample resampling, not population confidence intervals or confidence intervals for the window-weighted headline.

### Post-hoc control, explicitly exploratory

The large Transformer failure triggered one additional frozen-model diagnostic on the first 100 contiguous 512-target WikiText test windows, scoring the same final 128 targets. Transformer mean loss worsens **2.333637 → 5.285505**, while GRU improves **2.207163 → 2.176278**. Thus the failure also appears on the original source corpus's official test split, rather than only under FineWeb domain shift. This control was not in the initial protocol; it made no changes to models, choices, gates, or reported primary scores. These flat WikiText windows may cross article boundaries, unlike the per-document FineWeb evaluation. See `wikitext_diagnostic.py/json`.

## Data and frozen protocol

Source: [HuggingFaceFW/FineWeb-Edu pinned dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md), revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, subset `sample-10BT`, split `train`. Attribution: HuggingFaceFW/FineWeb-Edu. Card license [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/), also subject to [Common Crawl terms](https://commoncrawl.org/terms-of-use); underlying web text may retain third-party rights. Corpus text and copied dataset card are not redistributed.

Fetch exactly viewer rows 0–99. Response has `partial=false`, no truncated cells, and 370,599 original UTF-8 text bytes. Keep at most the first 4,097 bytes per document, without normalization or joining documents: 252,205 retained bytes. Byte caps may end inside a Unicode codepoint; the target is explicitly byte prediction, not re-decoded text. Every document retains independent reset boundaries.

- Primary: 1,935 complete 128-target windows, **247,680 targets**, all 100 documents. First source byte and incomplete trailing targets in each document are excluded consistently.
- Paired context: 456 complete 512-target windows from 97 documents. Score only the last 128 targets of each: **58,368 identical targets** in each condition. Short input is the corresponding fresh 128-byte suffix; long input is all 512 bytes. IDs refer to the parent 512-target window.
- Exact-overlap diagnostic: none of the 1,935 complete 129-byte primary source windows occurs verbatim in WikiText training text. This does not exclude shorter/near duplicates or all historical contamination. No data was removed after this check.
- Source response and every capped document have SHA256 records. The observed viewer `x-revision` is recorded. The viewer URL itself moves: reproduction fails closed if revision or hashes change. If it no longer serves matching rows, the retained local snapshot or reconstruction from the pinned source dataset is required; the downloader does not silently replace the sample.

Use all three v14 final 4,000-step seeds for each family. The stronger-trained baseline is the existing two-block 131,904-parameter RoPE/RMSNorm/SwiGLU Transformer; recurrence has 132,160 parameters. Verify checkpoint/model hashes, and rebuild statistical counts from the exact v14 4,096,000-target ledger. No new LR, mixture, checkpoint, seed, threshold or order is chosen on FineWeb. No gradient/count adaptation occurs. This is a stronger baseline than v12, **not a converged large Transformer or frontier reference**.

The prespecified descriptive quality screen is absolute aggregate loss difference <=.02 bpb for every corresponding seed. **No model pair passes** on the primary transfer task. Therefore latency numbers below are not an equal-quality speedup. The threshold is not a statistical equivalence test, and no model is selected for deployment.

## Measured latency, memory and costs

An independent test lasting .082 seconds may have overlapped the original run's timing. Those original values remain in `results.json` for transparency, but headline timings use the separate `retime.py` rerun after the audits and diagnostic. It uses the same eight eligible document IDs (0–7) at both prefix lengths; no quality scores change.

Single-thread CPU float32, batch 1. Each probe computes one full next-byte distribution after prefill plus 32 teacher-forced cached updates: **33 distributions**, not exactly 32 sampled generated bytes. Eight documents, five repeats each, discard two warmups: 24 retained probes. Neural numbers below are medians across seed-specific p50 values.

| Model | End-to-end probe, 128-byte prefix | 512-byte prefix | 512-prefix prefill p50 | 512-prefix cached-update p50 |
|---|---:|---:|---:|---:|
| Transformer | 24.237 ms | 34.914 ms | 7.813 ms | .824 ms |
| GRU | 16.007 ms | 28.631 ms | 17.460 ms | .350 ms |
| Statistical | 13.971 ms | 12.919 ms | .374 ms | .389 ms |

Small statistical timing differences between lengths are noise/workload details, not evidence that longer context is faster: this model only consumes its four-byte suffix. Decode p50 summarizes sequence-average update latency; local p95 is across 24 probes, not service or individual inter-token p95. Fixed model/order timing, host scheduling, and a small sample remain limitations. Sampling, queues, network, tokenizer, request scheduling, retries, tools and service overhead are absent. Prefill computes all neural position logits, not an optimized last-position-only head. Timing reaches 160/544 observed bytes; it is not the exact scoring protocol or a long-context retrieval test.

At 544 observed bytes, Transformer cache tensors occupy **557,056 bytes** versus GRU **512 bytes**. Statistical suffix is four logical bytes, but the count table does not disappear: v14 measured 54,023,171 bytes in Python objects versus 4,506,831 serialized, excluding interpreter/allocator/transient overhead. These inherited table footprints were not remeasured here. Neural weights and runtime memory also remain.

Primary experiment wall time **57.35 seconds**, including an 11.73-second statistical rebuild, excluding development, post-hoc control, retiming and audits. No incremental paid compute was purchased; CPU dollar allocation is unknown. Reusing checkpoints does not erase historical fitting: v14's pilots and final fits cost 684.05 CPU wall-seconds Transformer and 1,421.11 GRU, plus statistical fitting. This is not an equal-total-resource comparison.

## Commercial-token economic gate: unchanged and unmeasured

Primary pages rechecked 2026-11-04 around 11:13 UTC: [Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits), [Meta models](https://dev.meta.ai/docs/models), [Runpod prices](https://www.runpod.io/pricing). Meta Contributor remains $0.10/$0.002/$0.20 per million uncached/cached/output; half-price targets remain **$0.05/$0.001/$0.10**. Contributor permits content training and excludes max reasoning, so its retail price is not proof of private/max-reasoning parity or inference cost. Runpod H100 PCIe/SXM list $2.89/$3.99 per GPU-hour, storage extra; no availability or whole-replica quote was obtained.

At 30% margin, output-only cost can consume at most $0.07/M billable output: **3,968.25 generated billable tokens/s per $1/hour** at full utilization, or **6,613.76 at 60%**, before separate expenses. If one hidden token per visible token is unbilled, generated-work requirements double. Always account for the full replica and every hidden/generated token.

Reproduced mixed ledger: 10M uncached + 5M cached input, 2M visible + 2M hidden output. With hidden billed, target revenue $0.905 and 30%-margin total cost budget $0.6335. Assume $1/h total replica cost, 60% utilization, cached work 10%, and $0.10 separate expense per whole ledger: 8k prefill exhausts the budget; at 80k prefill, decode needs **3,917.31 tokens/s**, or **5,565.53** with hidden unbilled. This is illustrative accounting, not observed heavy-user usage or new measured economics. `economics_audit.py/json` and `thresholds.py` retain phase conservation.

CPU byte loss/latency **cannot be compared numerically with commercial subword-token capacity at GPU rental rates**. The new findings expose quality and context failure modes; they do not clear the cost gate. The $500/month goal likewise remains unproven: $500 gross with 30% margin and $150 assumed non-serving costs leaves $200 for serving, or $141.67 under illustrative 20% included tax. Training/failed runs, support, tools, payments, storage/network, idle headroom and failover must be explicit without double counting. Same workload and task quality must be preserved.

## Reproduction and audit

First reproduce v14 to obtain the exact checkpoints and training ledger, and its pinned WikiText data. From `experiments/v15`, with the CPU dependencies in `requirements.txt`:

    python prepare_data.py
    V14_DIR=../v14 WIKITEXT_DATA=../v14 python run.py
    V14_DIR=../v14 WIKITEXT_DATA=../v14 python wikitext_diagnostic.py
    V14_DIR=../v14 WIKITEXT_DATA=../v14 python audit_final.py
    V14_DIR=../v14 WIKITEXT_DATA=../v14 python retime.py
    python summarize.py
    python -m unittest -v test_v15 test_thresholds
    python economics_audit.py

Thirteen final unit checks pass. [Independent audit](audit.md) separately verifies actual-window losses, cached/full parity, causal behavior and an independent attention/RoPE calculation. No timing pass proves whole-host isolation. Raw viewer content and per-window numerical arrays are local regeneration intermediates, not included; hashes are in `intermediate_manifest.json`. Published per-document scores, complete summaries, source, protocol and audit outputs retain the evidence without redistributing corpus text.

## Next justified gate

Do not extrapolate short-context timing to longer-context quality. Train and validate context length explicitly, retaining a competent Transformer comparator, then repeat transfer on a genuinely held-out, broader sample at matched total resources. Long-context retrieval/reasoning and real service-token benchmarks remain required. This experiment supports two narrow findings: domain transfer can reverse a statistical baseline's ranking, and these short-trained Transformer checkpoints fail the tested context extrapolation.
