# v16: a fixed attention span repairs the tested context failure

2026-11-04. Weight-preserving inference intervention, evaluated on a fresh 100-row Hugging Face FineWeb-Edu slice. No paid compute, training, production changes, or frontier/AGI claim.

## New result

Restricting each Transformer layer to **128 attended positions including the current position** removes the large 512-byte-context loss failure observed in v15. The window was fixed from the original training context, before scoring the new rows. No weights, hyperparameters or seeds were fitted or selected on this evaluation.

All conditions below score the **same 58,880 tail targets**:

| Model/condition | Mean bits/byte | Initialization sample SD |
|---|---:|---:|
| Original Transformer, full 512-byte attention | 5.588126 | .224522 |
| Same Transformer, reset to final 128 input bytes | 3.046410 | .015341 |
| Same weights, 128-position local attention over 512 input bytes | **3.026077** | .013898 |
| Frozen GRU, full 512-byte input | 2.951686 | .029108 |
| Frozen statistical five-gram, full 512-byte input | 3.355238 | deterministic |

Lower is better. The intervention improves every paired window relative to the failed full-attention condition, and improves mean loss relative to resetting the model in all three seeds. All primary 128-byte-window scores are **exactly unchanged** in this run. This supports a narrow practical result: the tested failure is avoidable with a fixed attention-span restriction, without retraining these checkpoints. It does not establish the unique cause of the original failure or a general Transformer limitation.

The prescribed two-sided .02-bpb comparable-quality screen against reset128 is not met by all seeds: differences are **−.021449, −.021312 and −.018238**. Two miss the strict equality band because their loss is slightly **better**, not because of a harmful regression. This is descriptive, not a statistical equivalence/noninferiority test. We report the improvement without moving the gate after seeing results or claiming an equal-quality speedup.

## Smaller persistent cache, but no speedup here

After a 512-byte prefix and 32 observed cached updates, actual retained state storage is **130,064 bytes** for local attention versus **557,056 bytes** for the original Transformer: a **76.65% reduction**. The local value includes 16 bytes of absolute-position metadata across two layers. Independent checks verify that trimmed tensors own compact backing storage, rather than being views retaining a large allocation.

The straightforward implementation is slightly slower, not faster:

| Model | 512-prefix end-to-end p50 probe | Prefill p50 | Mean cached-update p50 | Retained state storage |
|---|---:|---:|---:|---:|
| Original Transformer | 27.463 ms | 6.723 ms | .633 ms | 557,056 bytes |
| Local Transformer | 29.401 ms | 7.783 ms | .658 ms | 130,064 bytes |
| GRU | 23.701 ms | 14.837 ms | .290 ms | 512 bytes |
| Statistical model | 11.771 ms | .331 ms | .358 ms | Four-byte logical suffix, plus count table |

Neural entries are medians of seed-specific p50 values. The local probe is about 7.1% slower than full attention on this CPU despite fixing loss and reducing retained cache. Explicit masks, position bookkeeping and cache copies have costs. No inference-dollar saving follows merely from fewer cache bytes. The statistical table remains about 54.02 MB of Python objects according to v14's measurement; its suffix is not its total memory.

## Intervention and limits

`local_attention.py` retains the same QKV/output weights and parameter names. Its causal mask admits only positions from `current-127` through `current`. It keeps **127 previous K/V positions** per layer for the next call, cloning them into compact storage, plus an int64 absolute-position counter. RoPE continues from the actual absolute position after cache trimming. All original weights are checked bit-for-bit unchanged.

This is a standard sliding/local-attention idea, **not an invention claim**; [Mistral 7B](https://arxiv.org/abs/2310.06825) is one prior example of sliding-window attention. Our fixed inference intervention is not a reproduction of that trained large model, and its results cannot be imported here.

The reset reference starts cold at the suffix boundary; local attention already has a populated history. The small improvement over reset may therefore reflect warm-start/local-context effects, not useful reasoning over distant text. This experiment does not isolate those mechanisms.

Two stacked local-attention layers can transmit information across up to roughly **255 input positions**, so this is not equivalent to resetting the whole model every 128 bytes. It also does not retain arbitrary distant facts. A longer-context task could lose necessary information when attention is restricted; this experiment measures next-byte loss, not exact recall, reasoning, instruction following or task accuracy.

Persistent KV is bounded, but this implementation still computes dense long-prefix attention with an explicit mask and full prefill activations/logits. It does **not** prove linear prefill cost, a peak-memory reduction of 76.65%, total-process RAM savings of that size, or optimized accelerator performance. Model weights, temporary allocations and runtime remain. The baseline is v14's stronger-trained 131,904-parameter two-block RoPE/RMSNorm/SwiGLU Transformer, not a converged or frontier-quality model. GRU has 132,160 parameters. No architecture is presumed to win.

## Fresh data and locked evaluation

Dataset attribution: HuggingFaceFW/FineWeb-Edu, [`sample-10BT`, pinned revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md). The card declares [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/), also subject to [Common Crawl terms](https://commoncrawl.org/terms-of-use); underlying text may have third-party rights. Corpus text is not redistributed.

Use viewer rows **100–199**, whereas v15 used 0–99. No content-based row selection or filtering. Their 407,643 original UTF-8 bytes are capped at the first 4,097 bytes per document, retaining 255,009 bytes. Byte caps may cut a codepoint; this is byte prediction, not text re-decoding. Raw response and per-document hashes verify the snapshot. The moving viewer endpoint must match the recorded revision and hashes; otherwise reproduction stops rather than silently replacing the sample.

- Primary invariance check: 1,954 complete 128-target windows, **250,112 targets**, all 100 documents. Model state resets each window.
- Main paired test: 460 complete 512-target windows across 98 documents; score only the final 128 targets, **58,880 identical targets per condition**. Reset128 uses the exact corresponding input suffix. No state crosses document boundaries.
- No exact whole-document byte-hash duplicate against v15 was found. This does not exclude partial/near duplicates, overlap with training, or other contamination. The first rows of a parent sample are still an ordered convenience sample, not a representative probability sample.
- Reuse all v14 final 4,000-step checkpoints for seeds 1/2/3. Verify checkpoint/source/ledger hashes. Rebuild the statistical table from the original 4,096,000 sampled training targets. No parameter, learning-rate, checkpoint, seed, n-gram order or attention-window search occurs here.
- The intervention was motivated by prior v15 test findings, but its fixed window is checked on new rows. These are new evaluation observations, not an untouched global research program or broad confirmation.

Bits/byte is cross-entropy, not accuracy or intelligence. Initialization SD does not measure corpus-level uncertainty. Different text/reset protocols mean cross-version losses should not be treated as a controlled training improvement.

## Timing protocol and actual cost accounting

Timing is a separate process after the scoring and independent CPU audits. All models and both prefix lengths use the same first eight documents eligible for 544 input bytes, with row IDs recorded in `timings.json`. Single CPU thread, float32, batch 1. Each sequence computes **33 next-byte distributions**: one after prefill, plus 32 teacher-forced cached updates. Five repetitions per document discard two warmups, leaving 24 local probes.

p95 is across local sequence probes, not production or individual inter-token p95. Decode times summarize each sequence's mean update time. Sampling, queues, network, tokenizer, tools, scheduling and service overhead are absent. Fixed model order and uncontrolled host scheduling remain limitations. No whole-host isolation is claimed. The probe reaches 160/544 observed bytes for the two prefix lengths; primary128 parity alone does not establish parity for the following cached updates. Cross-version timings were taken at different times and must not be compared as identical hardware conditions.

Main scoring run: **60.35 seconds**, including **14.03 seconds** rebuilding statistical counts, excluding development, audit, separate benchmark and postprocessing. No incremental paid compute was purchased; CPU dollar allocation is unknown. Frozen checkpoint reuse does not erase v14's historical fitting/search costs: 684.05 s Transformer and 1,421.11 s GRU, plus statistical fitting. This is not a resource-matched training contest.

## Existing economic requirements are not cleared

[Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits) and [Runpod pricing](https://www.runpod.io/pricing) were rechecked 2026-11-04 around 14:14 UTC. Half-Contributor target rates remain **$0.05/M uncached input, $0.001/M cached input, $0.10/M output**. Contributor content-training permission and unavailable max reasoning remain non-equivalence caveats; retail prices do not disclose inference costs. H100 PCIe/SXM reference prices remain $2.89/$3.99 per GPU-hour, with storage extra and no availability or whole-replica quote.

With a 30% margin, the output-only floor is **3,968.25 billable generated tokens/s per $1/hour** at full utilization or **6,613.76 at 60%**, before separate expenses. All generated hidden work counts; visible-only billing with one hidden token per visible doubles the required generated rate.

The unchanged illustrative mixed ledger is 10M uncached + 5M cached input and 2M visible + 2M hidden output. At $1/h total replica expense, 60% utilization, 10% cached-work assumption and $0.10 other cost per complete ledger, 8k prefill is infeasible; 80k prefill requires **3,917.31 generated tokens/s**, or **5,565.53** if hidden output is unbilled. `economics_audit.py/json` reproduce the same phase-conserving arithmetic; it is not new measured serving performance or an observed user workload.

**CPU bytes are not commercial subword tokens.** The observed cache reduction and quality recovery cannot qualify these thresholds or a sustainable $500 product. At $500 gross monthly, 30% margin and $150 assumed other costs leave $200 serving spend, or $141.67 under illustrative 20% included tax. Training/failed runs, support, tools, payments, storage/network, idle headroom and failover must be included without double counting. Preserve task quality and workload before claiming savings.

## Reproduce and audit

First reproduce v14 and its pinned WikiText data/checkpoints. From `experiments/v16`, with CPU dependencies from `requirements.txt`:

    python prepare_data.py
    V14_DIR=../v14 WIKITEXT_DATA=../v14 python run.py
    V14_DIR=../v14 WIKITEXT_DATA=../v14 python audit_final.py
    python -m unittest -v test_local_attention test_thresholds
    V14_DIR=../v14 WIKITEXT_DATA=../v14 python benchmark.py
    python summarize.py
    python economics_audit.py > economics_audit.json

Sixteen final unit checks pass. The [independent audit](audit.md) verifies weights, short-context invariance, explicit attention math, actual compact cache storage, absolute offsets and full/chunked parity, including actual evaluation windows. It does not freshly rescore every corpus target. Raw rows and per-window score arrays are local regeneration intermediates; their hashes are retained, while per-document scores, source, complete summaries and audit outputs are published without corpus text.

## Next decision

Carry forward the fixed-span safeguard as a testable candidate, not a deployment recommendation. The next substantive quality gate needs tasks that genuinely require distant information and a properly context-trained strong Transformer baseline. Efficient kernels could change latency, but storage savings alone are insufficient. No paid compute or production activation is authorized by this report.
