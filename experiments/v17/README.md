# v17: chunked prefill with matched measured quality

2026-11-04. Bounded CPU implementation experiment on fresh Hugging Face FineWeb-Edu rows 200–299. No paid compute, retraining, production changes or frontier/AGI claim.

## Question and fair comparator

v16's fixed-span attention repaired the short-trained Transformer's context failure and reduced retained KV, but its simple CPU implementation was slower. v17 asks a different question: can processing prefill in fixed 128-byte chunks preserve that model's predictions while reducing the cost of large masked attention operations?

**Both monolithic and chunked paths compute the vocabulary output head only for requested final positions.** The baseline therefore receives the same obvious output-head optimization. Quality scoring requests the final 128 positions; timing requests only the final position. There is no model/seed/chunk-size selection on the new data. The chunk size is fixed to the existing 128-position attention span.

## Result: useful at 2,048 bytes, not at every length

**All six seed/length quality comparisons pass.** The largest per-window loss difference is **2.362e-7 bits/byte**, the largest tail-logit difference is **3.815e-5**, and retained cache storage matches. This directly matches measured prediction quality for an implementation speed comparison; it does not establish matching frontier-task quality.

| Prefix | Local monolithic end-to-end p50 | Local chunked end-to-end p50 | Median paired monolithic/chunked ratio |
|---|---:|---:|---:|
| 128 bytes | 23.253 ms | 23.181 ms | .997× |
| 512 bytes | 28.888 ms | 29.523 ms | .976× |
| 2,048 bytes | **117.295 ms** | **55.709 ms** | **2.020×** |

At 2,048 bytes, all **96 paired retained probes** (32 per seed) are faster with chunking. Median paired prefill speedup is **2.680×**; median seed prefill p50 falls from **92.250 to 33.465 ms**. The end-to-end probe includes the same 32 cached updates, so its speedup is smaller. At 512 bytes, chunking is modestly slower; at 128, the small differences look like timing noise. All prespecified lengths are retained rather than reporting only the favorable one.

Both local implementations score **3.045317 bpb** on the 512-byte suffix test and **2.994677 bpb** on the 2,048-byte suffix test, to six decimals. Those are different target sets, so their difference is not a context-quality gain. The full-attention Transformer controls score 5.627692/5.576996 and GRU 2.967403/2.916385 respectively. Neither control is claimed to have matching task quality. Their 2,048-prefix p50 probe times are 70.014 ms (full attention) and 68.243 ms (GRU); those cross-model times do not establish an architectural cost winner.

The neural scoring script took **88.36 seconds**, excluding separate statistical evaluation, audits and benchmark. No paid compute or training was used. The improvement is a real measured CPU implementation result under the conditions below, not a production-throughput or billing claim.

The non-neural control scores 3.424136/3.407272 bpb on the same 512/2,048 target sets. Its low local timing does not compensate for its different prediction quality.


## What is and is not preserved

The embedding, two attention/FFN blocks, layer normalization, weights and bounded KV-cache implementation are unchanged from v16. `prefill.py` runs the same blocks with cached state, collects only the requested final hidden positions, and applies the final norm/head once. The output head does not feed back into the backbone or cache, so omitting unused projections does not remove model information.

Chunked execution changes floating-point operation grouping, so **numerical equivalence**, not bitwise identity or an absolute generated-sequence guarantee, is claimed. The frozen gate requires every scored window's loss difference <=2e-6 bpb and maximum local-tail logit difference <=1e-4. Independent tests cover partial final chunks, tails spanning chunks, ordinary all-position head parity, and subsequent decoding. No approximation such as quantization, pruning or truncating extra state is introduced beyond the already-existing v16 local-attention policy.

This is an implementation comparison of the same model, not an architectural quality victory. The model remains the stronger-trained but tiny v14 Transformer (131,904 parameters), with a 128-position per-layer attention span. It is not a converged large Transformer or frontier reference. Two layers can propagate information across about 255 input positions; chunking does not restore arbitrary long-range retrieval. A 2,048-byte prefix measurement is not evidence of 2,048-byte task memory.

## Locked data and scoring

Attribution: HuggingFaceFW/FineWeb-Edu, [`sample-10BT`, revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md). The card declares [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) and [Common Crawl terms](https://commoncrawl.org/terms-of-use); underlying web text may retain third-party rights. Corpus text is not redistributed.

Use exactly viewer rows 200–299, following v15's rows 0–99 and v16's 100–199. No content-based filtering or adaptation. The 492,444 original UTF-8 bytes are capped at 4,097 bytes per document, retaining **272,220 bytes**. Caps may end inside a codepoint because scoring is byte-based. Source response and document hashes identify the snapshot. The viewer endpoint is moving, so its revision and hashes must match or reproduction stops. A matching retained local snapshot or reconstruction from pinned source is needed if the viewer changes.

- At 512 input bytes: **496 windows, 63,488 scored tail targets, 98 documents**.
- At 2,048 input bytes: **94 windows, 12,032 scored tail targets, 61 documents**.
- Each condition at a given length scores identical final 128 targets of each complete document window, with state reset at the window boundary. No state crosses documents. The target sets differ across lengths, so their loss levels are not a controlled context-length comparison.
- All frozen v14 seeds 1/2/3 are used. Weight/source hashes are verified. No training, learning-rate, checkpoint, attention-span or chunk-size search occurs. Statistical counts are rebuilt from the original v14 ledger with the previously selected order.

These ordered slices are convenience samples, not representative probability samples or official held-out FineWeb splits. New row indices do not guarantee absence of partial duplicates or historical training overlap. Initialization spread is not a population confidence interval. Bits/byte is predictive cross-entropy, not accuracy, reasoning quality or intelligence.

## Timing protocol

The benchmark is separate from scoring and starts after the independent CPU audits. It uses the same first eight documents of at least 2,080 bytes at every tested prefix length (128/512/2,048). Actual row IDs are recorded. Single CPU thread, float32, batch 1; no accelerator or production service is involved.

Every probe computes one next-byte distribution after prefill and 32 teacher-forced cached updates: **33 distributions**, not exactly 32 sampled generated bytes. Six repetitions per document discard the first two, retaining **32 probes per model/seed/length**. Variants are randomly permuted per document/repetition using a fixed RNG seed. This reduces fixed-order bias but does not guarantee perfectly balanced order counts. Local monolithic/chunked probes are paired by seed, document and repetition. Statistical timing is performed once using the first seed's schedule, not repeated as fake independent training seeds.

Tables summarize medians across seed-specific p50 values. The paired-ratio summary separately reports the median of individual monolithic/chunked ratios; that need not equal the ratio of aggregate medians. All prespecified lengths and raw samples are retained. p95 is across local sequences, not service or individual inter-token p95. Decode latency is a sequence average. Input construction is outside the timer; full probability/softmax computation is included. Sampling, network, queues, tokenizer, tools and service scheduling are excluded.

The runtime reports an AMD EPYC 9V74 80-Core Processor; only one PyTorch compute thread was requested. See `runtime_info.json` for Python/library details. Reported host core count is not experiment parallelism or an exclusive allocation.

Host interference and a small sample remain limitations; no whole-host isolation is claimed. Do not compare these absolute times with v16 as though they were obtained under identical conditions. This experiment optimizes a common serving operation, not a complete service workload.

## Memory interpretation and implementation limits

The audit intercepts the actual explicit local-attention mask shapes. At a 2,048-byte monolithic prefill the mask has **2,048 × 2,048 = 4,194,304 Boolean elements**; fixed chunking's largest is **128 × 255 = 32,640**. This describes the largest mask, not measured process peak memory, total allocations, an allocated attention-score matrix, or a GPU-speed prediction. Intermediate chunks still perform work, and smaller chunks incur Python/cache-copy overhead.

Both local paths retain the same **130,064-byte** batch-1 cache, including absolute-position metadata. Chunking does not improve that persistent-cache number over v16. Model weights, temporary activations, allocator/interpreter and runtime remain. Statistical suffix storage is not the whole model: the inherited count table measured about 54.02 MB of Python objects in v14, not remeasured here.

No new training occurred. Frozen-checkpoint reuse does not erase v14's original fitting/search costs. The main scoring runtime and statistical rebuild time are recorded in their respective result files; development, audits, benchmark and postprocessing are separate. No incremental paid compute was bought, but CPU resource allocation in dollars is unknown.

## Existing economic requirements remain separate

[Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits) and [Runpod pricing](https://www.runpod.io/pricing) were rechecked 2026-11-04 around 17:23 UTC. Half-Contributor target rates remain **$0.05/M uncached input, $0.001/M cached input and $0.10/M output**. Contributor content-training permission and unavailable max reasoning prevent automatic equivalence to a private maximum-reasoning service; retail prices do not reveal inference cost. H100 PCIe/SXM references remain $2.89/$3.99 per GPU-hour, storage extra, with no inventory or whole-replica quote.

At 30% margin, the output-only requirement remains **3,968.25 billable generated tokens/s per $1/hour** at full utilization, or **6,613.76 at 60%**, before separate expenses. Unbilled hidden tokens still consume work. The unchanged illustrative ledger (10M uncached + 5M cached input, 2M visible + 2M hidden output) at $1/h total replica cost, 60% utilization, 10% cached-work assumption and $0.10 other cost is infeasible at 8k prefill; 80k prefill requires 3,917.31 generated t/s, or 5,565.53 with hidden output unbilled. Reproduced accounting is in `economics_audit.py/json` and `thresholds.py`; it is not new measured economics or observed user usage.

**Numerically matched byte-model quality does not establish frontier task quality or commercial-token capacity.** CPU byte latency cannot be substituted into GPU-priced subword-token economics. A $500 gross monthly product with 30% margin and $150 assumed other costs leaves $200 serving budget, or $141.67 under illustrative 20% included tax; profitability remains unproven. Include training/failed experiments, tools, support, payment costs, networking/storage, idle headroom and redundancy without double counting. Preserve the actual task workload and quality before claiming savings.

## Reproduction and audit

First reproduce v14's pinned WikiText data, training ledger and checkpoints. From `experiments/v17` with CPU dependencies from `requirements.txt`:

    python prepare_data.py
    V14_DIR=../v14 python run.py
    V14_DIR=../v14 WIKITEXT_DATA=../v14 python statistical_control.py
    V14_DIR=../v14 python audit_final.py
    python -m unittest -v test_prefill test_thresholds
    V14_DIR=../v14 WIKITEXT_DATA=../v14 python benchmark.py
    python summarize.py
    python economics_audit.py > economics_audit.json

Fourteen final unit checks pass. See [independent audit](audit.md), protocol, numerical results, per-document scores and complete timing samples. Raw dataset rows and per-window neural score arrays are local regeneration intermediates, not redistributed; their hashes are retained. No timing or model result is deployed to production.

## Next decision

Treat any measured gain as an implementation improvement within this model's narrow quality regime. A meaningful next capability gate still needs tasks requiring distant information, stronger properly trained baselines and actual tokenized service measurements. Chunking alone does not add model capability.
