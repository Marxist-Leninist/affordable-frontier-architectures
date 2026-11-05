# Affordable Frontier Architectures

Architecture-neutral research into neural, non-neural and hybrid language-model systems that could advance toward general intelligence at lower real cost for the same workload and quality. Transformer, recurrent/state-space, hybrid, sparse, and other neural designs are candidates—not predetermined winners. This repository contains accounting prototypes and bounded small-model experiments, not a demonstrated frontier/AGI system.

## Latest experiment: v25 distant-prefix content specificity

[Audited v25 report, code and measurements](experiments/v25/README.md). Recomputing neural outputs after replacing distant real text with unrelated text preserves every target and input length. The context-trained Transformer has nearly equal aggregate original/donor loss (3.421446/3.421562 bits per byte; conditional donor-effect interval spans zero), while its genuine512-byte suffix remains better at3.333290. Individual predictions can change greatly, so this is not content blindness or proof of length-only behavior. The artificial boundary and donor dependence limit inference; no useful long-context, frontier-quality or profitability gain is established. [Current sources and economic gates](experiments/v25/pricing_sources.md) remain unchanged.

## Prior experiment: v24 mixed-context tradeoff

[Audited v24 report, code and measurements](experiments/v24/README.md). Fixed 50/50 context training produces an intermediate result rather than beating both specialists: mixed loss is 3.18215/3.60269 bits per byte at nested512/full2048, versus short-trained 3.05462/4.43126 and long-trained 3.30596/3.39622. The same target ledger, initial weights and optimizer updates are used, but compute differs. Fresh held-out real text, seven tests and independent reconstruction support this narrow negative finding. No matched-quality speed or frontier/economic win follows. [Current prices and retained gates](experiments/v24/pricing_sources.md) remain unchanged.

## Prior experiment: v23 matched-context training

[Audited v23 code, results and limitations](experiments/v23/README.md). With identical next-byte training targets and optimizer updates, continuing the small full-attention Transformer at 2,048 context improves its full-context loss from 4.35784 to 3.56376 bits/byte. A 512-reset continuation instead reaches 4.57272. The longer-context arm wins that comparison by 1.00896 bits/byte but remains worse at 512 context, and longer context still hurts its own matched-target score by 0.08799. This repairs a training/evaluation mismatch, not evidence of distant-information reasoning, a matched-budget architecture victory, frontier parity or profitable deployment.

[Restored v19–v22 source, reports and detailed text results](archives/20261105_v19_v22/README.md) preserve preceding results, including the v22 finding that dynamic INT8 shrinks model serialization about 2.72x but slows the Transformer paths while modestly accelerating GRU. Corpus, weights and regenerable binary arrays are excluded from that text archive. [Current pricing and retained economic gates](experiments/v23/pricing_sources.md) keep the $0.05/$0.001/$0.10 per-million targets and 30% net-revenue margin unchanged.

## Prior experiment: v18 unreachable-prefix pruning

[Full v18 report, audited code and measurements](experiments/v18/README.md). On fresh FineWeb-Edu rows 300–399, skipping context outside the two-layer local model’s 255-byte prediction receptive field reduces a 2,048-byte-prefix CPU probe from **58.872 to 26.350 ms**, versus v17’s optimized chunked baseline. Median paired end speedup is **2.146×** (95/96 favorable). Preserving absolute RoPE positions passes all three seeds’ numerical quality gates; the failed position-reset control is retained. This saves work precisely because the model cannot use distant context, not because it solves long-range tasks. Commercial economics and frontier quality remain unproven.

## Prior experiment: v17 matched-quality chunked prefill

[Full v17 report, audited code and measurements](experiments/v17/README.md). On fresh FineWeb-Edu rows 200–299, fixed 128-byte prefill chunks preserve local-Transformer losses to floating-point precision while reducing the 2,048-byte-prefix CPU probe from **117.295 to 55.709 ms**. The median paired speed ratio is **2.020×**; all 96 paired long-prefix probes improve. Both paths receive the same output-head optimization. Chunking is slightly slower at 512 bytes and neutral at 128, so this is workload-dependent. No frontier-task or commercial-token economic pass is claimed.

## Prior experiment: v16 fixed-span attention safeguard

[Full v16 report, audited source and measured results](experiments/v16/README.md). On fresh FineWeb-Edu rows 100–199, restricting each layer to 128 attended positions with unchanged weights restores the tiny Transformer's 512-context loss from **5.58813 to 3.02608 bits/byte**, slightly better than reset128 (**3.04641**). Original 128-window scores are unchanged. Actual persistent cache falls **76.65%** to 130,064 bytes, but the measured CPU probe is **about 7.1% slower**. This is a tested quality/memory safeguard, not a speedup, long-range task result, or frontier-cost pass.

## Prior experiment: v15 transfer and context failure

[Full v15 report, tested source and numerical results](experiments/v15/README.md). Frozen v14 models evaluated on 100 FineWeb-Edu documents reverse the statistical baseline's ranking: GRU **3.00234**, Transformer **3.07665**, statistical **3.43396 bits/byte**. On identical scored targets, increasing input from 128 to 512 bytes sharply degrades these short-context-trained Transformer checkpoints (**3.07523 → 5.61408**) while GRU slightly improves. Independent alignment, cached/full and attention checks support the result; a post-hoc WikiText control reproduces the length failure. This is not a general architecture verdict. No comparable-quality pair or commercial-token economic pass is established.

## Prior experiment: v14 stronger training and quality-cost gate

[Full v14 report, tested source and numerical results](experiments/v14/README.md). Four times the sampled training targets and a wider learning-rate grid improve the tiny Transformer to **2.35970 bits/byte**, GRU to **2.23878**, and statistical control to **2.22609**. The neural gap roughly halves versus v12, but **none of the prespecified matched-quality gates passes**, so no equal-quality speedup is claimed. Direct CPU prefill-plus-update timing and live statistical RAM are measured; these are not commercial-token serving costs. Three final seeds, independent tests and artifact checks. [Current 30%-margin economic gate](experiments/v14/pricing_audit.md) remains unmeasured by these byte models.

## Prior experiment: v13 statistical–neural complementarity

[Full v13 report, tested source and numerical results](experiments/v13/README.md). Frozen validation-selected mixtures improve prediction on the same pinned WikiText-2 test split: Transformer+statistics **2.27116 bits/byte**, GRU+statistics **2.24240**, versus statistical-only **2.33517**. The best hybrid reduces loss by 3.97% versus the statistical model, at additive memory and inference cost. A 2.82MB serialized table occupies ~32.21MB as Python objects before runtime overhead. Three seeds, eight independent checks; no equal-compute, AGI or frontier-economics claim. [Updated official pricing caveats](experiments/v13/pricing.md) preserve the $500 sustainability goal.

## Prior experiment: v12 real-text neural and non-neural controls

[Full v12 report, tested source and numerical results](experiments/v12/README.md). On pinned Hugging Face WikiText-2 with the same sampled training targets, a modern two-block Transformer scored **2.7259 bits/byte**, parameter-matched gated recurrence **2.4833**, and an interpolated non-neural byte 5-gram **2.3352**. The GRU has faster cached CPU decoding but slower training/prefill; the n-gram uses larger sparse storage. This is short-context sample-limited prediction, not evidence of general reasoning, AGI or frontier economics. Prior experiments and failed controls remain preserved.

## Initial case study: 3T total-parameter serving economics — 2026-11-01

## Useful result first
A $500/user/month **real operating-cost ceiling** is not ruled out for a shared 3T sparse model, but there is no measured 3T throughput or quality evidence here. Under the explicit illustrative assumptions below, 10 million uncached input + 2 million visible output + 2 million hidden reasoning tokens/month costs **£222.26 ($277.82)** in allocated infrastructure, or **$279.07** including the assumed £1/user other cost. This fits $500 before omitted expenses, but does **not** meet a 70% margin on a $500 VAT-inclusive subscription. It is a falsifiable budget hypothesis, not a performance claim. The exact same fixed token workload is preserved in each sensitivity comparison.

A customer's hypothetical $10,000 API retail bill versus $500 subscription is a 20× **retail-price** ratio, not evidence of $10,000 underlying compute cost or verified subsidy. Need the exact API input/cached-input/output rates, actual token ledger including hidden reasoning, quality, latency and tool-work to compare the same workload. Token counts alone do not guarantee equivalent quality. No assertion about any provider's actual subscription price or subsidy is made.

## Reproduce
From the repository root (Python 3.10+; standard library only):

```sh
python -m unittest -v
python model.py > results.json
```

Files: `model.py`, `test_model.py`, `results.json`. Eight tests pass: weight bytes, budgets, linearity, utilization, capacity conservation, KV growth, hidden/cache accounting, invalid inputs. Stdlib only, no network access or live model calls. No production modifications, rentals, purchases or training.

## Assumptions, not benchmark results
- 32 H100 SXM 80 GB GPUs; illustrative $3.99/GPU/hour list-price input from Lambda's 8-GPU instance table, accessed 2026-11-01. Four nodes at that price **do not** imply cross-node fabric availability, a reserved cluster quote, or achievable serving performance.
- £0.80/USD is a scenario parameter, **not a verified current FX rate**. Infrastructure uplift 20% (placeholder for networking/storage/operations), 730 h/month; £89,478/month per always-on replica. Taxes on infrastructure assumed recoverable or otherwise to be added; 20% consumer VAT scenario only, not personal tax advice.
- Effective busy aggregate prefill 8,000 tokens/s. Decode sensitivities 500 / 1,500 / 3,000 output tokens/s. All are hypothetical targets, not extrapolated benchmark guarantees. Every hidden reasoning token counts as output work.
- 60% calendar utilization, including idle time/headroom. Measured busy throughput should already include kernel, routing and fabric penalties; do not count those twice. Phase costs add: `seconds = input/input_tps + output/output_tps`. No unearned simultaneous prefill/decode capacity or speculative speedup is credited.
- Baseline infrastructure cost rates: **£7.09/M input, £37.83/M output**. These are cost allocations, not proposed API selling prices. At 30% utilization each doubles. Cache hits do not become free: configurable effective-work fraction, plus omitted cache storage costs; assumed 50% cached input at 10% of prefill work is only a sensitivity.
- Baseline monthly workloads, input / total generated output: 100k / 20k costs £1.47; 1M / 200k £14.66; 10M / 2M £146.60. At decode 500–3,000 t/s, last workload spans £297.92–£108.76. Add 2M hidden tokens to that workload: baseline £222.26; with stated cache sensitivity £190.34.

## Subscription and profit accounting
For gross VAT-inclusive price P, desired margin m on ex-VAT sales, and £1 other/user cost: `serving_budget = P/1.2*(1-m)-1`. At m=70%, £20 / £200 / $500 (=£400 assumed FX) permit **£4 / £49 / £99** serving spend. These are deliberately explicit business assumptions, not laws of model design. At a $500 price, £222.26 + £1 expenses produce ~33% margin on £333.33 net sales **before omitted expenses**; £146.60 + £1 gives ~56%. Cost equal to revenue is break-even, not profit.

A dedicated baseline replica costs ~£89k/month, so a £20/£200/$500 individual budget cannot fund one. Pooling is essential. At baseline workload 100k/20k, 60%-utilization capacity is ~61,037 users/replica; minimum £20 subscribers to fund this replica at target 70% margin is ~22,370. For 1M/200k workload capacity is ~6,104, with ~1,827 £200 subscribers needed at the same target. These arithmetic windows do not guarantee demand, bursts, latency or feasible batching. An extra full failover replica roughly doubles replica expense absent productive sharing. Training, amortization, evaluation, staff, payment fees, abuse/fraud, retrieval/tools and enterprise support must be priced separately, not silently assumed covered by £1 or 20% uplift.

## Architectural research hypothesis
This is independent research. MoE is an established technique, not claimed as a new invention here. Target **3T total** with **100–200B active** parameters/token; illustrative center 150B. One accounting sketch is 60B always-active attention/shared/dense weights and 2.94T expert weights, with ~90B experts selected per token (about 3.06% expert weight activation). This is a budget allocation, not a fully dimensioned implementation. Shared experts count in always-active cost. Test quality at matched training compute and fixed user tasks before assuming this sparse activity supports frontier capability.

Candidate stack: balanced fine-grained experts; node-aware routing and communication overlap; compressed latent attention or measured GQA alternative; continuous batching and paged KV; prefix caching where valid; quantization-aware training or validated post-training quantization; optional exact speculative verification/MTP after an independent speed/quality gate. Scaling parameters alone does not yield frontier quality or intelligence.

### Weight storage and bandwidth
3T weight-only decimal storage: BF16 **6 TB**, FP8 3 TB, 4-bit **1.5 TB**, ideal ternary 1.58-bit 592.5 GB, ideal binary **375 GB**. Raw 80 GB card minima are 75 / 38 / 19 / 8 / 5 respectively, before activations, KV, redundancy, scales, padding or topology. Dense and MoE of equal total size both store all weights unless deliberately offloaded; active parameter savings do not make total weights disappear. BF16 could require a topology-aligned ~96-card example with headroom, not the 32-card baseline. 4-bit +15% storage overhead is 1.725 TB, leaving 579 GB on 32 cards after an illustrative 256 GB workspace reservation.

Binary/ternary values are lower bounds, not a practical BF16 checkpoint conversion or proven 3T quality recipe. BitNet's ternary representation is not binary and real implementations may use 2-bit packing, scales and higher-precision components. No speedup is granted merely from reducing bits.

Single-token weight reads alone: dense BF16 ~6 TB, dense 4-bit 1.5 TB; 150B-active 4-bit ~75 GB (before scales). H100 SXM rated bandwidth 3.35 TB/s/card is an optimistic hardware bound. Simple `bytes/aggregate_bandwidth` ignores layer serialization, uneven experts, all-to-all, tensor parallel collectives, KV reads and kernel efficiency. Batching amortizes reused weights but MoE batches touch different experts; never divide active bytes by full batch size without measuring overlap. CPU/NVMe offload may lower rental RAM cost but can break interactive bandwidth/latency. Dense 3T uses ~20× the parameter matmul FLOPs of 150B-active at the same token count, but that is **not** a 20× all-in price proof.

### Context, concurrency and latency
Illustrative compressed KV: 120 layers × (512 latent +64 positional) values ×2 bytes/token. 8,192 /32,768 /131,072 tokens require **1.13 /4.53 /18.12 GB per sequence** before fragmentation or extra states. With 579 GB available that is at most ~511 /127 /31 sequences. Compare illustrative BF16 GQA KV with 16 KV heads and head dimension128: `2*120*16*128*2 = 983,040 bytes/token`, ~8.05 GB/8k sequence. Attention compute still rises with context even if KV compression lowers storage; baseline throughput cannot be reused blindly at 128k.

1,500 aggregate decode t/s at 20 t/s per active user implies ~75 simultaneous decoding sequences during decode; at 128k the simple memory budget only holds ~31, so that combined latency/context assumption fails without more hardware or lower throughput. Aggregate throughput is not a single user's token speed. Need p50/p95 time-to-first-token, inter-token latency, queueing, batch mix and routing skew under real arrival traces. Specify average context as well as maximum context.

## Speculation, SAT, NAT, quality and training
Exact speculative decoding can preserve a target model's distribution with correct verification, but draft acceptance, draft cost, longer verification and batching determine savings; default credit is 1×. SAT (semi-autoregressive) and NAT (non-autoregressive) change factorization and may need distillation or refinement; original translation speed/quality results cannot establish frontier chat equivalence or fixed 20× savings. Test at identical quality and workload rather than cutting allowance or reasoning depth.

Training is wholly separate. A rough parameter-matmul planning expression `6*active_parameters*training_tokens` excludes attention, routing overhead, data, experiments, post-training and utilization. 150B active ×20T tokens yields ~1.8e25 FLOPs before those additions; dense3T gives ~3.6e26. These are not quotes or full training budgets. Do not amortize an unknown training bill to zero.

## Next bounded validation gate
Define an immutable workload ledger from the intended heavy user: uncached/cached input, visible/hidden output, context distribution, task quality thresholds, tool cost, concurrency and latency SLO. Run a small isolated MoE/attention/quantization prototype at matched compute; measure real busy prefill/decode throughput and quality, then replace scenario assumptions. Reject any configuration failing quality, memory or p95 latency even if spreadsheet cost looks good. No 3T run or production edit is authorized by this document.

## Primary sources (accessed 2026-11-01)
- Lambda H100 SXM instance price $3.99/GPU/hour in 8-GPU table, excludes applicable tax: https://lambda.ai/instances
- NVIDIA H100 SXM specifications, 80GB and 3.35TB/s. Advertised tensor throughput marked with sparsity; do not mistake it for dense throughput: https://www.nvidia.com/en-us/data-center/h100/
- UK standard VAT rate20%: https://www.gov.uk/vat-rates
- DeepSeek-V3 (671B total/37B active; MLA, shared/routed experts, MTP) supports architectural precedent, not 3T quality extrapolation: https://arxiv.org/abs/2412.19437 and https://arxiv.org/html/2412.19437v2
- BitNet ternary training research, not proof of 3T results: https://arxiv.org/abs/2402.17764
- Exact speculative decoding: https://arxiv.org/abs/2211.17192
- Semi-autoregressive translation: https://arxiv.org/abs/1808.08583
- Non-autoregressive translation: https://arxiv.org/abs/1711.02281

## Broader architecture track: not transformer-only
The research target is frontier-capable neural architectures within the cost envelope, not a predetermined transformer or an established 'AGI-level' claim. Include (A) sparse transformer+MLA baseline, (B) selective state-space/recurrent backbone such as Mamba/Mamba-2, (C) attention–SSM hybrid with sparse expert FFNs. Pure recurrent SSM decoding maintains a finite state rather than full attention KV for every past token, reducing the context-length-dependent cache burden; it still reads model weights and computes state updates. State size/dtype, convolution state, branching/rollback and batching must be measured. Finite compressed state may lose exact context information; evaluate multi-hop retrieval, exact recall, coding, long reasoning and tool-use, not just perplexity or small benchmarks.

Hybrid attention layers preserve explicit retrieval paths while reducing their count, but their remaining KV cache still grows with context. First approximation is `hybrid_state = attention_layer_fraction * baseline_KV + SSM_state`; this is not a throughput prediction. Test attention fractions 1/2,1/4,1/8 at matched compute, then reject failures rather than changing the workload. Jamba provides a published hybrid MoE precedent. None of these papers proves AGI, 3T frontier equivalence or a 20× all-in saving. The generic token-cost accounting remains usable after actual throughput and state footprints replace the transformer scenario inputs.

Additional primary sources: Mamba https://arxiv.org/abs/2312.00752 ; Mamba-2/structured state-space duality https://arxiv.org/abs/2405.21060 ; Jamba hybrid MoE https://arxiv.org/abs/2403.19887 ; Jamba-1.5 https://arxiv.org/abs/2408.12570 .
