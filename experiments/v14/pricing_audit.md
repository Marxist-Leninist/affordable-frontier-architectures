# Independent v14 economic gate

Verified 2026-11-04 around 08:18 UTC. Read-only primary-source checks; no rental, payment, model API request or production operation. This audit independently recalculates the earlier `economics/target_thresholds` formulas. Numerical artifact: `economics_audit.json`; reproduce with `python economics_audit.py` (standard library, Python 3.10+). Embedded assertions check the reference floor, phase conservation in each feasible scenario, and budget exhaustion.

## Current sources

[Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits) lists Muse Spark Contributor at $0.10/M uncached input, $0.002/M cached input and $0.20/M output. The research target is half those retail rates: **$0.05/$0.001/$0.10**. Contributor grants use of supplied content and responses for model training. Retail prices reveal neither underlying inference cost nor subsidy. [Meta model documentation](https://dev.meta.ai/docs/models) says max reasoning is unavailable for Muse Spark 1.3 Contributor. Thus its price does not establish parity with private service or maximum-reasoning results. Hidden-token billing below is an explicit sensitivity, not a newly verified assertion about Meta's accounting.

[Runpod pricing](https://www.runpod.io/pricing), displayed update October 28, 2026, lists Pods: A6000 48GB $0.59/h, RTX4090 24GB $0.89/h, A100 80GB $1.79/h, H100 PCIe 80GB $2.89/h and H100 SXM 80GB $3.99/h. Storage is separately priced. Multi-node H100 SXM cluster pricing says contact sales, so a per-Pod price multiplied by node count does not establish a suitable interconnected cluster quote. Inventory, a deployment quote, tax and real achieved throughput were not checked. All three requested source pages were accessible; no pricing-source blocker occurred.

## Output-only floor

At $0.10/M billable output with 30% revenue margin, all costs share **$0.07/M**. For total allocated replica expense $1/hour, spending that entire envelope on hardware requires **3,968.25 busy aggregate generated tokens/s at 100% utilization**, or **6,613.76 at 60%**. This is before prefill and separate expenses. One hidden token per visible token with visible-only billing doubles these to **7,936.51 / 13,227.51**. Hidden work never vanishes when it is unbilled. The hourly rate must cover the entire replica, including all GPUs, not one card in a multi-card model.

## Mixed ledger and a nonzero expense allocation

Fixed illustration, not observed heavy-user usage: 10M uncached input, 5M cached input, 2M visible output and 2M hidden output. All 4M generated tokens consume decode work. Cached input has assumed 10% uncached prefill work. At 60% calendar utilization, $1/h whole-replica expense and 30% margin:

| Hidden billed? | Other costs per complete ledger | Revenue | Hardware budget after other | Required decode t/s, prefill 8k | Prefill 80k | Prefill 800k |
|---|---:|---:|---:|---:|---:|---:|
| Yes | $0 | $0.905 | $0.6335 | 71,607.59 | 3,233.34 | 2,951.52 |
| Yes | $0.10 | $0.905 | $0.5335 | Infeasible | 3,917.31 | 3,511.13 |
| No | $0 | $0.705 | $0.4935 | Infeasible | 4,279.40 | 3,799.27 |
| No | $0.10 | $0.705 | $0.3935 | Infeasible | 5,565.53 | 4,779.91 |

At 8k prefill, prefill alone costs $0.6076389; infinite decode cannot rescue any row whose hardware budget is lower. Other costs are **per complete ledger**, not per million tokens, and $0.10 is a sensitivity rather than a measured allocation. Input revenue can fund decode, so a mixed requirement can be below the output-only floor without contradiction.

The explicit model is `cost = hourly/(3600*utilization) * ((uncached + 0.1*cached)/prefill_tps + (visible+hidden)/decode_tps) + other`. Costs must satisfy `cost <= revenue*(1-margin)`. It conserves shared serial phase capacity. If prefill/decode genuinely overlap, measure end-to-end capacity rather than granting free overlap or charging the same capacity twice.

## What the experiment can and cannot clear

CPU next-byte timing is not provider subword-token throughput. Teacher-forced scoring, a fixed-prefix microbenchmark, or reciprocal median step latency cannot establish sustained service throughput, real cache-hit cost, end-to-end p95, paid-hour utilization or a dollar allocation. Do not multiply a byte rate by these GPU prices or label a model as passing the economic gate. The meaningful v14 result can be a quality/training/latency tradeoff at its measured byte task; the frontier-token economic threshold remains unmeasured.

Before qualification: preserve fixed tasks and quality, hidden work and tokenization; measure input/context/output distributions, batching, arrivals, cache eviction, queueing, p50/p95 TTFT and inter-token latency, complete replica cost and useful busy utilization. Include unsuccessful requests, training and failed experiments, evaluation, support, tools, payment fees, networking, storage, idle headroom and redundancy without double counting. The 30% figure here is a planning margin after specified costs, not proof of accounting gross or operating profit.

The separate $500 gross/month scenario with $150 assumed non-serving cost and 30% margin leaves $200 serving budget without included tax, or $141.67 with illustrative 20% included tax. Those assumptions and usage must be specified independently. No profitable frontier-quality $500 product, AGI capability or provider subsidy is established.
