# Preserved economic scenario assumptions

GBP200/GBP20 remain separate unproven stretch tiers, not currency substitutions.

## Historical prices checked November 15, not newly refreshed by v69

[Runpod pricing](https://www.runpod.io/pricing), page updated November 12, still lists H100 PCIe $2.89/hour and SXM $4.39/hour. [OpenAI pricing](https://developers.openai.com/api/docs/pricing) still lists standard search $10/1,000 calls plus model-priced search-content tokens and a full twenty-minute 1-GB container $0.03. The page also mentions minute billing for eligible sessions with a five-minute minimum; this does not justify shortening the unchanged twenty-minute planning workload. Quotes are not capacity reservations, invoices or measured serving costs.

## Same explicit workload and 30% margin accounting

Both fixed scenarios cover all 30 days. They are historical v43 planning assumptions, retained through v55–v59, not measured customer usage and not reconstruction of missing v52's ledger.

- Typical: 300 sessions × 10 calls; 8k mean/16k peak input; assumed 75% cache hit; 6M uncached plus 18M cached input; 6M visible plus 6M extra hidden output; 600 searches; 300 full container sessions; six burst calls/minute and two concurrent calls; $10 provisional overhead.
- Heavy: 3,000 sessions × 20 calls; 32k mean/64k peak input; assumed 80% cache hit; 384M uncached plus 1,536M cached input; 240M visible plus 240M extra hidden output; 30,000 searches; 3,000 full container sessions; 60 burst calls/minute and 20 concurrent calls; $25 provisional overhead.

At $500 subscription revenue, an illustrative 3% collection deduction leaves $485 net receipts. A 30% margin on those receipts permits $339.50 all other costs. Heavy tools cost $390; adding assumed overhead gives $415 and leaves -$75.50 before model computation or other unmeasured costs. Typical tools and overhead total $25, leaving $314.50; a positive remainder is only necessary, not sufficient, for profitability. Collection fees are already deducted and must not be counted twice.

At target prices $0.05/$0.001/$0.10 per million uncached/cached/visible-output tokens, the analogous budgets after tools and overhead remain -$24.376678 typical and -$384.624256 heavy. No hidden output, retry, tool session, context or peak load was removed to improve a result. That sentence records the prior audit, not a rerun of economic code in v69.

Full costs still include allocated GPU hours and idle replicas, CPU/RAM, storage/cache writes, bandwidth, orchestration, monitoring/support/safety, retries, hidden inference, tools, licensing and training/R&D/capital allocations. Provisional overhead is not a measured complete total. Any extra search content must be included if not already counted in inputs. Actual cache hits, GPU throughput/utilization, tail latency, bursts, safety, recall and frontier-equivalent quality need separate measurement. CPU diagnostic time must not be translated into GPU or commercial costs.

The small historical language-only gates, even if reproduced, cannot override v43's known recall failures or lack of demonstrated speed improvement. No paid compute, production change or gated agreement was used in the research.


