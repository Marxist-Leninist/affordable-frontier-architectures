# v23 source audit

Checked 2026-11-05 11:12 UTC for the 11:15 research run. Baseline: `affordable_frontier_run_20261105_0815.txt` and `frontier_research_v22_economics.md`. Public primary pages were reopened; no paid resources, production access, third-party writes, dataset shards, gated terms or user content submission.

## Prices: no material change from v22

- [Meta Model API pricing](https://dev.meta.ai/docs/pricing-rate-limits): Muse Spark 1.3 Contributor remains **US$0.10/M uncached input, $0.002/M cached input, $0.20/M output**, unchanged from November 3 and v22. Requested half-price targets remain **$0.05/M uncached, $0.001/M cached, $0.10/M output**. Contributor trades permission to train on submitted content for retail discounts; it is neither underlying-cost evidence nor permission to submit user content. No long-context premium is listed; web-search grounding is separately $2.50/1,000 queries. No new claim that tools or hidden work are free follows.
- [Runpod pricing](https://www.runpod.io/pricing), updated November 4: Pods H100 PCIe **$2.89/hour**, H100 SXM **$3.99/hour**; Serverless H100 **$5.39/hour**. Standard network storage remains **$0.07/GB-month under 1TB**, **$0.05 above 1TB**. These are unchanged. Pods prices do not establish clustered fabric, guaranteed availability or measured throughput. Serverless has different paid-time behavior and cannot silently replace the retained Pods assumption.

## Frozen dataset cards: same limitations

- [HuggingFaceFW/fineweb-edu, revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md) remains publicly readable. Real Common Crawl-derived educational web text, filtered with a classifier using Llama3-70B-Instruct annotations. Text-generation use is documented. Card retains ODC-By 1.0 and Common Crawl terms; underlying source rights and privacy obligations remain relevant.
- [Salesforce/wikitext, revision b08601e04326c79dfdd32d625aee71d232d685c3](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md) remains publicly readable. Wikipedia Good/Featured articles for language modeling. Metadata still lists CC-BY-SA-3.0 and GFDL while licensing prose links CC-BY-SA-4.0. Preserve the inconsistency, attribution/share-alike obligations and underlying rights; do not silently resolve it.

This is a frozen-card availability/provenance/license audit, not a new dataset download, hash audit, legal certification or contamination test. Preserve existing experiment configuration, samples, splits, preprocessing, tokenizer, seeds, token counts and held-out separation in the experiment manifests. Repeated-test-selection limits remain.

## Economics retained without rerunning old hypothetical analysis

No rechecked price changes alter v22's mixed API or full-month subscription accounting. Mixed API ledger stays **10M uncached +5M cached +2M visible +2M hidden**: gross revenue **$0.905** if hidden generation is billed, **$0.705** otherwise. Under the retained illustrative 3% payment fee and no included tax, 30%-margin all-in limits remain **$0.614495 / $0.478695**. Hidden generation consumes work in both cases.

Subscription ledger stays **10M input +2M visible +2M hidden per average subscriber/month**, distinct from the API ledger. The unchanged hypothetical 64-H100 pool, $500 gross subscription, assumed 20% included tax, 3% gross payment fee and $150 other-cost reserve retain v22's **1,683-subscriber** 30%-margin boundary, versus **877** at break-even. Preserve its full heavy workload, population-tail scenario, active-window/burst capacity requirements, concurrency and latency caveats. GBP200/GBP20 tiers stay separate. The source check supplies no evidence that these assumptions are achievable or costs complete.

The conditional output-only hardware floor remains **3,968.25 billable output tokens/s per $1/hour**, before other costs, at 100% billable productive utilization; divide by productive utilization once. Actual comparisons require mixed revenue less fees/taxes, measured full work and all-in cost <=70% of net revenue. Frontier-quality parity remains independently unproved. No architectural winner or sustainable service conclusion follows from unchanged sources; fresh measured quality/performance is the useful next contribution.

No blocker to this source audit. Missing frontier-quality, full-system throughput/latency, memory/fabric and lifecycle cost evidence still blocks a feasibility claim.
