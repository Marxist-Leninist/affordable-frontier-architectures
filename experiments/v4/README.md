# v4: prefix reuse must survive tenant isolation, eviction and model updates

2026-11-03 05:15 run. Repository head and v3 report read at pinned commit332a64705de206f9a4ed7fc300a776c6bb4886f7; private visibility verified. This is a new bounded CPU trace experiment, not a language-model or GPU benchmark.

## New finding

V3 showed decode acceleration alone leaves a prefill bottleneck. At its1028-subscriber economic threshold, the optimistic input-work gate requires at least **32.7626% of all input tokens cached**, equivalent to **39.9934% request hits** when4096of5000input tokens are eligible.

We replayed the full2,056,000-request month in each of10 deterministic scenarios (20.56M requests total). Each user still gets10M input,2M visible and2M hidden tokens. Four distinct4096-token prefixes per tenant; the remaining904tokens are unique. A model revision every500requests/user invalidates reuse. Cache keys isolate tenants, prefix identity and model revision. Full prefixes are atomic LRU entries; partial-block reuse is not simulated.

- Fully interleaved arrivals: capacities64,256 and1028prefixes all yield **zero hits**, despite heavy repeated text. Capacity4112 yields81.26464% cached input.
- Twenty-request tenant-local bursts: capacity64 yields **65.536% cached input**, clearing only the optimistic prefill-work gate. Capacity4112 again yields81.26464%.
- Both orderings contain exactly the same requests/tokens. They are alternative synthetic arrival patterns, NOT permission to reorder customers or delay requests to manufacture locality. Neither is a measured user trace or latency pass.
- **None establishes all-in feasibility.** V3's unchanged decode stage still needs2.37963× aggregate throughput. No measured decoder acceleration, cache-transfer time, model quality or full queue tails were obtained.

Thus cached fraction cannot be selected as a free discount knob. Actual arrival locality, tenant policy, model revisions and physical cache capacity can flip the prefill result while preserving the full workload.

## Architecture-neutral memory gate

Illustrative GQA KV geometry:120layers ×8KV heads ×128head width ×2(K,V) ×2bytes ×4096prefix tokens = **1.875GiB per prefix**. This is assumed geometry, not AGILLM or an identified3T architecture.

-64prefixes:120GiB;256:480GiB;1028:1927.5GiB;4112:7710GiB.
- An explicitly hypothetical64MiB recurrent-state checkpoint would instead require4/16/64.25/257GiB. This is a dimensional sensitivity, not a measured Mamba state size or equal-quality substitute. Recurrent caching must include convolution/SSM states, position and all other necessary context; hybrids also retain attention KV. Dense and sparse MoE transformers use the same cache logic, but MoE routing changes active computation, not necessarily KV size.
- For direct continuity with v2, its latent-attention sensitivity120×576×2bytes/token uses0.566231GB per4096-token prefix:1028entries need582.086GB, exceeding its579GB prefill remainder before cache overhead or active state. At most1022such entries fit that remainder; no addedcapacity is assumed. This is distinct from the GQA geometry above.
- Count aggregate shard bytes once, then multiply by physically duplicated replicas. Cached prefixes may share immutable active KV; do not double count shared bytes. Nonshared active suffix/state, draft state, allocator metadata/scales, rollback buffers, weights and workspace still occupy memory. This experiment gives cache payload only and does not certify fit or deployment feasibility.
- V3's illustrative3T target uses1.725TB weights/replica (4bit payload plus15% overhead), two target replicas plus a separate draft pool. Those assumptions are unchanged. They are not proof of AGI or of suitable quantization kernels. A cache requiring extra machines must add their full cost, rather than using v3's unchanged bill.

## Cost and timing ledger

Carry v3's72H100 GPU pool:32prefill+32decode+8draft,720hours,$3.99/GPU-hour,20% infrastructure uplift, **$248,209.92/month**, including idle capacity. Dedicated cost is the whole bill; pooled cost at1028users is$241.45/user before$50reserve. $500gross with20%tax scenario gives$416.67net; conditional total$291.45 leaves30.052% margin. This is an accounting scenario, not achieved sustainable economics: capacity, quality and omitted costs remain unverified.

The$50reserve must cover payment fees, support, tools, failures and amortized training/R&D; unknown training costs are not zero. Bandwidth, storage, nonrecoverable taxes and network fabric must fit the uplift or be added. Lambda advertises no egress fees for its service, but that does not eliminate cross-provider transfers or cache latency. Cache lookup, hashing, copying, eviction and cold recomputation overhead are omitted from the optimistic gate.

Token savings are not proportional hardware savings automatically: uncached suffix attention still reads cached K/V; attention costs, batching and kernel shape differ. Formula691.2/(1-cached_fraction) is only a constant-work-per-input-token optimistic gate inherited from v2, not measured prefill throughput. The bill is not multiplied by a cached-token discount. Input ledger remains10M, regardless of fewer computed prefixes. Output and hidden reasoning tokens are unchanged, and reasoning still delays first-visible output. Context stays5000prompt+2000generated. No workload or quality reduction is counted as success.

The$10,000API retail versus$500subscription ratio is a retail reference, not evidence of provider subsidy or underlying costs. At the old explicit scenario FX£0.80/USD (not currentquote),£200 leaves$95.83serving allowance after tax/30%margin/$50reserve, requiring2591users on thispool;£20 cannot cover thatreserve. Neither target is achieved here. No unlimited-use promise.

AGILLM historical per-token top1 routing remains source-verified by v2 at SHA1241dead5f786b7fa398c8f9859de182cc9c97341718ff9ffc8990f5e174942c. This run did not inspect live migrating trainer; no densebaseline assumed. Cache-only experiments cannot select a capability winner among transformers, MoE, Mamba, linearattention or hybrids.

## Correctness, evidence and next experiment

12 unit tests pass: cold misses, LRU, tenant/version isolation, unchanged trace multiset, token conservation, capacity validation and no false all-in success. These test the synthetic accounting implementation, NOT equality of actual cached neural outputs. Exact reuse requires identical model/tokenizer/adapter/position/mask and context identities plus correct state handling; a tuple simulation is no production security implementation.

Next useful test: capture a consented anonymized prefix-reuse trace with actual tenant boundaries, then run cached/uncached matched neural inference at fixed outputs/quality and arrival times. Measure first-visible/inter-token tails, aggregate throughput, resident bytes, hashing/transfer cost, version churn, and cold starts. No new paid resources are justified by these toy results.

Primary sources checked2026-11-03:
- [vLLM prefix-cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/): block-based reuse, eviction and optional salt isolation. Our model simplifies these into whole-prefix entries.
- [Mamba paper](https://arxiv.org/abs/2312.00752): selective recurrent state-space architecture; motivates a state-checkpoint sensitivity, not the arbitrary64MiB figure or equal frontier quality.
- [Lambda official pricing](https://lambda.ai/instances):8GPU H10080GB listing$3.99/GPU-hour plus applicabletax; no availability/fabric guarantee or rental made.

Reproduce with Python3.10+ standardlibrary: `python -m unittest -v` then `python experiment.py`. Results in results.json; no external packages, paid compute, production edits, workflows, licenses or visibility changes.
