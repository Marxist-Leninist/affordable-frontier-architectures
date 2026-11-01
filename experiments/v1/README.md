# v1: fixed-workload break-even frontier and memory/latency gates
2026-11-01. Architecture-neutral neural-language-model research; AGI is an aspiration, not a measured result. Separate directory: the root v0 case study remains unchanged.

## Changed finding
v0 supplied a hypothetical throughput and computed a cost. **v1 solves for the throughput that must actually be measured**, then checks conditional resource gates without granting any architecture a speedup.

For the unchanged **10M input +2M visible +2M hidden output/month** workload, a $500 tax-inclusive subscription, 20% tax scenario, $50/user non-serving reserve, 32 GPUs at $3.99/h with 20% infrastructure uplift, 60% utilization and 8,000 measured prefill tokens/s requires:

| Margin on ex-tax sales | Serving allowance | Minimum measured aggregate decode |
|---|---:|---:|
| 0% (break-even) | $366.67 | 1,021 tokens/s |
| 30% | $241.67 | **1,855 tokens/s** |
| 50% | $158.33 | 4,073 tokens/s |
| 70% | $75.00 | Impossible at that prefill rate: input alone exceeds allowance |

The 30% target therefore needs ~24% higher aggregate decode throughput than v0's unmeasured 1,500 tokens/s. With $50 reserve rather than v0's £1 placeholder, cost at that old operating point is $327.82, yielding about **21.3% margin** on $416.67 net revenue, before any unbudgeted expense. This is an operating contribution-margin scenario: **not a verified net-profit claim**. Training amortization, salaries, finance, taxes, payment fees, tools, support, failures and other costs must actually fit the reserve or be entered separately. $500 cost ceiling and $500 retail subscription are distinct: the latter also has tax and desired profit to fund.

At 128k context, 32×80GB cards, 4-bit weights +15% weight overhead, and 8GB/card workspace, estimated sequence-state capacity is **31** for full latent attention, **177** for 1/8-attention hybrid + assumed 1GB recurrent state, or **579** for pure recurrence with assumed 1GB state. Those are memory calculations, not quality or performance results. At an operating point of exactly20 generated tokens/s per active stream, 30% margin requires ~93 active decoding streams, so full-attention 128k fails this memory gate. Faster per-stream generation can lower needed concurrency; the calculation is not a universal impossibility result. KV compression, additional hardware or a different operating point may change it.

Conditional roofline checks at batch128 and128k also reject the full-attention scenario under the selected 50% bandwidth efficiency. Hybrid/recurrent cases survive those elementary gates, but **surviving does not demonstrate feasibility**: fabric, scheduling, attention/SSM work, uneven routing, and real kernel precision can make performance worse. No architecture gets a claimed throughput or a quality win from this model.

## Execute
From the repository root (Python 3.10+; standard library only):
```
cd experiments/v1
python -m unittest -v
python experiment.py
python cpu_state_probe.py
```
Stdlib only. Eleven tests pass, including frontier/cost identity, cache and idle effects, fixed recurrent state, context growth, precision memory failure, batch expert reuse, and invalid arguments. Outputs: `outputs/summary.json`, 864 economics rows in `economics.csv`, 108 resource rows in `resource_gates.csv`, 24 active-weight/state/precision rows in `active_state_precision.csv`; `cpu_state_probe.json` records a bounded actual CPU functional test.

The CPU probe implements an **untrained 64-state diagonal recurrence**, not Mamba or a language model. At lengths512 and8192 its state payload stays512bytes; chunked updates exactly equal contiguous updates, and copied branch state is independent. It establishes streaming and branching bookkeeping only; it cannot score recall, intelligence, language quality or estimate GPU throughput. Timings are incidental local CPU observations, never scaled to3T.

## Method and assumptions
Currency is USD throughout; no FX claims. Gross price500; tax scenarios0/20%; target margins0/30/50/70%; non-serving reserve50/150 dollars; GPU rates2/3.99/6 dollars; utilization30/60/80%; prefill4k/8k/16k; cache-hit fraction0/50%, cached work fraction10%. Cache sensitivities retain the exact same token ledger; a cache hit reduces assumed work, not requested input. Additional persistent prefix-cache memory is excluded and must be budgeted before approving a real case.

`allowance = price/(1+tax)*(1-margin)-other`

`remaining_decode_seconds = allowance*3600*utilization/hourly_cost - effective_input/prefill_tps`

`required_decode_tps = 4,000,000 / remaining_decode_seconds`

Nonpositive remainder is impossible at that operating point even with infinite decode speed. Utilization accounts for idle capacity once; throughput must be measured during busy operation. Input and output phase-times add; no overlapping free capacity or speculative speedup is assumed.

Resource scenarios keep3T TOTAL weights, vary dense3T versus75/150/300B ACTIVE parameters, weight precision4/8bit, sequence context8k/32k/128k, batch16/64/128 and recurrent state1/4GB. All recurrent sizes are **design sweep inputs**, not sizes derived from an implemented network. The default attention state uses120layers×576values×2bytes/token; hybrid fraction1/8 scales only that state. Layer counts, active params, state size and exact architecture must be jointly dimensioned later.

Weight union per batch is `min(total, active * batch**r)` with r=0/0.5/1: perfect reuse, intermediate and no-reuse-until-saturation. These are scenario endpoints, not a model of actual expert routing; shared dense weights may improve reuse. We must measure distinct expert weights and routing skew. Weight reads include15% overhead. Attention state is read once per step; recurrent state read and write counted. Compute lower bound is2×active×batch FLOPs, deliberately excluding attention/state updates. Step-time floor is max(parameter-matmul time, modeled memory-traffic time). Using50% of rated bandwidth and35% of non-structured-sparse BF16 throughput defines a **conditional** roofline, not a hardware-absolute theorem. Communication and layer sequentiality are omitted. No automatic INT4, binary, SSM or speculative acceleration.

32-card available state budget is579GB: 2,560GB −1,725GB weights −256GB workspace. Eight-bit3T does not fit32cards even before sequence state. Extra replicas for availability and spare capacity add costs. p95 queues and per-user turn arrival traces are not captured by average throughput economics.

## Research evidence and architecture-neutral decision rule
A controlled8B study comparing Mamba, Mamba-2, transformer and hybrid models found pure SSM weaknesses on copying, in-context learning and long-context reasoning, while its hybrid performed strongly. This supports testing both hybrid and pure recurrent candidates rather than declaring either universally superior; it does **not** establish3T/AGI capability. Its speed prediction is not imported into this experiment. Primary paper: https://arxiv.org/abs/2406.07887

Mamba-2 develops state-space duality and more efficient core-layer algorithms. A layer-kernel gain is not a measured full-service cost gain and cannot be assigned to an unimplemented3T architecture. Primary paper: https://arxiv.org/abs/2405.21060

v0 hardware inputs retained as scenarios: https://lambda.ai/instances ($3.99/GPU-hour example, not a multi-node quote) and https://www.nvidia.com/en-us/data-center/h100/ (80GB,3.35TB/s; BF16 dense peak taken as989.5TF/s, half advertised structured-sparse rate). Source inspection date2026-11-01; no purchase or capacity reservation. VAT scenario precedent: https://www.gov.uk/vat-rates .

Any neural language architecture can enter this ledger after producing real parameter/state counts and measured fixed-workload traces. A3T parameter count is not required for AGI and is not itself a quality metric. Keep a matched-quality sparse-attention baseline, an SSM/recurrent candidate and a hybrid candidate; do not restrict winners to a named family.

## Next experimental contract, not unauthorized GPU work
1. Freeze exact prompt/turn traces with all cached and hidden work, max and average context, tool requests and task success criteria. Set p95 first-token and inter-token SLOs; reasoning tokens can delay first visible output, even with good aggregate decode rate.
2. Instantiate genuinely dimensioned small candidate networks on already-authorized resources only. Match training data, compute and evaluation tasks; record all weights including shared experts. No pretrained quality assumption from parameter count.
3. Measure prefill and decode separately across the existing context/batch grid, log actual recurrent/KV bytes, quantization scales, unique expert reads, draft acceptance, fabric utilization, power if relevant, and busy/calendar utilization. Repeat after warmup with distributions, not a single peak.
4. Require exact-recall/copy, multi-hop retrieval, code tests, reasoning and tool-task non-inferiority margins specified before results. Failed quality cannot be repaired by reducing the workload or omitting hidden tokens from billing.
5. Replace hypothetical throughput inputs with measured operating points, reject memory/latency failures, then demand all-in expense evidence. Scaling extrapolations remain hypotheses until validated; no automatic3T training or production change.
