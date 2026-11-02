# v3: exact speculation is not a free all-in speedup

2026-11-02, 17:15 research run. Read v2 at commit `55021eaabc8e43b6c19e1b3826b5225fc51a1a4a` before choosing this experiment. Earlier versions are unchanged.

## New result

**Decode acceleration alone cannot rescue v2's shared 8-hour arrival-window economics while prefill capacity and monthly infrastructure stay unchanged.**

We tested 180 conditional speculative-decoding operating points: acceptance 50/70/90/98%, draft blocks 1/2/4/8/16, draft token latency 1/5/10 ms, and verification incremental-cost slopes 0.05/0.2/1.0. Of these, 98 have hypothetical single-stream speedup above one, but **none pass the combined capacity/margin necessary gate**. This follows from the unchanged prefill bottleneck, not a failed real-model benchmark.

- v2's 64-GPU pool needs 913 subscribers for its 30% contribution-margin scenario. Its prefill stage only supports 691.2 subscribers' full work arriving over 8 hours/day, even with infinitely fast decode.
- An illustrative extra 8-GPU draft pool raises monthly spending to **$248,209.92** and the economic minimum to **1,028 subscribers**. Cost including the $50 reserve is then **$291.45/user**, approximately 30.05% margin on $416.67 net revenue, **if** capacity and the reserve suffice. That is not demonstrated.
- At 1,028 subscribers, prefill needs at least **1.487×** its baseline throughput and decode at least **2.380×** baseline aggregate throughput, before tail-latency headroom. These are necessary thresholds, not achieved results.
- A representative 90%-acceptance case with 5 ms draft tokens and verification slope 0.2 peaks at block 4: **1.845×** hypothetical stream speedup. Block 16 falls to **1.432×** as additional draft/verification costs outweigh useful output.
- The most optimistic point reaches 6.792× hypothetical stream speedup but still cannot fix prefill. Single-stream speedup is **not automatically aggregate batch speedup**.

The ledger stays **10M input + 2M visible + 2M hidden tokens/user/month**. No context, reasoning budget, quality standard or allowance was reduced. The illustrative $10,000 API retail bill versus $500 subscription is a retail-price ratio, not verified provider cost, subsidy or 20× savings.

## Actual CPU experiment: correctness, not capability

`speculative.py` implements modified rejection sampling on an **untrained four-symbol Markov target and draft**, with prefix-dependent probabilities. It drafts a block, verifies proposals against target conditionals, and either emits a residual correction on rejection or a target bonus token after full acceptance. Rejected suffixes are discarded; state follows committed tokens only.

For target p and draft q, proposal x is accepted with min(1, p(x)/q(x)). Rejection draws from normalized max(p−q, 0). Accepted mass min(p, q) plus rejection mass reconstructs p. Tests include identical/disjoint support, zero draft support and the exact one-step identity.

Actual CPU probes emit 400,000 tokens each at block lengths 1, 4 and 8, **1.2M total**. Maximum empirical transition-row total-variation distances from the target are **0.003265, 0.001723 and 0.001925**. Algebraic error is at floating-point roundoff. **18 tests pass**. Independent review also checked disjoint-support sampling and an alternating chain after repeated rejection.

These checks detect toy implementation mistakes; they are not a formal proof for every implementation or a language/reasoning benchmark. Equal distributions need not produce identical sequences under the same seed. No trained network, GPU kernel, frontier benchmark or production inference ran. CPU timings are not extrapolated to 3T. Preserving a weak target does not improve its intelligence or fix language drift.

Exact speculation can wrap any autoregressive neural architecture with exact conditionals and correct state rollback, including transformers, sparse MoE, recurrent/state-space models and hybrids. Verification costs differ. Approximate SAT/NAT is not automatically exact speculative sampling.

## Conditional timing and economic model

For constant IID acceptance a and block length k:

```
expected emitted = sum(a**i for i in range(k+1))
iteration time = k*draft_time + target_step_time*(1+beta*k) + overhead
stream speedup = target_step_time * expected_emitted / iteration_time
```

Baseline target step is 50 ms; iteration overhead is 1 ms. The report counts k+1 target-scored positions and k draft proposals per iteration, including wasted proposals. IID acceptance and affine verification costs are sweep assumptions, not measurements. Real acceptance varies with prompts, temperature, quantization, tasks and batch conditions. Target numerics, tokenizer and sampling transforms must match the intended distribution.

Cost is charged for 32 prefill + 32 decode + 8 draft GPUs at $3.99/GPU-hour, 20% infrastructure uplift and 720 hours. All idle time is included; there is no second utilization divisor. The zero-extra-GPU sensitivity still charges draft latency and does not establish contention-free execution. It only shows that prefill remains limiting even without an extra hardware bill.

**This is not a latency pass.** Tokens emitted in bursts have inter-burst gaps equal to iteration time. Those can exceed v2's 50 ms inter-token SLO despite faster average output. Pacing needs buffers and can change first-visible latency. No TPOT quantiles or full speculative arrival queue were simulated: candidates fail an earlier necessary capacity gate. Hidden tokens still consume draft/verification work and delay visible output.

## Remaining all-in costs and resource gates

- **Revenue and overhead:** $500 gross, a 20% tax scenario, $416.67 net sales, 30% target margin and $50 non-serving reserve. Actual payment fees, support, tools, salaries, failures and amortized training/R&D must fit the reserve or be added. An unknown training bill cannot be amortized to zero. Nonrecoverable infrastructure taxes also need adding.
- **Pooling:** the full 72-GPU dedicated bill is $248,209.92/month, nowhere near $500 for one subscriber. Extra replicas or failover add cost. No multi-node availability/fabric quote was obtained.
- **Weights and state:** v2's illustrative 3T target uses 4-bit payload plus 15% packing/scales, 1.725 TB per target replica or 3.45 TB for two copies. This is not AGILLM's actual native/packed representation. Draft weights/scales and state must fit the extra hardware; draft dimensions and memory fit are **unverified**. Quantization earns no speedup without suitable measured kernels.
- **Context and caching:** requests retain 5k input, growing to 7k total, plus speculative positions and rollback buffers. Recurrence needs snapshots or recomputation; attention needs target and draft KV state. Cached-input fraction stays zero. Draft prompt encoding, correction/catch-up work, KV memory, cache eviction, network congestion and batch interference are optimistic omissions, not free operations. The extra eight GPUs alone do not establish draft capacity.
- **Bandwidth and storage:** v2's assumed 50 ms state transfer and 20% infrastructure uplift remain unverified. Real fabric, temporary buffers, host storage and network costs must be measured.
- **Stretch budgets:** using the old explicit scenario FX of £0.80/USD, not a current quote, £200 allows $95.83 serving cost after tax, margin and the same reserve. It requires at least 2,591 subscribers for this pool. £20 cannot cover the reserve. These reject this scenario, not all possible products.
- **AGILLM routing:** v2 source-verified per-token top-1 routing against SHA256 `1241dead5f786b7fa398c8f9859de182cc9c97341718ff9ffc8990f5e174942c`. One FFN expert is selected per token/layer, but a batch can touch all experts. This run assumes no dense AGILLM baseline and did not inspect the migrating trainer or claim current runtime verification.

## Primary sources and next measurement

Accessed 2026-11-02:

- [Leviathan, Kalman and Matias, ICML 2023](https://arxiv.org/abs/2211.17192): exact speculative decoding and parallel verification. Algorithmic basis, not an imported speedup guarantee.
- [Chen et al., 2023](https://arxiv.org/abs/2302.01318): speculative sampling and target-distribution preservation within hardware numerics, with 70B-model experiments. Not evidence of 3T economics here.
- [Lambda official pricing](https://lambda.ai/instances): 8-GPU H100 SXM 80GB listing at $3.99/GPU-hour, plus applicable tax. No rental or commitment made.

No transformer, Mamba, hybrid or recurrent winner is selected by reputation. A smaller exact draft can accelerate a fixed target without replacing its quality, but acceptance, draft resources and target verification must be measured together. The next discriminating experiment should measure a complete target/draft pair at matched task quality, including prompt prefill, batch throughput, acceptance distribution, KV/state rollback and first-visible/inter-token tails. Use already-authorized compute only. A 3T parameter total is neither required nor evidence of AGI.

## Reproduce

Python 3.10+, standard library only. From `experiments/v3`:

```
python -m unittest -v
python experiment.py
```

Outputs contain 180 envelope rows, three 400k-token probes and the exact summary configuration. No Actions workflow, credentials, visibility, license, production process or training state was changed. No paid compute was acquired.
