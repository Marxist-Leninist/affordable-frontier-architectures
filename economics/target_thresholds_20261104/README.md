# Frontier price target: threshold audit, 2026-11-04

This is a reproducible supplement to the private repository's v13 economics.py. No model benchmark, paid compute, rental, or production change occurred. This supplement preserves all prior results. Existing v13 output_tps_required supports margin, but its generated table calls the default margin=0. The new table explicitly uses 30%, correcting the interpretation rather than claiming a new experimental result.

## Main result
At $0.10/M billable generated output and a 30% revenue margin, the entire cost envelope is $0.07/M. If hardware consumes that whole envelope, each $1/hour requires **3,968.25 busy aggregate generated tokens/s at 100% utilization**, or **6,613.76 at 60%**. This is a favorable phase-only floor, not an all-in service prediction. Hardware must receive LESS than $0.07/M if other costs share the envelope. With 20% hourly infrastructure uplift and 60% utilization, the bar is 7,936.51 per $1/hour of base hardware; reserve another $0.02/M for separate expenses and it becomes 11,111.11.

Runpod displayed rates, checked 2026-11-04: A6000 $0.59/h requires 2,341 ideal / 3,902 at 60%; RTX4090 $0.89/h requires 3,532 / 5,886; A100 $1.79/h requires 7,103 / 11,839; H100 PCIe $2.89/h requires 11,468 / 19,114; H100 SXM $3.99/h requires 15,833 / 26,389 tokens/s. Source: https://www.runpod.io/pricing (page says updated 2026-10-28). Storage is extra; inventory and actual deployment quotes were not checked. GPU memory/interconnect differ: these are cost floors, not interchangeable capable deployments.

Lambda B200 is $6.69/GPU/hour on an 8-GPU instance: https://lambda.ai/instances . This is not a one-GPU rental at that price; the 8-card bill is $53.52/hour. Multiply threshold by actual GPU count and include all replicas. Taxes extra; recoverability is not assumed. No cluster fabric or availability guarantee is implied.

## Mixed workload changes the answer
The v13 fixed illustrative ledger is 10M uncached input, 5M cached input, 2M visible + 2M hidden output. Its target retail bill is $0.905 if all hidden tokens are billed, giving **$0.6335 total cost budget** at 30%. This ledger is NOT measured heavy-user usage. Input revenue can legitimately help fund decode, but prefill work consumes budget too. A per-output phase threshold is not necessarily a lower bound for the mixed service because input and output can cross-subsidize.

Example sensitivities at 60% utilization, no separate expenses, cached input costing 10% of uncached prefill work:
- $1/hour and 8k prefill tokens/s require 71,608 generated decode tokens/s; 80k prefill lowers the requirement to 3,233.
- $3.99/hour and 8k prefill exhausts the budget before any output. At 80k prefill, decode must reach 18,895 tokens/s.
- Original 32-card $3.99/h, 20% uplift baseline costs $153.216/h. With 8k prefill, prefill alone costs $93.10 versus the $0.6335 total envelope. Even 800k prefill is insufficient. Infinite decode cannot rescue an already exhausted prefill budget. These are illustrative assumptions, NOT measurements.

Hidden output always consumes work. If output billing includes it, count it in both revenue and work. If $0.10/M means visible-only billing, one hidden token per visible token doubles the output-only generated-throughput requirement. The output_threshold other_per_million argument is a cost per million VISIBLE tokens; hidden tokens do not multiply that cost automatically. Use a total explicit allocation consistent with your ledger.

## Subscription is a separate product
$500 gross monthly with 30% margin and $150 assumed non-serving allocation leaves $200 serving budget before included tax, or $141.67 with an illustrative 20% included tax. This does NOT show a profitable plan or a market-equivalent allowance. If the target API workload costs $0.905 at target rates, a $500 plan must define its additional value and actual usage distribution. Do not treat the scenario ledger as sufficient proof of a compelling subscription.

The cost definitions must distinguish gross margin/COGS from operating margin after training, research, staff and other allocations. Our 30% envelope after all specified costs is a planning contribution/profit target; accounting classification needs its own decision. Fixed $150 and 20% overhead are placeholders, not measured all-in figures.

## Measurement gate before any architecture qualifies
Measure complete replica hourly costs including CPU/RAM, storage/network, cache residency and failover. Record paid-hours utilization separately from busy throughput; include retries, abandoned requests and hidden work. Validate serving tokenization against the target billing tokenizer. Byte-model CPU rates cannot be used as provider tokens/s.

Keep workload, task quality and acceptance thresholds fixed. Report context-length distribution, cache reuse/eviction, concurrency and arrivals, p50/p95 first-token and inter-token latency, and achieved aggregate prefill/decode or end-to-end workload throughput. Do not assume a short-context throughput extends to long context or that batch throughput gives individual users that speed. Training/failed experiments/tools/payment/support allocations remain explicit. No architecture here has demonstrated frontier quality or this price target.

Our mixed model adds prefill and decode GPU-equivalent seconds. If actual serving overlaps them, benchmark end-to-end capacity rather than granting free overlap or charging overlapping capacity twice. Hardware rate uplift and separate costs must not cover the same expense twice.

## Reproduce
Python 3.10+ standard library only, from this directory:

    python -m unittest -v
    python thresholds.py > results.json

Seven tests pass: reference math, utilization, hidden billing, exhausted budgets, mixed phase conservation, prefill infeasibility and invalid inputs. A null required rate means budget infeasible at the specified prefill, not zero required work.

Private source reviewed:
https://github.com/Marxist-Leninist/affordable-frontier-architectures/blob/main/experiments/v13/economics.py
https://github.com/Marxist-Leninist/affordable-frontier-architectures/blob/main/experiments/v13/pricing.md
