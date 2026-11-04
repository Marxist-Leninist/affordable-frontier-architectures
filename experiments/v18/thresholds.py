"""USD sensitivity model. No benchmark, network calls, or third-party dependencies."""
import json
from math import isfinite

def check(**values):
    if any(not isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('Inputs must be finite and nonnegative')

def output_threshold(hourly, utilization=1., price=.10, margin=.30,
                     other_per_million=0., hidden_per_visible=0., hidden_billable=True):
    """Output-only phase. Hourly is TOTAL allocated replica cost, not per-GPU by default.
    Return minimum busy generated tps; all generated work counts, only billable earns.
    Other cost excludes the hourly expense, avoiding double counting.
    """
    check(**locals_without_boolean(locals()))
    if not 0 < utilization <= 1 or not 0 <= margin < 1 or price <= 0:
        raise ValueError('Invalid utilization, price or margin')
    generated_per_visible=1+hidden_per_visible
    billable_per_visible=generated_per_visible if hidden_billable else 1
    budget_per_visible=price*(1-margin)*billable_per_visible-other_per_million
    if budget_per_visible <= 0:
        return None
    return hourly*1e6*generated_per_visible/(3600*utilization*budget_per_visible)

def locals_without_boolean(d):
    return {k:v for k,v in d.items() if not isinstance(v,bool)}

def mixed_threshold(hourly, utilization, prefill_tps, uncached, cached, visible,
                    hidden, cached_work=.1, input_price=.05, cached_price=.001,
                    output_price=.1, margin=.3, other=0., hidden_billable=True):
    """Sequential phase accounting. Rates per million, token ledger in actual tokens.
    Returns None if prefill alone consumes budget. Add all replica expenses to hourly;
    add tools/training/support allocation to other. Do not allocate them twice.
    """
    check(**locals_without_boolean(locals()))
    if not 0 < utilization <= 1 or prefill_tps <= 0 or hourly <= 0 or not 0 <= margin < 1 or not 0 <= cached_work <= 1:
        raise ValueError('Invalid input')
    revenue=(uncached*input_price+cached*cached_price+(visible+(hidden if hidden_billable else 0))*output_price)/1e6
    budget=revenue*(1-margin)-other
    cost_second=hourly/(3600*utilization)
    prefill_seconds=(uncached+cached*cached_work)/prefill_tps
    remaining=budget/cost_second-prefill_seconds
    return dict(revenue=revenue,total_cost_budget=budget,prefill_cost=prefill_seconds*cost_second,
                required_busy_generated_tps=(visible+hidden)/remaining if remaining>0 else None)

def report():
    rates={'Runpod RTX A6000 48GB':.59,'Runpod RTX4090 24GB':.89,
           'Runpod A100 80GB':1.79,'Runpod H100 PCIe 80GB':2.89,'Runpod H100 SXM 80GB':3.99,
           'Lambda B200 180GB per GPU in 8-GPU instance':6.69}
    rows=[]
    for name,rate in rates.items():
        rows.append(dict(reference=name,usd_gpu_hour=rate,
          ideal_tps=output_threshold(rate),utilization_60pct_tps=output_threshold(rate,.6),
          uplift20pct_utilization60pct_tps=output_threshold(rate*1.2,.6),
          uplift20pct_utilization60pct_other_02_per_M_tps=output_threshold(rate*1.2,.6,other_per_million=.02)))
    return dict(scope='Required throughput, NOT measured performance or quality parity. Rates checked 2026-11-04.',
        reference_sources=['https://www.runpod.io/pricing','https://lambda.ai/instances'],
        price_per_M_output=.1,margin=.3,reference_thresholds=rows,
        mixed_fixed_ledger=dict(uncached=10_000_000,cached=5_000_000,visible=2_000_000,hidden=2_000_000),
        mixed_sensitivity=[dict(hourly=h,prefill_tps=p,**mixed_threshold(h,.6,p,10e6,5e6,2e6,2e6)) for h in [1,3.99,32*3.99*1.2] for p in [8000,80000,800000]],
        hidden_unbilled_1to1_at_1usd_hour=output_threshold(1,hidden_per_visible=1,hidden_billable=False))

if __name__=='__main__': print(json.dumps(report(),indent=2,allow_nan=False))
