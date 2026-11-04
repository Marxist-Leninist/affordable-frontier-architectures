"""Independent explicit phase-budget arithmetic; no benchmark or network access."""
import json, math
from pathlib import Path

def report():
    h,u,margin=1.,.6,.3
    uncached,cached,visible,hidden=10e6,5e6,2e6,2e6
    rows=[]
    for billed in [True,False]:
        for other in [0.,.1]:
            for prefill in [8000,80000,800000]:
                revenue=(uncached*.05+cached*.001+(visible+(hidden if billed else 0))*.1)/1e6
                hardware_budget=revenue*(1-margin)-other
                phase_cost=h/(3600*u)
                prefill_seconds=(uncached+.1*cached)/prefill
                remaining_seconds=hardware_budget/phase_cost-prefill_seconds
                rate=(visible+hidden)/remaining_seconds if remaining_seconds>0 else None
                if rate is not None:
                    cost=phase_cost*(prefill_seconds+(visible+hidden)/rate)+other
                    assert math.isclose(cost,revenue*(1-margin),rel_tol=1e-12)
                rows.append(dict(hidden_billable=billed,other_usd=other,prefill_tps=prefill,revenue=revenue,
                    hardware_budget_after_other_usd=hardware_budget,prefill_cost=prefill_seconds*phase_cost,
                    required_busy_generated_tps=rate))
    output={}
    for utilization in [1.,.6]:
        for bill_hidden in [True,False]:
            generated_per_visible=2.
            billed_per_visible=2. if bill_hidden else 1.
            rate=h*1e6*generated_per_visible/(3600*utilization*.1*(1-margin)*billed_per_visible)
            output[f'ut{utilization}_hidden_billable{bill_hidden}']=rate
    assert math.isclose(output['ut1.0_hidden_billableTrue'],3968.2539682539687)
    assert rows[3]['required_busy_generated_tps'] is None
    return dict(scope='Illustrative arithmetic, not measured service capacity or actual user workload',
        hourly_usd=h,utilization=u,margin=margin,prices_per_million=dict(uncached=.05,cached=.001,output=.1),
        cached_work_fraction=.1,ledger=dict(uncached=uncached,cached=cached,visible=visible,hidden=hidden),
        output_only=output,rows=rows)

if __name__=='__main__':
    print(json.dumps(report(),indent=2,allow_nan=False))
