"""Bounded deterministic accounting sensitivity, not observed workloads or performance."""
import json, math
from pathlib import Path

DAYS=30
SECONDS=DAYS*24*3600
INPUT=10_000_000
VISIBLE=2_000_000
HIDDEN=2_000_000

def calculate(reserve=150, subscribers=1000, active_hours=8, burst=1, cached_fraction=0, tax=.2, fee=.03, price=500):
    # Fees are 3% of gross, excluded from net revenue, never charged again in reserve.
    net=price/(1+tax)-price*fee
    allowance=.7*net-reserve
    hourly=64*3.99*1.2
    monthly=hourly*720
    n_min=math.ceil(monthly/allowance) if allowance>0 else None
    available=DAYS*active_hours*3600/burst
    effective_input=INPUT*((1-cached_fraction)+.1*cached_fraction)
    return dict(reserve=reserve,subscribers=subscribers,active_hours=active_hours,burst=burst,cached_fraction=cached_fraction,
        gross=price,net=net,serving_allowance=allowance,monthly_pool_cost=monthly,
        minimum_subscribers_30pct=n_min,minimum_subscribers_break_even=math.ceil(monthly/(net-reserve)) if net>reserve else None,
        contribution_margin_at_n=(net-reserve-monthly/subscribers)/net,
        required_prefill_tps=subscribers*effective_input/available,
        required_generated_decode_tps=subscribers*(VISIBLE+HIDDEN)/available,
        hypothetical_capacity_prefill=8000*available/effective_input,
        hypothetical_capacity_decode=1500*available/(VISIBLE+HIDDEN))

def report():
    rows=[calculate(reserve=r,subscribers=n,active_hours=a,burst=b,cached_fraction=c)
          for r in [50,150] for n in [100,1000,2000] for a in [8,24] for b in [1,4] for c in [0,.5]]
    boundaries=[]
    for reserve in [50,150]:
        n=calculate(reserve=reserve)['minimum_subscribers_30pct']
        for burst in [1,4]:
            boundaries.append(calculate(reserve=reserve,subscribers=n,burst=burst))
    # Exact workload conservation: 80% at .5x, 15% at 2x and 5% at 6x; E[m]=1.
    mix=[(.8,.5),(.15,2),(.05,6)]
    mean=sum(p*m for p,m in mix)
    variance=sum(p*(m-mean)**2 for p,m in mix)
    assert math.isclose(mean,1)
    assert math.isclose(variance,1.6)
    assert math.isclose(calculate()['monthly_pool_cost'],220631.04)
    for row in boundaries:
        assert row['contribution_margin_at_n']>=.3
        assert calculate(reserve=row['reserve'],subscribers=row['subscribers']-1)['contribution_margin_at_n']<.3
    # Fixed USD/GBP scenario rate inherited from v2, NOT a current FX quote.
    stretch=[dict(gbp=g,scenario_usd_per_gbp=1.25,**calculate(price=g*1.25)) for g in [200,20]]
    return dict(scope='Analytical necessary gates only; unmeasured throughput/workload/quality; no deployment claim',
        reference_prices=dict(meta_input=.1,meta_cached=.002,meta_output=.2,runpod_h100_sxm_hourly=3.99),
        targets_per_million=dict(input=.05,cached=.001,output=.1),
        subscription_ledger=dict(input=INPUT,visible=VISIBLE,hidden=HIDDEN,sessions=2000,prompt_per_session=5000,visible_per_session=1000,hidden_per_session=1000),
        population=dict(fractions_and_multipliers=mix,mean_multiplier=mean,variance_multiplier=variance),
        boundaries=boundaries,rows=rows,stretch=stretch,
        validation="Workload conservation and integer margin boundaries asserted")

if __name__=='__main__':
    Path(__file__).with_name('results.json').write_text(json.dumps(report(),indent=2,allow_nan=False)+'\n')
    print(json.dumps(report()['boundaries'],indent=2))
