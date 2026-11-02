"""Deterministic full-month FCFS queue gates, not an LM/GPU benchmark."""
from dataclasses import dataclass, asdict
import heapq, math, json, csv
from pathlib import Path
DAYS=30
DAY=86400
MONTH=DAYS*DAY
REQUESTS_PER_USER=2000
INPUT=5000
VISIBLE=1000
HIDDEN=1000

@dataclass(frozen=True)
class Pool:
    prefill_tps: float=8000
    decode_slots: int=100
    stream_tps: float=20
    prefill_gpus: int=32
    decode_gpus: int=32
    gpu_hour: float=3.99
    uplift: float=1.2
    state_transfer_seconds: float=0.05
    gross_usd: float=500
    tax: float=.2
    other_usd: float=50
    target_margin: float=.3
    first_visible_slo_seconds: float=60
    def __post_init__(self):
        nums=(self.prefill_tps,self.decode_slots,self.stream_tps,self.prefill_gpus,self.decode_gpus,self.gpu_hour,self.uplift,self.gross_usd,self.first_visible_slo_seconds)
        if any(not math.isfinite(x) for x in (self.tax,self.other_usd,self.state_transfer_seconds,self.target_margin)):raise ValueError('finite cost/time required')
        if any(not math.isfinite(x) or x<=0 for x in nums):raise ValueError('positive finite inputs required')
        if any(type(x) is not int for x in (self.decode_slots,self.prefill_gpus,self.decode_gpus)):raise ValueError('counts must be integers')
        if not 0<=self.target_margin<1 or self.tax<0 or self.other_usd<0 or self.state_transfer_seconds<0:raise ValueError('invalid cost/time')
    @property
    def hourly(self):return (self.prefill_gpus+self.decode_gpus)*self.gpu_hour*self.uplift
    @property
    def net(self):return self.gross_usd/(1+self.tax)
    @property
    def allowance(self):return self.net*(1-self.target_margin)-self.other_usd
    @property
    def min_users_for_margin(self):return math.ceil(self.hourly*MONTH/3600/self.allowance) if self.allowance>0 else None
    def within_window_capacity(self,active_hours):
        # Within-arrival-window throughput, NOT full-day stability or a tail guarantee.
        per_user_requests_per_day=REQUESTS_PER_USER/DAYS
        decode=self.decode_slots*self.stream_tps/(VISIBLE+HIDDEN)
        prefill=self.prefill_tps/INPUT
        return min(prefill,decode)*active_hours*3600/per_user_requests_per_day

def arrivals(users,active_hours=24):
    """Exactly2000 requests/user, evenly spaced across30days; all users same window.
    Deterministic phase offsets smooth within each window: optimistic, not measured.
    """
    if type(users) is not int or users<1 or not 0<active_hours<=24:raise ValueError('bad arrivals')
    for day in range(DAYS):
        n=((day+1)*REQUESTS_PER_USER)//DAYS-(day*REQUESTS_PER_USER)//DAYS
        count=n*users
        for i in range(count):yield day*DAY+(i+.5)*active_hours*3600/count

def simulate(users,active_hours,p=Pool()):
    prefill_free=0.
    slots=[0.]*p.decode_slots
    first_visible=[]; completion=[]; wait=[]
    total=0
    for arrival in arrivals(users,active_hours):
        slot_free=heapq.heappop(slots)
        # Just-in-time admission: no unbounded queue of prefetched states.
        # Fixed durations permit prefill to complete as earliest decode slot frees.
        pf_start=max(arrival,prefill_free,slot_free-INPUT/p.prefill_tps-p.state_transfer_seconds)
        prefill_free=pf_start+INPUT/p.prefill_tps
        ready=prefill_free+p.state_transfer_seconds
        start=max(ready,slot_free)
        finish=start+(HIDDEN+VISIBLE)/p.stream_tps
        heapq.heappush(slots,finish)
        first_visible.append(start+(HIDDEN+1)/p.stream_tps-arrival)
        completion.append(finish-arrival)
        wait.append((pf_start-arrival)+(start-ready))
        total+=1
    first_visible.sort();completion.sort();wait.sort()
    def q(xs,pct):return xs[max(0,math.ceil(len(xs)*pct)-1)]
    end=max(slots)
    billed_seconds=max(MONTH,end) # pay to drain all work, no dropped requests
    serving=p.hourly*billed_seconds/3600/users
    cost=serving+p.other_usd
    return {'users':users,'active_hours':active_hours,'requests':total,'input_tokens_per_user':INPUT*total/users,'visible_tokens_per_user':VISIBLE*total/users,'hidden_tokens_per_user':HIDDEN*total/users,'p95_first_visible_seconds':q(first_visible,.95),'p95_completion_seconds':q(completion,.95),'p95_queue_seconds':q(wait,.95),'fraction_first_visible_slo':sum(t<=p.first_visible_slo_seconds for t in first_visible)/total,'decode_calendar_utilization':total*(HIDDEN+VISIBLE)/p.stream_tps/(p.decode_slots*MONTH),'prefill_calendar_utilization':total*INPUT/p.prefill_tps/MONTH,'drain_after_month_seconds':max(0,end-MONTH),'serving_cost_per_user_usd':serving,'all_in_scenario_cost_usd':cost,'scenario_margin':(p.net-cost)/p.net,'latency_gate':q(first_visible,.95)<=p.first_visible_slo_seconds,'margin_gate':cost<=p.net*(1-p.target_margin),'note':'hypothetical operating point, fixed reserve, no model capability measured'}

def main():
    out=Path(__file__).parent/'outputs';out.mkdir(exist_ok=True)
    pool=Pool();minimum=pool.min_users_for_margin
    rows=[simulate(n,h,pool) for n in (400,minimum,1200) for h in (24,8)]
    with (out/'queues.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    # GBP scenarios use explicit old v0 scenarioFX only, NOT today's exchange rate.
    budgets=[]
    for label,usd in [('USD500',500),('GBP200 at assumed0.80GBPperUSD',250),('GBP20 at assumed0.80GBPperUSD',25)]:
        p=Pool(gross_usd=usd)
        budgets.append({'budget':label,'usd_scenario':usd,'min_users_for30pct_margin':p.min_users_for_margin,'capacity_users_24h':p.within_window_capacity(24),'capacity_users_8h':p.within_window_capacity(8),'other_reserve':p.other_usd})
    summary={'classification':'analytical simulation; no measured model performance or quality','pool':asdict(pool),'monthly_pool_cost_usd':pool.hourly*MONTH/3600,'minimum_users':minimum,'capacity_24h':pool.within_window_capacity(24),'capacity_8h':pool.within_window_capacity(8),'rows':rows,'stretch':budgets}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
