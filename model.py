"""Illustrative capacity economics, not measured 3T throughput. Python stdlib only."""
from dataclasses import dataclass, asdict
from math import ceil
import json

@dataclass(frozen=True)
class Serving:
    gpus: int = 32
    usd_gpu_hour: float = 3.99
    gbp_per_usd: float = 0.80  # scenario, NOT current FX quote
    infrastructure_multiplier: float = 1.20
    utilization: float = 0.60  # busy-equivalent capacity/calendar capacity
    input_tps: float = 8000  # assumed aggregate throughput, NOT benchmark
    output_tps: float = 1500  # assumed aggregate throughput, NOT benchmark
    hours_month: float = 730
    def __post_init__(self):
        if not 0 < self.utilization <= 1 or min(self.gpus, self.usd_gpu_hour, self.gbp_per_usd, self.infrastructure_multiplier, self.input_tps, self.output_tps, self.hours_month) <= 0:
            raise ValueError('positive inputs and 0 < utilization <= 1 required')
    @property
    def hourly(self):
        return self.gpus*self.usd_gpu_hour*self.gbp_per_usd*self.infrastructure_multiplier
    @property
    def rates(self):
        return tuple(self.hourly*1e6/(3600*self.utilization*t) for t in (self.input_tps,self.output_tps))
    def seconds(self, inputs, outputs):
        if min(inputs, outputs) < 0: raise ValueError('negative tokens')
        return inputs/self.input_tps + outputs/self.output_tps
    def cost(self, inputs, outputs):
        return self.hourly*self.seconds(inputs,outputs)/(3600*self.utilization)
    def subscribers_capacity(self, inputs, outputs):
        return self.hours_month*3600*self.utilization/self.seconds(inputs,outputs)

def budget(price, vat=.20, target_margin=.70, other_cost=1):
    """Desired margin of ex-VAT sales after serving + explicit other per-user costs."""
    if price < 0 or vat < 0 or not 0 <= target_margin < 1 or other_cost < 0: raise ValueError('bad budget')
    return max(0, price/(1+vat)*(1-target_margin)-other_cost)

def weights_bytes(params=3e12, bits=4, overhead=1):
    if min(params,bits,overhead) <= 0: raise ValueError('positive storage inputs required')
    return params*bits/8*overhead

def kv_bytes(context, layers=120, latent=512, rope=64, bytes_element=2):
    return context*layers*(latent+rope)*bytes_element

def metered_cost(serving, input_tokens, visible_output_tokens, hidden_output_tokens=0,
                 cache_hit_fraction=0, cached_input_work_fraction=1):
    """Cache discount is an assumed work fraction, not automatic or an API rate."""
    if not 0 <= cache_hit_fraction <= 1 or not 0 <= cached_input_work_fraction <= 1:
        raise ValueError('cache fractions must be in [0,1]')
    if min(input_tokens,visible_output_tokens,hidden_output_tokens) < 0:
        raise ValueError('negative tokens')
    effective_input=input_tokens*((1-cache_hit_fraction)+cache_hit_fraction*cached_input_work_fraction)
    return serving.cost(effective_input,visible_output_tokens+hidden_output_tokens)

def report():
    s=Serving()
    workloads=[('light',100_000,20_000),('regular',1_000_000,200_000),('heavy',10_000_000,2_000_000)]
    scenarios=[]
    for tps in (500,1500,3000):
        x=Serving(output_tps=tps)
        scenarios.append({'assumed_output_tps':tps,'gbp_per_million_input_output':x.rates,'workloads':[{'name':n,'inputs':i,'outputs':o,'monthly_gbp':x.cost(i,o),'capacity_subscribers':x.subscribers_capacity(i,o)} for n,i,o in workloads]})
    return {'warning':'Illustrative assumptions; no actual 3T benchmark or quality result', 'assumptions':asdict(s),'monthly_cluster_gbp':s.hourly*s.hours_month,'budgets':{str(p):{'serving_gbp':budget(p),'minimum_subscribers_to_cover_cluster_at_target_margin':ceil(s.hourly*s.hours_month/budget(p))} for p in (20,200,400)},'weight_storage_decimal_TB':{str(b):weights_bytes(bits=b)/1e12 for b in (16,8,4,1.58,1)},'kv_decimal_GB_per_sequence':{str(c):kv_bytes(c)/1e9 for c in (8192,32768,131072)},'scenarios':scenarios,'primary_USD500_at_assumed_FX':{'gross_GBP':400,'serving_budget_GBP':budget(400),'heavy_10M_input_2M_visible_no_hidden_cost':metered_cost(s,10e6,2e6),'heavy_plus_2M_hidden_cost':metered_cost(s,10e6,2e6,2e6),'heavy_plus_hidden_50percent_cache_10percent_cache_work_cost':metered_cost(s,10e6,2e6,2e6,.5,.1)}}

if __name__=='__main__': print(json.dumps(report(),indent=2))
