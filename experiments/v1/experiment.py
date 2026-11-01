"""Architecture-neutral necessary-condition analysis; not a throughput predictor."""
from dataclasses import dataclass, asdict
from math import ceil, floor, inf
from itertools import product
import csv,json
from pathlib import Path

INPUT=10_000_000
OUTPUT=4_000_000  # 2M visible + 2M hidden, never shortened
@dataclass(frozen=True)
class Economy:
    price:float=500
    tax:float=.20
    margin:float=.30
    other:float=50  # placeholder reserve, not verified full non-serving expenses
    gpus:int=32
    gpu_hour:float=3.99
    uplift:float=1.2
    utilization:float=.60
    prefill_tps:float=8000  # required measured prefill operating point, not prediction
    cache_hit:float=0
    cached_work:float=.1
    def __post_init__(self):
        if min(self.price,self.gpus,self.gpu_hour,self.uplift,self.prefill_tps)<=0 or self.other<0 or self.tax<0: raise ValueError('invalid positive parameter')
        if not 0<self.utilization<=1 or not 0<=self.margin<1 or not 0<=self.cache_hit<=1 or not 0<=self.cached_work<=1: raise ValueError('invalid fraction')
    @property
    def hourly(self):return self.gpus*self.gpu_hour*self.uplift
    @property
    def allowance(self):return self.price/(1+self.tax)*(1-self.margin)-self.other
    @property
    def input_work(self):return INPUT*((1-self.cache_hit)+self.cache_hit*self.cached_work)
    def cost(self,decode):
        if decode<=0:raise ValueError('decode must be positive')
        return self.hourly/(3600*self.utilization)*(self.input_work/self.prefill_tps+OUTPUT/decode)+self.other
    def required_decode(self):
        remaining=self.allowance*3600*self.utilization/self.hourly-self.input_work/self.prefill_tps
        return OUTPUT/remaining if remaining>0 else None
    def record(self):
        t=self.required_decode()
        return {**asdict(self),'serving_allowance_usd':self.allowance,'required_decode_tps':t,'required_batch_at_exactly20tps':ceil(t/20) if t else None,'cost_at1500tps':self.cost(1500)}

@dataclass(frozen=True)
class Architecture:
    name:str
    active:float
    attention_fraction:float
    recurrent_state_gb:float=0
    bits:float=4
    total:float=3e12
    def __post_init__(self):
        if not 0<self.active<=self.total or not 0<=self.attention_fraction<=1 or self.recurrent_state_gb<0 or self.bits<=0:raise ValueError('invalid architecture')
    def sequence_bytes(self,context):return context*120*576*2*self.attention_fraction+self.recurrent_state_gb*1e9
    def gate(self,context,batch,gpus=32,overlap_exponent=1,bw_eff=.5,compute_eff=.35):
        if context<=0 or batch<=0 or not 0<=overlap_exponent<=1 or not 0<bw_eff<=1 or not 0<compute_eff<=1:raise ValueError('invalid gate')
        weights=self.total*self.bits/8*1.15
        available=gpus*80e9-weights-gpus*8e9  # workspace assumption
        state=self.sequence_bytes(context)
        capacity=max(0,floor(available/state)) if state else None
        unique=min(self.total,self.active*batch**overlap_exponent)
        weight_read=unique*self.bits/8*1.15
        # Attention cache read once; recurrent state read+write once per decode step.
        state_traffic=batch*(context*120*576*2*self.attention_fraction+2*self.recurrent_state_gb*1e9)
        bandwidth_seconds=(weight_read+state_traffic)/(gpus*3.35e12*bw_eff)
        # Non-sparse BF16 peak 989.5 TF/s/card; NOT advertised 2:4-sparse number.
        # Excludes attention/SSM ops: optimistic parameter matmul-only lower bound.
        compute_seconds=2*self.active*batch/(gpus*989.5e12*compute_eff)
        floor_seconds=max(bandwidth_seconds,compute_seconds)
        return {'name':self.name,'active_B':self.active/1e9,'attention_fraction':self.attention_fraction,'state_GB':self.recurrent_state_gb,'bits':self.bits,'context':context,'batch':batch,'overlap_exponent':overlap_exponent,'memory_capacity_sequences':capacity,'memory_gate':capacity is None or batch<=capacity,'step_time_lower_bound_ms':1000*floor_seconds,'roofline_upper_bound_tps':batch/floor_seconds,'20tps_latency_gate':floor_seconds<=.05,'note':'passing is necessary only; omitted communication, attention/SSM ops and scheduling'}

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    economics=[Economy(tax=tax,margin=m,other=o,gpu_hour=p,utilization=u,cache_hit=c,prefill_tps=i).record() for tax,m,o,p,u,c,i in product((0,.2),(0,.3,.5,.7),(50,150),(2,3.99,6),(.3,.6,.8),(0,.5),(4000,8000,16000))]
    arches=[Architecture('dense_attention',3e12,1),Architecture('sparse_attention',150e9,1),Architecture('sparse_hybrid_1of8',150e9,.125,1),Architecture('sparse_recurrent',150e9,0,1)]
    resources=[a.gate(c,b,overlap_exponent=r) for a,c,b,r in product(arches,(8192,32768,131072),(16,64,128),(0,.5,1))]
    active_sensitivity=[Architecture('sparse_active_sensitivity',a, .125,s,bits=bits).gate(32768,128,overlap_exponent=r) for a,s,bits,r in product((75e9,150e9,300e9),(1,4),(4,8),(0,1))]
    for filename,rows in [('economics.csv',economics),('resource_gates.csv',resources),('active_state_precision.csv',active_sensitivity)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary={'warning':'Analytic thresholds and necessary gates only; NO measured model quality or throughput','fixed_workload':{'input':INPUT,'visible_output':2_000_000,'hidden_output':2_000_000},'baseline':Economy().record(),'margin_sensitivity':[Economy(margin=m).record() for m in (0,.3,.5,.7)],'baseline_resources_at128k_batch128':[a.gate(131072,128) for a in arches],'row_counts':{'economics':len(economics),'resources':len(resources),'active_state_precision':len(active_sensitivity)}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False))
if __name__=='__main__':run(Path(__file__).parent/'outputs')
