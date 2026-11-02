"""Conditional speculation envelope; measured acceptance/timings must replace inputs."""
import math,json,csv,itertools
from pathlib import Path
from speculative import probe

def expected_tokens(alpha,k):
    if not math.isfinite(alpha) or not 0<=alpha<=1 or type(k) is not int or k<1:raise ValueError('invalid acceptance/count')
    return sum(alpha**i for i in range(k+1))

def point(alpha,k,draft_seconds,beta,target_seconds=.05,overhead_seconds=.001,draft_gpus=8):
    values=(draft_seconds,beta,target_seconds,overhead_seconds)
    if any(not math.isfinite(x) or x<0 for x in values) or target_seconds<=0 or type(draft_gpus) is not int or draft_gpus<0:raise ValueError('invalid timing/resource')
    emitted=expected_tokens(alpha,k)
    verify=target_seconds*(1+beta*k)
    duration=k*draft_seconds+verify+overhead_seconds
    speed=target_seconds*emitted/duration
    total_gpus=64+draft_gpus
    hourly=total_gpus*3.99*1.2
    # Same 30-day,10M input +2M hidden+2M visible,20%tax,$50reserve,30%margin.
    allowance=(500/1.2)*.7-50
    min_users=math.ceil(hourly*720/allowance)
    # ONLY an optimistic aggregate scaling gate: no actual batch speedup measured.
    decode_capacity8h=432*speed
    prefill_capacity8h=691.2
    return {'acceptance':alpha,'draft_block':k,'draft_token_seconds':draft_seconds,'verify_linear_increment':beta,'expected_tokens_per_iteration':emitted,'iteration_seconds':duration,'hypothetical_single_stream_speedup':speed,'target_scored_positions_per_emitted':(k+1)/emitted,'draft_proposed_per_emitted':k/emitted,'draft_gpus':draft_gpus,'monthly_pool_usd':hourly*720,'minimum_users_for30pct_margin':min_users,'optimistic_decode_capacity_users_8h':decode_capacity8h,'unchanged_prefill_capacity_users_8h':prefill_capacity8h,'optimistic_combined_capacity_users_8h':min(decode_capacity8h,prefill_capacity8h),'passes_8h_capacity_and_margin_necessary_gate':min(decode_capacity8h,prefill_capacity8h)>=min_users,'necessary_prefill_multiplier':min_users/prefill_capacity8h,'necessary_decode_multiplier':min_users/432,'note':'iid acceptance and hypothetical verify/draft latency; no measured batch scaling'}

def main():
    path=Path(__file__).parent/'outputs';path.mkdir(exist_ok=True)
    rows=[point(a,k,d,b) for a,k,d,b in itertools.product((.5,.7,.9,.98),(1,2,4,8,16),(.001,.005,.01),(.05,.2,1.))]
    with (path/'envelope.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for k in (1,4,8):(path/f'correctness_k{k}.json').write_text(json.dumps(probe(k=k),indent=2)+'\n')
    representative=[point(.9,k,.005,.2) for k in (1,2,4,8,16)]
    summary={'classification':'conditional analytic gates plus untrained categorical CPU correctness probe','baseline_commit':'55021eaabc8e43b6c19e1b3826b5225fc51a1a4a','fixed_workload_per_user':{'input':10000000,'visible':2000000,'hidden':2000000},'rows':len(rows),'rows_with_stream_speedup_gt1':sum(r['hypothetical_single_stream_speedup']>1 for r in rows),'rows_passing_unchanged_prefill_8h_gate':sum(r['passes_8h_capacity_and_margin_necessary_gate'] for r in rows),'representative':representative,'best_hypothetical_point':max(rows,key=lambda r:r['hypothetical_single_stream_speedup']),'zero_extra_gpu_prefill_gate':point(.98,16,.001,.05,draft_gpus=0),'limits':['No measured model quality or GPU throughput','Exact rejection algorithm preserves categorical target; does not improve target quality','A per-stream speedup cannot be automatically applied to batch throughput','No v2 arrival trace or token allowance reduced','No production changes or spending']}
    (path/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
