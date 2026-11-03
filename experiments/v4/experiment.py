"""Exact-prefix LRU trace accounting; synthetic workload, not an LM benchmark."""
from collections import OrderedDict
import json, math
from pathlib import Path
USERS=1028
REQUESTS=2000
PREFIX=4096
INPUT=5000
PREFILL_CAPACITY=691.2

def trace(users=USERS, requests=REQUESTS, burst=1):
    if users<1 or requests<1 or burst<1: raise ValueError('positive sizes')
    for start in range(0,requests,burst):
        for user in range(users):
            for j in range(start,min(start+burst,requests)):
                # Four reusable prefixes per tenant; global model revision each500/user.
                yield (user, j%4, j//500)

def replay(keys,capacity):
    if type(capacity) is not int or capacity<0: raise ValueError('nonnegative integer capacity')
    lru=OrderedDict(); hits=total=0
    for key in keys:
        total+=1
        if key in lru:
            hits+=1;lru.move_to_end(key)
        elif capacity:
            lru[key]=None
            if len(lru)>capacity:lru.popitem(last=False)
    cached=hits*PREFIX
    fraction=cached/(total*INPUT) if total else 0
    return dict(requests=total,hits=hits,computed_input_tokens=total*INPUT-cached,cached_input_fraction=fraction,prefill_capacity_users=PREFILL_CAPACITY/(1-fraction),prefill_capacity_gate=PREFILL_CAPACITY/(1-fraction)>=USERS,decode_required_speedup=USERS/432,all_in_feasible=False)

def main():
    rows=[]
    for burst in (1,20):
        for capacity in (0,64,256,1028,4112):
            r=replay(trace(burst=burst),capacity)
            r.update(burst=burst,capacity_prefixes=capacity,kv_cache_bytes=capacity*PREFIX*2*120*8*128*2,recurrent_checkpoint_bytes=capacity*64*1024**2,latent_cache_bytes=capacity*PREFIX*120*576*2)
            rows.append(r)
    summary=dict(classification='synthetic cache trace and necessary gates; no model quality or GPU timing measured',users=USERS,requests_per_user=REQUESTS,input_per_user=10_000_000,visible_per_user=2_000_000,hidden_per_user=2_000_000,required_cached_input_fraction=1-PREFILL_CAPACITY/USERS,monthly_pool_usd=248209.92,net_revenue_per_user=500/1.2,reserve_per_user=50,scenario_margin=(500/1.2-(248209.92/USERS+50))/(500/1.2),rows=rows)
    Path(__file__).with_name('results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
