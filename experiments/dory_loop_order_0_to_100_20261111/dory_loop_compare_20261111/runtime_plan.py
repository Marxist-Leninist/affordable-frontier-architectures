"""Pure runtime-only cohort selection; no ML or dataset imports."""
import math
COUNTS=(0,1,2,8,32,100)
def predict(a,b,n):
    return (n/4)*(101*a+5050*b)+32*(7*a+147*b)

def choose(samples,job_ids,elapsed,calibration_elapsed):
    if len(job_ids)!=6 or len(set(job_ids))!=6:raise ValueError('Six distinct checkpoint identities required')
    if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid elapsed time')
    if not math.isfinite(calibration_elapsed) or not 0<=calibration_elapsed<=180:raise ValueError('Calibration cap exceeded')
    if elapsed<calibration_elapsed:raise ValueError('Total elapsed cannot exclude calibration')
    envelopes={}
    for job in job_ids:
        rows=[s for s in samples if s['job']==job]
        expected={(r,k) for r in COUNTS for k in range(2)}
        if len(rows)!=12 or {(s['loops'],s['repeat']) for s in rows}!=expected:raise ValueError('Incomplete calibration')
        if any(s['status']!='completed' or not math.isfinite(s['seconds']) or s['seconds']<=0 for s in rows):raise ValueError('Invalid/nonfinite calibration')
        a=max(s['seconds'] for s in rows if s['loops']==0)
        b=max(max(0.,s['seconds']-a)/s['loops'] for s in rows if s['loops']>0)
        envelopes[job]={'a':a,'b':b}
    if len(samples)!=72:raise ValueError('Unexpected calibration rows')
    candidates=[]
    for n in (32,16,8):
        times={j:predict(e['a'],e['b'],n) for j,e in envelopes.items()}
        global_bound=elapsed+600+1.5*sum(times.values())
        ok=global_bound<=7200 and all(1.5*t<=1050 for t in times.values())
        candidates.append({'windows_per_domain':n,'predicted_seconds_per_checkpoint':times,'global_bound_seconds':global_bound,'fits':ok})
    chosen=next((c['windows_per_domain'] for c in candidates if c['fits']),None)
    return dict(status='selected' if chosen else 'blocked_no_fit',windows_per_domain=chosen,envelopes=envelopes,candidates=candidates,elapsed_debited_seconds=elapsed,calibration_elapsed_seconds=calibration_elapsed)

def validate_protocol(cfg):
    """Reject drift in assumptions used by the closed-form batch-cost rule."""
    expected={'counts':list(range(101)),'selected_counts':[0,1,2,4,8,32,100],
              'selected_windows_per_domain':128,'batch':8,'domains':['test','fineweb'],
              'global_wall_cap_seconds':7200,'per_checkpoint_wall_cap_seconds':1200,
              'seeds':[1,2,3],'topologies':['split','whole']}
    for key,value in expected.items():
        if cfg.get(key)!=value:raise ValueError('Runtime formula protocol mismatch: '+key)
    rule=cfg['runtime_only_cohort_rule']
    constants={'candidates':[32,16,8],'synthetic_counts':list(COUNTS),'repetitions':2,
               'warmup_loops':2,'batch':8,'calibration_cap_seconds':180,
               'global_reserve_seconds':600,'prediction_margin':1.5,
               'per_checkpoint_prediction_limit_seconds':1050}
    for key,value in constants.items():
        if rule.get(key)!=value:raise ValueError('Runtime rule mismatch: '+key)
