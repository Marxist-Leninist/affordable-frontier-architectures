"""Prospective runtime-only amendment. Standard library; no model execution."""
import argparse,hashlib,json,math
from pathlib import Path
from runtime_plan import choose as original_choose
NAMES=('calibration_raw.json','cohort_selection.json','blocker.json')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def predict(a,b,n=8):return (n/4)*(101*a+5050*b)+16*(3*a+102*b)
def verify_calibration(directory,receipt_manifest,original_source_sha):
    if receipt_manifest['original_source_manifest_sha256']!=original_source_sha:raise ValueError('Original source manifest changed')
    for name in NAMES:
        if sha(directory/name)!=receipt_manifest['receipts_sha256'][name]:raise ValueError('Calibration receipt changed: '+name)
    raw,selection,blocker=[read(directory/name) for name in NAMES]
    if blocker.get('status')!='blocked_no_fit' or blocker.get('heldout_loaded') is not False:raise ValueError('Original attempt must have stopped before heldout loading')
    if raw.get('synthetic_input')!='arange(8*96) modulo256' or raw.get('synthetic_target')!='arange(8*32) modulo256' or raw.get('batch')!=8 or raw.get('counts')!=[0,1,2,8,32,100] or raw.get('repetitions')!=2 or raw.get('warmup_loop_count')!=2:raise ValueError('Calibration input provenance mismatch')
    ids=[f'{k}_{s}' for s in (1,2,3) for k in ('split','whole')]
    if len(raw['warmups'])!=6 or {w['job'] for w in raw['warmups']}!=set(ids) or any(w['status']!='completed' for w in raw['warmups']):raise ValueError('Incomplete original warmups')
    if set(selection['checkpoint_hashes'])!=set(ids):raise ValueError('Incomplete checkpoint identities')
    rebuilt=original_choose(raw['samples'],ids,selection['elapsed_debited_seconds'],selection['calibration_elapsed_seconds'])
    for key in ('status','windows_per_domain','envelopes','candidates'):
        if rebuilt[key]!=selection[key]:raise ValueError('Original decision does not match synthetic timings: '+key)
    if rebuilt['status']!='blocked_no_fit':raise ValueError('Amendment applies only to original no-fit result')
    return selection

def plan(selection,new_setup_elapsed):
    if not math.isfinite(new_setup_elapsed) or new_setup_elapsed<0:raise ValueError('Invalid new setup debit')
    prior=selection['elapsed_debited_seconds']
    if not math.isfinite(prior) or prior<0:raise ValueError('Invalid original elapsed debit')
    times={j:predict(v['a'],v['b']) for j,v in selection['envelopes'].items()}
    bound=prior+new_setup_elapsed+600+1.5*sum(times.values())
    fits=bound<=7200 and all(1.5*t<=1050 for t in times.values())
    return dict(status='selected' if fits else 'blocked_no_fit',windows_per_domain=8,selected_windows_per_domain=64,selected_counts=[0,2,100],predicted_seconds_per_checkpoint=times,global_bound_seconds=bound,prior_elapsed_debited_seconds=prior,new_setup_debited_seconds=new_setup_elapsed,checkpoint_hashes=selection['checkpoint_hashes'],envelopes=selection['envelopes'])

def validate_protocol(cfg):
    expected={'counts':list(range(101)),'selected_counts':[0,2,100],'cohort_windows_per_domain':8,'selected_windows_per_domain':64,'batch':8,'domains':['test','fineweb'],'global_wall_cap_seconds':7200,'per_checkpoint_wall_cap_seconds':1200,'seeds':[1,2,3],'topologies':['split','whole'],'margin':1.5,'global_reserve_seconds':600,'per_checkpoint_prediction_limit_seconds':1050}
    for k,v in expected.items():
        if cfg.get(k)!=v:raise ValueError('Amended protocol mismatch: '+k)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-receipts',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=Path(__file__).resolve().parent
    if a.output.exists():raise RuntimeError('Refusing receipt manifest overwrite')
    manifest={'original_source_manifest_sha256':sha(root/'source_hashes.json'),'receipts_sha256':{n:sha(a.prepare_receipts/n) for n in NAMES}}
    verify_calibration(a.prepare_receipts,manifest,sha(root/'source_hashes.json'))
    a.output.write_text(json.dumps(manifest,indent=2)+'\n')
