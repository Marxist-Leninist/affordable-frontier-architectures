"""Sequential training, then all-fit-gated evaluation. Never overlaps NAT."""
import argparse,json,time,sys,hashlib
from pathlib import Path
import control
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--go-after-nat-complete',action='store_true');a=p.parse_args()
    if not a.go_after_nat_complete:p.error('Explicit parent go-ahead required')
    lock=control.ownership();control.install()
    for name,digest in json.loads((ROOT/'source_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    cfg=json.loads((ROOT/'protocol.json').read_text());deadline=time.monotonic()+cfg['global_wall_cap_seconds']
    jobs=[(kind,seed) for seed in cfg['seeds'] for kind in (cfg['topologies'] if seed%2 else cfg['topologies'][::-1])]
    if any((ROOT/'runs'/f'{kind}_s{seed}').exists() for kind,seed in jobs):raise RuntimeError('Refusing duplicate or partial existing runs; inspect first')
    control.run_owned([sys.executable,str(ROOT/'test_correctness.py')],min(30,deadline-time.monotonic()))
    for phase in ['training','evaluation']:
        for kind,seed in jobs:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise RuntimeError('Global wall cap reached')
            cmd=[sys.executable,str(ROOT/'train.py'),'--go-after-nat-complete','--kind',kind,'--seed',str(seed)]
            if phase=='evaluation':cmd.append('--evaluate-only')
            control.run_owned(cmd,min(remaining,cfg['run_wall_cap_seconds']+15))
            file='training.json' if phase=='training' else 'result.json'
            r=json.loads((ROOT/'runs'/f'{kind}_s{seed}'/file).read_text())
            assert r['status']==('training_complete' if phase=='training' else 'completed'),f'{phase} incomplete: {kind} seed{seed}'
    print('All six training and evaluation runs completed.')
if __name__=='__main__':main()
