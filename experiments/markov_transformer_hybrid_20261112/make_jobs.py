#!/usr/bin/env python3
"""make_jobs.py -- write a job list for launch_grid.py.

  python3 make_jobs.py --tag main --data tinystories,wikitext --variants base,mc1_poe,... --seeds 1,2,3 --lrs 2e-3 \
      --steps 2000 [--bs 16 --d 128 --L 4 --H 4 --T 128]
Seeds are the outer loop, so every variant has its first seed before any variant gets its second.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--variants', required=True)
    ap.add_argument('--seeds', default='1')
    ap.add_argument('--lrs', default='2e-3')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--T', type=int, default=128)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--L', type=int, default=4)
    ap.add_argument('--H', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--list', default=None, help='file name jobs_<list>.json (default: the tag)')
    ap.add_argument('--append', action='store_true', help='add to an existing list instead of replacing it')
    a = ap.parse_args()
    a.list = a.list or a.tag
    jobs = []
    if a.append:
        try:
            jobs = json.load(open(f'jobs_{a.list}.json'))
        except FileNotFoundError:
            pass
    for seed in a.seeds.split(','):
        for data in a.data.split(','):
            for lr in a.lrs.split(','):
                for v in a.variants.split(','):
                    name = f'{a.tag}_{data}_{v}_lr{lr}_s{seed}'
                    jobs.append({'name': name, 'args': ['--data', f'data/{data}', '--variant', v, '--seed', seed, '--lr', lr,
                                                        '--steps', a.steps, '--bs', a.bs, '--T', a.T, '--d', a.d, '--L', a.L,
                                                        '--H', a.H, '--warmup', a.warmup]})
    json.dump(jobs, open(f'jobs_{a.list}.json', 'w'), indent=1)
    print(len(jobs), 'jobs ->', f'jobs_{a.list}.json')


if __name__ == '__main__':
    main()
