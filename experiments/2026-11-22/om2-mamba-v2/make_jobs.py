#!/usr/bin/env python3
"""Write jobs.json: longest runs first so the two slow om1 runs overlap the rest."""
import json
from pathlib import Path

SEEDS = (5617, 5649)
EXTRA = ['--steps', '2000', '--eval-every', '250', '--warmup', '100', '--batch', '8', '--ctx', '256']
order = ['om1'] + ['m2', 'om2g', 'om2a', 'om2p', 'om2', 'om2pos'] + ['tf']
jobs = []
jobs += [dict(arm='om1', seed=s, extra=EXTRA) for s in SEEDS]
for arm in order[1:]:
    for s in SEEDS:
        jobs.append(dict(arm=arm, seed=s, extra=EXTRA))
Path(__file__).with_name('jobs.json').write_text(json.dumps(jobs, indent=1) + '\n')
print(len(jobs), 'jobs')
