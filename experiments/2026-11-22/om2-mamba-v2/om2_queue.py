#!/usr/bin/env python3
"""Run the OM2 job list with N single-thread workers. Kill switches: STOP (no new jobs), PAUSE."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
jobs = json.loads((ROOT / 'jobs.json').read_text())
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
running, pending = {}, list(jobs)


def log(obj):
    with open(ROOT / 'queue.log', 'a') as f:
        f.write(json.dumps(dict(obj, t=time.strftime('%Y-%m-%d %H:%M:%S'))) + '\n')


log(dict(event='start', workers=N, jobs=len(jobs)))
while pending or running:
    if (ROOT / 'STOP').exists() and pending:
        log(dict(event='stop', dropped=pending))
        pending = []
    while pending and len(running) < N and not (ROOT / 'PAUSE').exists():
        j = pending.pop(0)
        d = ROOT / 'runs' / f"{j['arm']}-{j['seed']}"
        if (d / 'result.json').exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        fh = open(d / 'log.txt', 'a')
        cmd = [sys.executable, '-u', str(ROOT / 'om2bench.py'), 'run', '--arm', j['arm'], '--seed', str(j['seed']),
               '--out', str(d)] + [str(x) for x in j.get('extra', [])]
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        running[p.pid] = (p, j, time.time(), fh)
        log(dict(event='launch', job=j, pid=p.pid))
    for pid, (p, j, t0, fh) in list(running.items()):
        if p.poll() is not None:
            fh.close()
            del running[pid]
            log(dict(event='exit', job=j, rc=p.returncode, wall_s=round(time.time() - t0, 1)))
    (ROOT / 'queue_status.json').write_text(json.dumps(dict(
        running=[dict(r[1], pid=pid, wall_s=round(time.time() - r[2])) for pid, r in running.items()],
        pending=pending, t=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1))
    time.sleep(5)
log(dict(event='done'))
