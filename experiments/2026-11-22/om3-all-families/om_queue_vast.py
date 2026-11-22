#!/usr/bin/env python3
"""OM job queue for a shared Vast box: N single-thread workers at nice 19, with a memory guard.

Launches only while MemAvailable >= MIN_LAUNCH_GB; if it drops below KILL_GB the newest worker is stopped and requeued,
so the box's own services keep their memory. Kill switches in this folder: STOP (no new jobs), PAUSE (hold launches).
Usage: om_queue_vast.py WORKERS JOBS_JSON
"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
N = int(sys.argv[1])
jobs = json.loads((ROOT / sys.argv[2]).read_text())
MIN_LAUNCH_GB, KILL_GB = float(os.environ.get('OM_MIN_LAUNCH_GB', '10')), float(os.environ.get('OM_KILL_GB', '5'))
os.nice(19)
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='0',
           PYTHONDONTWRITEBYTECODE='1')


def avail_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1048576
    return 0.0


def log(obj):
    with open(ROOT / 'queue.log', 'a') as f:
        f.write(json.dumps(dict(obj, t=time.strftime('%Y-%m-%d %H:%M:%S'), avail_gb=round(avail_gb(), 1))) + '\n')


running, pending = {}, list(jobs)
log(dict(event='start', workers=N, jobs=len(jobs)))
while pending or running:
    if (ROOT / 'STOP').exists() and pending:
        log(dict(event='stop', dropped=pending))
        pending = []
    while (pending and len(running) < N and not (ROOT / 'PAUSE').exists() and avail_gb() >= MIN_LAUNCH_GB):
        j = pending.pop(0)
        d = ROOT / 'runs' / f"{j['arm']}-{j['seed']}"
        if (d / 'result.json').exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        fh = open(d / 'log.txt', 'a')
        cmd = [sys.executable, '-u', str(ROOT / j.get('script', 'om2bench.py')), 'run', '--arm', j['arm'],
               '--seed', str(j['seed']), '--out', str(d)] + [str(x) for x in j.get('extra', [])]
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT), start_new_session=True)
        running[p.pid] = (p, j, time.time(), fh)
        log(dict(event='launch', job=j, pid=p.pid))
        time.sleep(3)
    if running and avail_gb() < KILL_GB:
        pid = max(running, key=lambda k: running[k][2])
        p, j, t0, fh = running.pop(pid)
        os.killpg(pid, signal.SIGTERM)
        p.wait()
        fh.close()
        pending.append(j)
        log(dict(event='memory_guard_requeue', job=j, pid=pid))
    for pid, (p, j, t0, fh) in list(running.items()):
        if p.poll() is not None:
            fh.close()
            del running[pid]
            log(dict(event='exit', job=j, rc=p.returncode, wall_s=round(time.time() - t0, 1)))
    (ROOT / 'queue_status.json').write_text(json.dumps(dict(
        running=[dict(r[1], pid=pid, wall_s=round(time.time() - r[2])) for pid, r in running.items()],
        pending=pending, avail_gb=round(avail_gb(), 1), t=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1))
    time.sleep(5)
log(dict(event='done'))
