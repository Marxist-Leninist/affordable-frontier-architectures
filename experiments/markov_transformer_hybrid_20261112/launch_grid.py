#!/usr/bin/env python3
"""launch_grid.py JOBS.json SLOTS [AFTER_STATUS.json] -- run the jobs of a list, SLOTS at a time, each single-threaded
at nice 19. With a third argument it first waits until that other list's status file says FINISHED.

Skips a job whose result file already says done. Holds new starts while the machine has under 2.5 GB of memory
available. `touch STOP` in this folder ends everything within seconds (each run also checks it every 20 steps).
Progress is in grid_status_<list>.json.
"""
import json
import os
import subprocess
import sys
import time


def mem_available_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) // 1024
    return 10 ** 6


def main():
    jobs = json.load(open(sys.argv[1]))
    slots = int(sys.argv[2])
    tag = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    os.makedirs('runs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MALLOC_ARENA_MAX='2')
    pending, running, done, failed = list(jobs), {}, [], []
    after = sys.argv[3] if len(sys.argv) > 3 else None       # start only once this other list's status file says finished
    while after:
        if os.path.exists('STOP'):
            return
        try:
            if json.load(open(after)).get('state', '').startswith(('FINISHED', 'STOPPED')):
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    t0 = time.time()

    def status(state):
        json.dump({'list': tag, 'state': state, 'total': len(jobs), 'done': len(done), 'failed': failed,
                   'running': [j['name'] for _, j in running.values()], 'pending': len(pending),
                   'elapsed_min': round((time.time() - t0) / 60, 1), 'mem_available_mb': mem_available_mb(),
                   'updated': time.strftime('%H:%M:%SZ', time.gmtime())}, open(f'grid_status_{tag}.json', 'w'))

    while pending or running:
        if os.path.exists('STOP'):
            for p, _ in running.values():
                p.terminate()
            status('STOPPED')
            return
        while pending and len(running) < slots and mem_available_mb() > 2500:
            j = pending.pop(0)
            out = f"runs/{j['name']}.json"
            try:
                if json.load(open(out)).get('status') == 'done':
                    done.append(j['name'])
                    continue
            except Exception:  # noqa: BLE001
                pass
            log = open(f"logs/{j['name']}.log", 'w')
            p = subprocess.Popen(['nice', '-n', '19', sys.executable, 'train.py', *[str(x) for x in j['args']], '--out', out],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            running[p.pid] = (p, j)
        time.sleep(3)
        for pid in list(running):
            p, j = running[pid]
            rc = p.poll()
            if rc is None:
                continue
            del running[pid]
            (done if rc == 0 else failed).append(j['name'])
        status('RUNNING')
    status('FINISHED' if not failed else 'FINISHED_WITH_FAILURES')


if __name__ == '__main__':
    main()
