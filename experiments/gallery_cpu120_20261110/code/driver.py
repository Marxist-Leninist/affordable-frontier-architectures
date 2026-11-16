"""Run a queue of jobs, one single-threaded process per CPU core, each inside its own no-network namespace.

Usage: driver.py QUEUE.json OUT_DIR GALLERY AUDIT DATA PYTHON [cores, default "0,1"]
QUEUE.json is a list of {"arm": "transformer" | <gallery index>, "seed": int}. Finished jobs (a result JSON
exists) are skipped, so the driver can be stopped and started again. A file named STOP in OUT_DIR ends the
run after the jobs in flight.
"""
import json
import pathlib
import subprocess
import sys
import time

queue_path, out_dir, gallery, audit, data, python = sys.argv[1:7]
cores = [int(c) for c in (sys.argv[7] if len(sys.argv) > 7 else "0,1").split(",")]
out = pathlib.Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
here = pathlib.Path(__file__).resolve().parent
queue = json.loads(pathlib.Path(queue_path).read_text())
log = open(out / "driver.log", "a", buffering=1)


def tag(job):
    return f"{'T96' if job['arm'] == 'transformer' else 'g' + str(job['arm'])}_s{job['seed']}"


def launch(job, core):
    # Jobs inherit this process's environment as it is. oneDNN's choice of instruction set is left at its
    # default, so BF16 products use the fastest kernels the processor has (AMX where present); each result
    # records the ONEDNN_MAX_CPU_ISA it saw. run_job.py pins itself to one thread.
    cmd = ["taskset", "-c", str(core), "unshare", "-rn", "--", python, "-I", str(here / "run_job.py"),
           "--arm", str(job["arm"]), "--seed", str(job["seed"]), "--gallery", gallery, "--audit", audit,
           "--data", data, "--out", str(out), "--budget", str(job.get("budget", 120)), "--marks", "60"]
    if job.get("save_weights"):
        cmd.append("--save-weights")
    err = open(out / f"{tag(job)}.stderr", "w")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err), time.time()


pending = [j for j in queue if not (out / f"{tag(j)}.json").exists() and not (out / f"{tag(j)}.failed").exists()]
running = {}
log.write(f"{time.strftime('%H:%M:%S')} start pending={len(pending)} of {len(queue)} cores={cores}\n")
while pending or running:
    for core in cores:
        if core not in running and pending and not (out / "STOP").exists():
            job = pending.pop(0)
            running[core] = (job, *launch(job, core))
    time.sleep(2)
    for core, (job, proc, t0) in list(running.items()):
        rc = proc.poll()
        if rc is None and time.time() - t0 > 1500:
            proc.kill()
            rc = -9
        if rc is not None:
            del running[core]
            ok = rc == 0 and (out / f"{tag(job)}.json").exists()
            if not ok:
                (out / f"{tag(job)}.failed").write_text(f"exit {rc}\n")
            log.write(f"{time.strftime('%H:%M:%S')} {tag(job)} {'ok' if ok else 'FAILED rc=' + str(rc)} {time.time() - t0:.0f}s left={len(pending)}\n")
    if (out / "STOP").exists() and not running:
        break
log.write(f"{time.strftime('%H:%M:%S')} driver exit pending={len(pending)}\n")
