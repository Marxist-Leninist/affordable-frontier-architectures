"""Readiness, exclusive ownership, and bounded own-child cleanup."""
import fcntl,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STOP=False
CHILD=None

def ready():
    nat=ROOT.parent/'nat_long_20261111'
    assert json.loads((nat/'pipeline_status.json').read_text())['status']=='training_complete','NAT training incomplete'
    assert json.loads((nat/'postprocess_status.json').read_text())['status']=='completed','NAT postprocessing incomplete'
    for name in ['evaluation_results.json','standalone_timings.json','summary_long.json']:
        assert (nat/name).is_file(),f'Missing NAT receipt: {name}'

def ownership():
    ready()
    handle=(ROOT/'.execution.lock').open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return handle

def stop(signum,frame):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.terminate()

def install():
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def run_owned(cmd,timeout):
    global CHILD
    if STOP:raise RuntimeError('controller cancelled')
    deadline=time.monotonic()+timeout
    CHILD=subprocess.Popen(cmd,cwd=ROOT)
    try:
        while CHILD.poll() is None:
            if STOP or time.monotonic()>=deadline:
                CHILD.terminate()
                try:CHILD.wait(timeout=10)
                except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait(timeout=10)
                raise RuntimeError('owned child cancelled or timed out')
            try:CHILD.wait(timeout=.2)
            except subprocess.TimeoutExpired:pass
        if STOP:raise RuntimeError('controller cancelled')
        if CHILD.returncode:raise RuntimeError(f'child exited {CHILD.returncode}')
    finally:
        if CHILD is not None and CHILD.poll() is None:
            CHILD.terminate()
            try:CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait(timeout=10)
        CHILD=None
