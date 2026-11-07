"""Durable per-method launcher after first long process received exit137."""
import sys,resource,json
from run import *
resource.setrlimit(resource.RLIMIT_CPU,(180,185));verify();(P/'checkpoints').mkdir(exist_ok=True)
kind=sys.argv[1];seed=int(sys.argv[2]);budget=None if kind=='bptt' else json.loads((P/f'bptt_{seed}_equal_data.json').read_text())['training_cpu_seconds']
rows,cpu=train(kind,seed,source_data(),budget)
write(f'completed_{kind}_{seed}.json',dict(rows=rows,cpu=cpu))
