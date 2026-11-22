#!/usr/bin/env python3
"""Same-machine speed and memory table for every arm: each arm in its own fresh process, one thread, rounds alternate
the arm order. Measures training seconds per step (batch 8 x 256, AdamW), training peak RSS (raw, and above the
process baseline measured after imports and data loading but before the model is built), prefill tokens/s (1 x 256),
decode ms per token (median and p95 over 192 tokens after a 64-byte prompt), and decode state bytes after 256 tokens:
every tensor the step() cache holds at batch 1, and the per-sequence part (batch-2 cache minus batch-1 cache), which
leaves out input-independent tables a decoder may keep in its cache (Hyena's filter taps).
Weights are untrained, since no arm's training or decode work depends on them, except the state-gated store, whose
decode skips the lookup when its gate is shut: set OM_TRAINED_<ARM>=path/to/weights.pt to time an arm's prefill and
decode with trained weights (used for om2g).
Usage: speed_all.py ARM[,ARM...] OUT_JSON [ROUNDS]"""
import json, os, resource, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def state_bytes(s):
    import torch
    if isinstance(s, torch.Tensor):
        return s.numel() * s.element_size()
    if isinstance(s, (list, tuple)):
        return sum(state_bytes(x) for x in s)
    if isinstance(s, dict):
        return sum(state_bytes(x) for x in s.values())
    return 0


def worker(arm):
    import numpy as np
    import torch
    from torch.nn import functional as F
    sys.path.insert(0, str(HERE))
    import om3bench  # noqa: F401  (patches om2bench.build)
    import om2bench as ob
    torch.set_num_threads(1)
    torch.manual_seed(1)
    train, dev, test, _ = ob.load_all()
    base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    m = ob.build(arm)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    rng = np.random.default_rng(3)
    ts = []
    m.train()
    for i in range(13):
        offs = rng.integers(0, len(train) - 257, size=8)
        xb = torch.from_numpy(np.stack([train[o:o + 257] for o in offs]).astype(np.int64))
        t0 = time.perf_counter()
        lg = m(xb[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 256), xb[:, 1:].reshape(-1))
        pen = getattr(m, 'last_penalty', None)
        (loss + 0.02 * pen if pen is not None else loss).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        ts.append(time.perf_counter() - t0)
    train_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    trained = os.environ.get('OM_TRAINED_' + arm.upper())
    if trained:
        m.load_state_dict(torch.load(trained, map_location='cpu'))
    sample = next(w['data'] for w in test if w['domain'] == 'article')
    tm = ob.timing(m, sample, 256, reps=8, prompt=64, gen=192)
    with torch.no_grad():
        m.eval()
        st = st2 = None
        for t in sample[:256]:
            _, st = m.step(torch.tensor([int(t)]), st, hard=True)
        for t in sample[:256]:
            _, st2 = m.step(torch.tensor([int(t), int(t)]), st2, hard=True)
    out = dict(arm=arm, params=ob.nparams(m), train_s_per_step=statistics.median(ts[3:]), train_peak_rss_mib=train_rss,
               train_base_rss_mib=base_rss, train_model_rss_mib=train_rss - base_rss,
               prefill_tok_s=tm['prefill_tok_s'], decode_ms_median=tm['decode_ms_median'], decode_ms_p95=tm['decode_ms_p95'],
               decode_tok_s=tm['decode_tok_s'], decode_state_bytes_at_256=state_bytes(st),
               decode_state_bytes_per_seq_at_256=state_bytes(st2) - state_bytes(st), trained_weights=bool(trained),
               torch=torch.__version__)
    print(json.dumps(out), flush=True)


def main():
    if sys.argv[1] == '--worker':
        return worker(sys.argv[2])
    arms, out_path = sys.argv[1].split(','), Path(sys.argv[2])
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    res = {a: [] for a in arms}
    for r in range(rounds):
        for arm in (arms if r % 2 == 0 else list(reversed(arms))):
            p = subprocess.run([sys.executable, '-u', __file__, '--worker', arm], capture_output=True, text=True, env=env,
                               timeout=3600)
            line = [l for l in p.stdout.splitlines() if l.startswith('{')]
            if p.returncode or not line:
                res[arm].append(dict(error=(p.stderr or p.stdout)[-500:]))
            else:
                res[arm].append(json.loads(line[-1]))
            print(arm, res[arm][-1], flush=True)
    summary = {}
    for arm, rs in res.items():
        ok = [x for x in rs if 'error' not in x]
        if not ok:
            summary[arm] = dict(error=rs[-1].get('error'))
            continue
        summary[arm] = {k: (statistics.median(x[k] for x in ok) if isinstance(ok[0][k], (int, float)) else ok[0][k])
                        for k in ok[0] if k != 'arm'}
        summary[arm]['rounds'] = len(ok)
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '')
    out_path.write_text(json.dumps(dict(cpu=cpu, logical_cpus=os.cpu_count(), threads=1, time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                        arms=summary, raw=res), indent=1) + '\n')


if __name__ == '__main__':
    main()
