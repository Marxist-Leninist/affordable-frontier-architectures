#!/usr/bin/env python3
"""CPU time and memory: transformers' Mamba-2 torch path ('hf') vs the patched kernel ('new').

Same weights, same inputs, same thread count. Each implementation runs in its own fresh process per round,
rounds alternate which implementation goes first, and every timed pass records wall time, user and system
CPU time, and minor/major page faults. Each case runs under the default glibc allocator and, when available,
under tcmalloc (LD_PRELOAD), because the stock path allocates large short-lived temporaries and its speed then
depends strongly on the allocator and on the machine's memory pressure. The machine's memory and swap state is
recorded before each process starts. Writes results/bench_<label>.json.
"""
import argparse, json, os, platform, resource, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TCMALLOC = [p for p in ('/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4', '/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4',
                        '/opt/conda/lib/libtcmalloc_minimal.so.4') if os.path.exists(p)]
CASES = {
    'fwd_bwd_small_b8_T256': dict(kind='fwd_bwd', model='small', batch=8, T=256, reps=5, warm=2),
    'forward_130m_T256': dict(kind='forward', model='130m', batch=1, T=256, reps=3, warm=1),
    'forward_130m_T512': dict(kind='forward', model='130m', batch=1, T=512, reps=2, warm=1),
    'forward_130m_T2048': dict(kind='forward', model='130m', batch=1, T=2048, reps=3, warm=1, impls=('new',)),
    'fwd_bwd_130m_b1_T256': dict(kind='fwd_bwd', model='130m', batch=1, T=256, reps=3, warm=1, impls=('new',)),
}


def meminfo():
    m = {}
    for line in open('/proc/meminfo'):
        k, v = line.split(':', 1)
        if k in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            m[k] = int(v.split()[0]) // 1024
    m['load1'] = os.getloadavg()[0]
    return m


def cpu_model():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def worker(case, impl, threads, repo):
    import torch
    from torch.nn import functional as F
    from transformers import Mamba2Config, Mamba2ForCausalLM
    sys.path.insert(0, str(HERE))
    import mamba2_cpu_ssd as fast
    torch.set_num_threads(threads)
    c = CASES[case]
    torch.manual_seed(0)
    if c['model'] == 'small':
        m = Mamba2ForCausalLM(Mamba2Config(vocab_size=256, hidden_size=128, state_size=32, num_heads=8, head_dim=32,
                                           expand=2, n_groups=1, conv_kernel=4, num_hidden_layers=4, chunk_size=32,
                                           use_cache=False, tie_word_embeddings=True, pad_token_id=0,
                                           bos_token_id=0, eos_token_id=0))
    else:
        m = Mamba2ForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
    if impl == 'new':
        fast.patch_mamba2(m)
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, m.config.vocab_size, (c['batch'], c['T'] + 1), generator=g)
    passes = []
    for i in range(c['warm'] + c['reps']):
        r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
        if c['kind'] == 'fwd_bwd':
            m.train()
            lg = m(input_ids=x[:, :-1], use_cache=False).logits
            F.cross_entropy(lg.flatten(0, 1), x[:, 1:].flatten()).backward()
            m.zero_grad(set_to_none=True)
        else:
            m.eval()
            with torch.no_grad():
                m(input_ids=x[:, :-1], use_cache=False)
        t1, r1 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
        if i >= c['warm']:
            passes.append(dict(wall_s=t1 - t0, user_s=r1.ru_utime - r0.ru_utime, sys_s=r1.ru_stime - r0.ru_stime,
                               minor_faults=r1.ru_minflt - r0.ru_minflt, major_faults=r1.ru_majflt - r0.ru_majflt))
    print(json.dumps(dict(case=case, impl=impl, passes=passes,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)), flush=True)


def summarise(passes):
    med = lambda k: statistics.median(p[k] for p in passes)
    w = [p['wall_s'] for p in passes]
    return dict(median_s=med('wall_s'), min_s=min(w), max_s=max(w), n=len(passes), user_s=med('user_s'), sys_s=med('sys_s'),
                minor_faults=med('minor_faults'), major_faults=med('major_faults'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--repo', default='AntonV/mamba2-130m-hf')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--cases', default=','.join(CASES))
    ap.add_argument('--allocators', default='glibc,tcmalloc')
    ap.add_argument('--label', default='machine')
    ap.add_argument('--worker', nargs=2)
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker[0], a.worker[1], a.threads, a.repo)
    import torch, transformers
    out = dict(label=a.label, cpu=cpu_model(), logical_cpus=os.cpu_count(), threads=a.threads, torch=torch.__version__,
               transformers=transformers.__version__, started=time.strftime('%Y-%m-%d %H:%M:%S'), memory_at_start=meminfo(),
               tcmalloc=TCMALLOC[0] if TCMALLOC else None, cases={})
    allocs = [x for x in a.allocators.split(',') if x == 'glibc' or (x == 'tcmalloc' and TCMALLOC)]
    for case in a.cases.split(','):
        impls = CASES[case].get('impls', ('hf', 'new'))
        for alloc in allocs:
            env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
            if alloc == 'tcmalloc':
                env['LD_PRELOAD'] = TCMALLOC[0]
            runs, rss, mem, err = {i: [] for i in impls}, {i: 0.0 for i in impls}, [], {}
            for rnd in range(a.rounds):
                order = impls if rnd % 2 == 0 else tuple(reversed(impls))
                for impl in order:
                    mem.append(dict(impl=impl, **meminfo()))
                    p = subprocess.run([sys.executable, '-u', __file__, '--threads', str(a.threads), '--repo', a.repo,
                                        '--worker', case, impl], capture_output=True, text=True, env=env, timeout=3600)
                    line = [l for l in p.stdout.splitlines() if l.startswith('{')]
                    if p.returncode or not line:
                        err[impl] = (p.stderr or p.stdout)[-600:]
                        continue
                    r = json.loads(line[-1])
                    runs[impl] += r['passes']
                    rss[impl] = max(rss[impl], r['peak_rss_mib'])
            row = {k: v for k, v in CASES[case].items() if k != 'impls'}
            row['memory_before_each_process'] = mem
            if err:
                row['errors'] = err
            for impl in impls:
                if runs[impl]:
                    row[impl] = dict(summarise(runs[impl]), peak_rss_mib=rss[impl])
            if row.get('hf') and row.get('new'):
                row['speedup'] = row['hf']['median_s'] / row['new']['median_s']
                row['speedup_user_cpu'] = row['hf']['user_s'] / max(1e-9, row['new']['user_s'])
                row['memory_ratio'] = row['hf']['peak_rss_mib'] / row['new']['peak_rss_mib']
            out['cases'].setdefault(case, {})[alloc] = row
            print(json.dumps({case: {alloc: {k: row.get(k) for k in ('hf', 'new', 'speedup', 'speedup_user_cpu')}}}), flush=True)
    out['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / f'bench_{a.label}.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
