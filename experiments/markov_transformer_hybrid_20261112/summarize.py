#!/usr/bin/env python3
"""summarize.py [--runs runs] [--tag main] [--out results] -- tables from the finished runs.

For each corpus and variant: held-out test loss (mean over seeds), the seed-paired difference from the classical
transformer with a 95% paired-t interval, the dev curve at fixed steps, how many steps the variant needs to reach the
loss the classical transformer ends at, the measured CPU cost per step, and the comparison at equal CPU time.
"""
import argparse
import glob
import json
import math
import os
import statistics as st

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def interp_loss_at(evals, step):
    """dev loss at a (fractional) step, linear in log(step) between evaluations; None outside the measured range"""
    pts = [(e['step'], e['dev']) for e in evals]
    if step < pts[0][0] or step > pts[-1][0]:
        return None
    for (s0, l0), (s1, l1) in zip(pts, pts[1:]):
        if s0 <= step <= s1:
            f = (math.log(step) - math.log(s0)) / (math.log(s1) - math.log(s0))
            return l0 + f * (l1 - l0)
    return pts[-1][1]


def step_reaching(evals, target):
    """first (fractional) step at which the dev loss is <= target; None if never"""
    pts = [(e['step'], e['dev']) for e in evals]
    if pts[0][1] <= target:
        return float(pts[0][0])
    for (s0, l0), (s1, l1) in zip(pts, pts[1:]):
        if l1 <= target < l0:
            f = (l0 - target) / (l0 - l1)
            return math.exp(math.log(s0) + f * (math.log(s1) - math.log(s0)))
    return None


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None, None
    sd = st.stdev(xs)
    return m, sd, (T95.get(n, 2.0) * sd / math.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--lr', default=None, help='only runs at this learning rate (default: all, grouped)')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    R = {}
    for f in sorted(glob.glob(os.path.join(a.runs, a.tag + '_*.json'))):
        r = json.load(open(f))
        if r.get('status') != 'done':
            continue
        g = r['args']
        if a.lr and float(g['lr']) != float(a.lr):
            continue
        key = (os.path.basename(g['data']), g['d'], g['L'], g['steps'], g['bs'], float(g['lr']))
        R.setdefault(key, {}).setdefault(g['variant'], {})[g['seed']] = r
    out = {}
    lines = []
    for key in sorted(R):
        data, d, L, steps, bs, lr = key
        V = R[key]
        if 'base' not in V:
            continue
        base = V['base']
        meta = {}
        try:
            meta = json.load(open(os.path.join('data', data, 'meta.json')))
        except Exception:  # noqa: BLE001
            pass
        chain = {o: v['test'] for o, v in meta.get('chain_alone_loss', {}).items()}
        hdr = f'{data}  d={d} L={L} steps={steps} batch={bs} lr={lr:g}'
        lines += ['', '## ' + hdr, '']
        if chain:
            lines += ['Markov chain alone, test loss: ' + ', '.join(f'order {int(o) - 1}: {v:.3f}' for o, v in sorted(chain.items())), '']
        lines += ['| variant | seeds | test loss | vs classical (paired) | 95% interval | dev@100 | dev@400 | dev@end | steps to reach '
                  'classical\'s final loss | CPU ms/step | cost ratio | at equal CPU time |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
        res = {}
        order = ['base'] + sorted(v for v in V if v != 'base')
        for v in order:
            runs = V[v]
            seeds = sorted(runs)
            tests = [runs[s]['test_loss'] for s in seeds]
            m, sd, _ = mean_ci(tests)
            both = [s for s in seeds if s in base]
            diffs = [runs[s]['test_loss'] - base[s]['test_loss'] for s in both]
            dm, dsd, dci = mean_ci(diffs) if diffs else (None, None, None)
            ratios = [runs[s]['cpu_ms_per_step'] / base[s]['cpu_ms_per_step'] for s in both]
            ratio = st.median(ratios) if ratios else None
            reach, eqt = [], []
            for s in both:
                target = base[s]['evals'][-1]['dev']
                sr = step_reaching(runs[s]['evals'], target)
                reach.append(sr)
                r_s = runs[s]['cpu_ms_per_step'] / base[s]['cpu_ms_per_step']
                l_eq = interp_loss_at(runs[s]['evals'], steps / r_s) if r_s > 1 else runs[s]['evals'][-1]['dev']
                if l_eq is not None:
                    eqt.append(l_eq - target)

            def dev_at(step):
                xs = [interp_loss_at(runs[s]['evals'], step) for s in seeds]
                xs = [x for x in xs if x is not None]
                return sum(xs) / len(xs) if xs else None
            reach_ok = [x for x in reach if x is not None]
            res[v] = {'seeds': seeds, 'test_mean': m, 'test_sd': sd, 'tests': tests, 'diff_mean': dm, 'diff_sd': dsd,
                      'diff_ci95': dci, 'diffs': diffs, 'dev100': dev_at(100), 'dev400': dev_at(400), 'dev_end': dev_at(steps),
                      'steps_to_reach_base_final': reach, 'cpu_ms_per_step': st.median([runs[s]['cpu_ms_per_step'] for s in seeds]),
                      'cost_ratio': ratio, 'equal_time_diff': (sum(eqt) / len(eqt)) if eqt else None,
                      'chain_cpu_ms': st.median([runs[s].get('prior_cpu_ms_per_step', 0.0) for s in seeds]),
                      'params': runs[seeds[0]]['params'],
                      'extra': {k: runs[seeds[0]].get('test_info', {}).get(k) for k in ('lam', 'gate', 'mk_b')}}
            f = lambda x, p=3: '-' if x is None else f'{x:.{p}f}'   # noqa: E731
            reach_txt = '-' if v == 'base' else (('never in %d' % steps) if not reach_ok else
                                                 f'{sum(reach_ok) / len(reach_ok):.0f} ({100 * sum(reach_ok) / len(reach_ok) / steps:.0f}%)'
                                                 + ('' if len(reach_ok) == len(reach) else f' [{len(reach_ok)}/{len(reach)} seeds]'))
            lines.append(f"| {v} | {len(seeds)} | {m:.3f}" + (f' ± {sd:.3f}' if sd is not None else '') +
                         f" | {'-' if v == 'base' else f(dm, 3) if dm is None else f'{dm:+.3f}'} | "
                         f"{'-' if dci is None or v == 'base' else f'[{dm - dci:+.3f}, {dm + dci:+.3f}]'} | {f(res[v]['dev100'])} | "
                         f"{f(res[v]['dev400'])} | {f(res[v]['dev_end'])} | {reach_txt} | {res[v]['cpu_ms_per_step']:.0f} | "
                         f"{'-' if v == 'base' or ratio is None else f'x{ratio:.2f}'} | "
                         f"{'-' if v == 'base' or res[v]['equal_time_diff'] is None else format(res[v]['equal_time_diff'], '+.3f')} |")
        out[hdr] = {'chain_alone_test': chain, 'variants': res}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f'summary_{a.tag}.json'), 'w'), indent=1)
    txt = '\n'.join(lines) + '\n'
    open(os.path.join(a.out, f'summary_{a.tag}.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
