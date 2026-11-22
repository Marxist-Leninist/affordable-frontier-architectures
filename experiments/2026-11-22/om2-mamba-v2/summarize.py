#!/usr/bin/env python3
"""Aggregate OM2 runs: quality, speed, training cost, recall, gate use, paired contrasts."""
import json, math, random, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'runs'
DOMAINS = ('article', 'fiction', 'dialogue')
LN2 = math.log(2.0)
NAMES = dict(tf='Transformer', m2hf='Mamba-2 (HF reference code)', m2='Mamba-2 (HF definition and init, our kernel)',
             om1='Our Mamba v1', om2='Our Mamba v2',
             om2pos='Our Mamba v2, positive decay only', om2p='v2 + placebo sidecar (no lookup)',
             om2a='v2 + always-on exact store', om2g='v2 + state-gated exact store')
ORDER = ('tf', 'm2hf', 'm2', 'om1', 'om2', 'om2pos', 'om2p', 'om2a', 'om2g')

rows = {}
for f in sorted(RUNS.glob('*/result.json')):
    r = json.loads(f.read_text())
    if r.get('status') == 'ok' and 'test' in r:
        rows.setdefault(r['arm'], []).append(r)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def macro_from_docs(docs):
    per = {}
    for d in docs:
        a = per.setdefault(d['domain'], [0.0, 0])
        a[0] += d['nll']
        a[1] += d['targets']
    return statistics.mean(per[k][0] / per[k][1] / LN2 for k in per)


def time_to(curve, target):
    prev = None
    for c in curve:
        if c['dev_bpb'] <= target:
            if prev is None:
                return c['train_s']
            f = (prev['dev_bpb'] - target) / max(1e-12, prev['dev_bpb'] - c['dev_bpb'])
            return prev['train_s'] + f * (c['train_s'] - prev['train_s'])
        prev = c
    return None


summary = {}
for arm in ORDER:
    rs = rows.get(arm)
    if not rs:
        continue
    s = dict(name=NAMES[arm], seeds=[r['seed'] for r in rs], params=rs[0]['params'],
             test_bpb=mean(r['test']['macro'] for r in rs),
             per_seed_bpb=[r['test']['macro'] for r in rs],
             domains={d: mean(r['test'].get(d) for r in rs) for d in DOMAINS},
             train_s=mean(r['train_s'] for r in rs), s_per_step=mean(r['train_s_per_step'] for r in rs),
             dev_start=mean(r['curve'][0]['dev_bpb'] for r in rs), dev_final=mean(r['curve'][-1]['dev_bpb'] for r in rs),
             prefill_tok_s=mean(r['timing']['prefill_tok_s'] for r in rs),
             decode_tok_s=mean(r['timing']['decode_tok_s'] for r in rs),
             decode_ms_median=mean(r['timing']['decode_ms_median'] for r in rs),
             decode_ms_p95=mean(r['timing']['decode_ms_p95'] for r in rs),
             peak_rss_mib=max(r['peak_rss_mib'] for r in rs), parity_max_abs=max(r['parity_max_abs'] for r in rs),
             copy={g: dict(second_acc=mean(r['copy'][g]['second_acc'] for r in rs),
                           first_acc=mean(r['copy'][g]['first_acc'] for r in rs),
                           copy_gain_bpb=mean(r['copy'][g]['copy_gain_bpb'] for r in rs)) for g in rs[0]['copy']},
             sample=rs[0]['timing']['greedy_sample'][:160])
    if 'test_hard' in rs[0]:
        s['test_bpb_hard'] = mean(r['test_hard']['macro'] for r in rs)
        s['read_rate_hard'] = mean(r['gate_hard']['read_rate'] for r in rs)
    if 'gate_soft' in rs[0]:
        s['gate_open_soft'] = mean(r['gate_soft']['open_rate'] for r in rs)
        s['gate_mean_soft'] = mean(r['gate_soft']['mean_gate'] for r in rs)
    s['bits_per_byte_gained_per_train_min'] = (s['dev_start'] - s['dev_final']) / (s['train_s'] / 60.0)
    summary[arm] = s

if 'tf' in summary:
    tf = summary['tf']
    target = max(s['dev_final'] for s in summary.values())
    for arm, s in summary.items():
        s['quality_index'] = 100.0 * 2 ** (tf['test_bpb'] - s['test_bpb'])
        s['train_speed_rel'] = tf['train_s'] / s['train_s']
        s['decode_speed_rel'] = s['decode_tok_s'] / tf['decode_tok_s']
        s['passes_quality_gate'] = s['quality_index'] >= 100.0
        s['combined_index'] = (100.0 * (s['quality_index'] / 100.0 * s['train_speed_rel'] * s['decode_speed_rel']) ** (1 / 3)
                               if s['passes_quality_gate'] else None)
        s['dev_target_bpb'] = target
        s['train_s_to_target'] = mean(time_to(r['curve'], target) for r in rows[arm])

# paired document bootstrap over seeds (descriptive)
contrasts = {}
pairs = [('om2', 'om1'), ('om2', 'm2'), ('om2', 'om2pos'), ('om2pos', 'm2'), ('om2', 'tf'), ('m2', 'om1'),
         ('m2', 'tf'), ('om1', 'tf'), ('om2g', 'om2a'), ('om2g', 'om2p'), ('om2a', 'om2p'), ('om2p', 'om2'),
         ('om2g', 'om2'), ('om2a', 'tf'), ('om2g', 'tf')]
rng = random.Random(20261122)
for a, b in pairs:
    if a not in rows or b not in rows:
        continue
    seeds = sorted(set(r['seed'] for r in rows[a]) & set(r['seed'] for r in rows[b]))
    if not seeds:
        continue
    da = {r['seed']: r['test_docs'] for r in rows[a]}
    db = {r['seed']: r['test_docs'] for r in rows[b]}
    diff = statistics.mean(macro_from_docs(da[s]) - macro_from_docs(db[s]) for s in seeds)
    n = len(da[seeds[0]])
    idx_by_dom = {}
    for i, d in enumerate(da[seeds[0]]):
        idx_by_dom.setdefault(d['domain'], []).append(i)
    boots = []
    for _ in range(2000):
        pick = [i for dm, ix in idx_by_dom.items() for i in (rng.choice(ix) for _ in ix)]
        vals = []
        for s in seeds:
            A = [da[s][i] for i in pick]
            Bv = [db[s][i] for i in pick]
            vals.append(macro_from_docs(A) - macro_from_docs(Bv))
        boots.append(statistics.mean(vals))
    boots.sort()
    per_seed = [macro_from_docs(da[s]) - macro_from_docs(db[s]) for s in seeds]
    contrasts[f'{a} minus {b}'] = dict(mean_bpb=diff, per_seed=per_seed, interval95=[boots[49], boots[1949]],
                                       seeds=seeds)

out = dict(summary=summary, contrasts=contrasts)
(ROOT / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')

L = ['# OM2 results', '', '| rank | arm | params | test bpb (macro) | article | fiction | dialogue | quality index | '
     'train s | bpb gained / train min | decode tok/s | decode p95 ms | combined |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
def f3(v):
    return '-' if v is None else f'{v:.3f}'


for k, (arm, s) in enumerate(sorted(summary.items(), key=lambda kv: kv[1]['test_bpb']), 1):
    dm = s['domains']
    L.append(f"| {k} | {s['name']} | {s['params']:,} | {s['test_bpb']:.4f} | {f3(dm['article'])} | {f3(dm['fiction'])} | "
             f"{f3(dm['dialogue'])} | {s.get('quality_index', float('nan')):.1f} | {s['train_s']:.0f} | "
             f"{s['bits_per_byte_gained_per_train_min']:.4f} | {s['decode_tok_s']:.1f} | {s['decode_ms_p95']:.2f} | "
             f"{'' if s.get('combined_index') is None else format(s['combined_index'], '.1f')} |")
L += ['', '## Exact recall under distractors (second occurrence, bytes 5-32 of a repeated 32-byte span)', '',
      '| arm | ' + ' | '.join(f'gap {g}' for g in next(iter(summary.values()))['copy']) + ' |',
      '|---|' + '---|' * len(next(iter(summary.values()))['copy'])]
for arm, s in summary.items():
    L.append(f"| {s['name']} | " + ' | '.join(f"{v['second_acc'] * 100:.1f}% (gain {v['copy_gain_bpb']:.2f})"
                                               for v in s['copy'].values()) + ' |')
L += ['', '## Paired contrasts (test macro bpb, negative = first arm better; document bootstrap, descriptive)', '']
for k, v in contrasts.items():
    L.append(f"- {k}: {v['mean_bpb']:+.4f} [{v['interval95'][0]:+.4f}, {v['interval95'][1]:+.4f}], per seed "
             + ', '.join(f'{x:+.4f}' for x in v['per_seed']))
for arm in ('om2p', 'om2g'):
    if arm in summary:
        s = summary[arm]
        L.append(f"- {s['name']}: soft gate open rate {s.get('gate_open_soft', float('nan')):.3f}, mean gate "
                 f"{s.get('gate_mean_soft', float('nan')):.3f}" + (f"; hard-gated test bpb {s['test_bpb_hard']:.4f} with "
                 f"reads on {s['read_rate_hard'] * 100:.1f}% of tokens" if 'test_bpb_hard' in s else ''))
(ROOT / 'RESULTS.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
