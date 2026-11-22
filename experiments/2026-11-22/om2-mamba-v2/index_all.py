#!/usr/bin/env python3
"""One at-a-glance index for every arm of OM2 + OM3: quality, exact recall, training speed, decode and prefill speed,
memory, and learning speed, each relative to the Transformer arm (= 100), plus a weighted combined index.

Speed and memory come from ONE same-machine table (speed_all.py), not from the training runs, which ran on different
machines. Quality and recall come from the training runs (mean over seeds).
Usage: index_all.py RUNS_DIR[,RUNS_DIR...] SPEED_JSON OUT_DIR"""
import json, math, statistics, sys
from pathlib import Path

RUN_DIRS = [Path(p) for p in sys.argv[1].split(',')]
SPEED_DOC = json.loads(Path(sys.argv[2]).read_text())
SPEED = SPEED_DOC['arms']
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
NAMES = dict(tf='Transformer', m2='Mamba-2', om1='Our Mamba v1', om2='Our Mamba v2', om2pos='Our Mamba v2, positive decay',
             om2p='v2 + placebo sidecar', om2a='v2 + always-on exact store', om2g='v2 + state-gated exact store ("Mamba-4" idea)',
             hyb='Transformer-Mamba hybrid', griffin='Griffin (RG-LRU + local attention)', gdn='Gated DeltaNet',
             titans='Titans-style (MAG, linear memory)', mlstm='xLSTM (mLSTM)', retnet='RetNet', rwkv7='RWKV-7-style', hyena='Hyena')
FAMILY = dict(tf='Transformer', m2='Mamba', om1='Mamba (ours)', om2='Mamba (ours)', om2pos='Mamba (ours)', om2p='Mamba (ours) + sidecar',
              om2a='Mamba (ours) + exact store', om2g='Mamba (ours) + exact store', hyb='Transformer-Mamba hybrid',
              griffin='Griffin / RecurrentGemma', gdn='Gated DeltaNet', titans='Google Titans', mlstm='xLSTM', retnet='RetNet',
              rwkv7='RWKV-7', hyena='Hyena')
SPEED_ARM = dict(m2='m2k')          # m2's compute is timed as m2k (same kernel, no transformers import)
WEIGHTS = dict(quality=0.35, recall=0.15, train_speed=0.15, decode_speed=0.15, state_memory=0.10, prefill_speed=0.05,
               train_memory=0.05)
GATE = 80.0                          # quality index below this is shown but not ranked

runs = {}
for d in RUN_DIRS:
    for f in sorted(d.glob('*/result.json')):
        r = json.loads(f.read_text())
        if r.get('status') == 'ok' and 'test' in r:
            runs.setdefault(r['arm'], []).append(r)


def mean(xs):
    return statistics.mean(xs)


rows = {}
for arm, rs in runs.items():
    sp = SPEED.get(SPEED_ARM.get(arm, arm), {})
    if 'train_s_per_step' not in sp:
        continue
    gaps = sorted(rs[0]['copy'], key=int)
    rows[arm] = dict(
        name=NAMES.get(arm, arm), family=FAMILY.get(arm, arm), seeds=[r['seed'] for r in rs], params=rs[0]['params'],
        bpb=mean([r['test']['macro'] for r in rs]),
        domains={k: mean([r['test'][k] for r in rs]) for k in ('article', 'fiction', 'dialogue')},
        recall=mean([mean([r['copy'][g]['second_acc'] for g in gaps]) for r in rs]),
        recall_by_gap={g: mean([r['copy'][g]['second_acc'] for r in rs]) for g in gaps},
        dev_start=mean([r['curve'][0]['dev_bpb'] for r in rs]), dev_final=mean([r['curve'][-1]['dev_bpb'] for r in rs]),
        steps=rs[0]['steps'], curves=[[(c['step'], c['dev_bpb']) for c in r['curve']] for r in rs],
        train_s_per_step=sp['train_s_per_step'], train_rss=sp['train_peak_rss_mib'], prefill=sp['prefill_tok_s'],
        decode_tok_s=sp['decode_tok_s'], decode_ms=sp['decode_ms_median'], decode_p95=sp['decode_ms_p95'],
        state_bytes=sp['decode_state_bytes_at_256'])

tf = rows['tf']
target = max(r['dev_final'] for r in rows.values())


def steps_to(curve, tgt):
    prev = None
    for s, v in curve:
        if v <= tgt:
            if prev is None:
                return s
            f = (prev[1] - tgt) / max(1e-12, prev[1] - v)
            return prev[0] + f * (s - prev[0])
        prev = (s, v)
    return None


for arm, r in rows.items():
    idx = dict(quality=100 * 2 ** (tf['bpb'] - r['bpb']), recall=100 * r['recall'] / tf['recall'],
               train_speed=100 * tf['train_s_per_step'] / r['train_s_per_step'],
               decode_speed=100 * r['decode_tok_s'] / tf['decode_tok_s'], prefill_speed=100 * r['prefill'] / tf['prefill'],
               state_memory=100 * tf['state_bytes'] / r['state_bytes'], train_memory=100 * tf['train_rss'] / r['train_rss'])
    r['index'] = idx
    r['combined'] = 100 * math.exp(sum(w * math.log(idx[k] / 100) for k, w in WEIGHTS.items()))
    r['ranked'] = idx['quality'] >= GATE
    minutes = r['steps'] * r['train_s_per_step'] / 60
    r['bits_gained_per_train_min'] = (r['dev_start'] - r['dev_final']) / minutes
    st = [steps_to(c, target) for c in r['curves']]
    r['train_min_to_common_target'] = (mean(st) * r['train_s_per_step'] / 60) if all(s is not None for s in st) else None
    r.pop('curves')

order = sorted(rows, key=lambda a: (not rows[a]['ranked'], -rows[a]['combined']))
(OUT / 'index.json').write_text(json.dumps(dict(weights=WEIGHTS, gate_quality_index=GATE, common_dev_target_bpb=target,
                                                speed_machine=dict(cpu=SPEED_DOC['cpu'], threads=1, time=SPEED_DOC['time']),
                                                order=order, rows=rows), indent=1) + '\n')
L = ['| rank | model | family | **combined** | quality (bpb) | exact recall | training speed (s/step) | decode (ms/token) | '
     'prefill | state memory at 256 tokens | training memory | bits gained per train minute |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|']
k = 0
for arm in order:
    r, i = rows[arm], rows[arm]['index']
    k += r['ranked']
    L.append(f"| {k if r['ranked'] else '-'} | {r['name']} | {r['family']} | **{r['combined']:.1f}** | {i['quality']:.0f} ({r['bpb']:.3f}) | "
             f"{i['recall']:.0f} ({r['recall'] * 100:.0f}%) | {i['train_speed']:.0f} ({r['train_s_per_step']:.2f}) | "
             f"{i['decode_speed']:.0f} ({r['decode_ms']:.1f}) | {i['prefill_speed']:.0f} | {i['state_memory']:.0f} "
             f"({r['state_bytes'] / 1024:.0f} KiB) | {i['train_memory']:.0f} ({r['train_rss']:.0f} MiB) | "
             f"{r['bits_gained_per_train_min']:.3f} |")
L += ['', 'Each column is an index with the Transformer = 100 (higher is better), raw value in brackets. Combined = weighted '
      'geometric mean: ' + ', '.join(f'{k} {w:.0%}' for k, w in WEIGHTS.items()) + f'. Rows with a quality index below {GATE:.0f} '
      'are shown but not ranked. Speed and memory: one machine (' + SPEED_DOC['cpu'] + '), one thread, every arm in its own process.']
(OUT / 'INDEX.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
