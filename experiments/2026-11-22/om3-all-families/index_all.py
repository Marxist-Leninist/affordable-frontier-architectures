#!/usr/bin/env python3
"""One at-a-glance index for every arm of OM2 + OM3: quality, repeat-span accuracy, training speed, decode and prefill
speed, inference and training memory, each relative to the Transformer arm (= 100), plus a weighted combined index, the
rank of every model under other reasonable ways of counting (so a reader can see which conclusions survive), the spread
of each combined score over seed x timing-round pairs, and marks for rows that are not separated.

Sources. Quality and repeat-span accuracy come from the training runs (mean over seeds). Speed and memory come from the
same-machine speed table built by merge_speed.py: every timing is the median over all valid per-process rounds of two passes
(the state-gated store and the control: the second pass only).
Definitions.
- quality: 100 * 2^(Transformer bits/byte - model bits/byte), the ratio of per-byte perplexities. Bits/byte is the mean
  of the three test domains (macro). The state-gated store (om2g) is scored as designed, reading only where its gate
  exceeds 0.5 (test_hard); its soft-read score is kept in its row as bpb_soft. (Its dev curves, used only for the
  minutes-to-target column and the time-to-quality alternative, are soft-read: the trainer evaluates dev without the gate.)
- repeat-span accuracy ("recall" in the data files; OM2 called it exact recall): next-byte accuracy on a 32-byte span
  repeated after 16-176 bytes of another document. No model predicts the repeat better than its first occurrence, so at
  this size it measures next-byte skill, not copying.
- training speed, decode speed, prefill speed: Transformer median / model median (medians over rounds).
- inference memory: fp32 weight bytes + per-sequence decode state after 256 tokens at batch 1 (the batch-2 minus batch-1
  cache, which leaves out input-independent tables such as Hyena's filter taps).
- training memory: peak RSS of the training step above the process baseline (PyTorch and data, about 0.46 GiB).
Choices made after seeing the results are each one of the alternative ways of counting below, so their effect shows in
the rank ranges and in the leader under each.
Spread and marks. Each pair is one seed and one timing round: that seed's quality and repeat-span accuracy against the
Transformer's at the same seed (both seeds train every arm on the same byte stream), and that round's own timings against
the Transformer's medians. The spread is the lowest and highest pair score. The shared machine switched between a fast and
a slow mode within tens of seconds, so single rounds are noisy; a row is marked "≈" when its spread overlaps the spread of
the row above (not separated from it). Under each alternative, leader and runner-up are called level the same way.
Usage: index_all.py RUNS_DIR[,RUNS_DIR...] SPEED_JSON OUT_DIR
(RUNS_DIR holds either <arm>-<seed>/result.json folders or flat <arm>-<seed>.json files.)"""
import json, math, statistics, sys
from pathlib import Path

RUN_DIRS = [Path(p) for p in sys.argv[1].split(',')]
SPEED_PATH = Path(sys.argv[2])
SPEED_DOC = json.loads(SPEED_PATH.read_text())
SPEED = SPEED_DOC['arms']
RAW = SPEED_DOC.get('raw', {})
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
NAMES = dict(tf='Transformer', m2='Mamba-2', om1='Our Mamba v1', om2='Our Mamba v2', om2pos='Our Mamba v2, positive decay',
             om2p='v2 + placebo sidecar', om2a='v2 + always-on exact store',
             om2g='v2 + state-gated exact store ("Mamba-4" idea, gate as designed)',
             hyb='Transformer-Mamba hybrid', griffin='Griffin (RG-LRU + local attention)', gdn='Gated DeltaNet (mixer only)',
             titans='Titans-style (MAG, linear memory)', mlstm='xLSTM (mLSTM)', retnet='RetNet', rwkv7='RWKV-7-style', hyena='Hyena')
FAMILY = dict(tf='Transformer', m2='Mamba', om1='Mamba (ours)', om2='Mamba (ours)', om2pos='Mamba (ours)', om2p='Mamba (ours) + sidecar',
              om2a='Mamba (ours) + exact store', om2g='Mamba (ours) + exact store', hyb='Transformer-Mamba hybrid',
              griffin='Griffin / RecurrentGemma', gdn='Gated DeltaNet', titans='Google Titans', mlstm='xLSTM', retnet='RetNet',
              rwkv7='RWKV-7', hyena='Hyena')
LABELS = dict(quality='quality', recall='repeat-span accuracy', train_speed='training speed', decode_speed='decode speed',
              inference_memory='inference memory', prefill_speed='prefill speed', train_memory='training memory',
              time_to_quality='time to quality', inference_memory_bf16='inference memory with bf16 weights',
              inference_memory_b8='inference memory at batch 8', state_only_memory='decode state only',
              decode_speed_best='decode speed (best round)', prefill_speed_best='prefill speed (best round)',
              train_memory_whole='whole-process training memory', inference_memory_total='inference memory with the whole decode cache',
              train_speed_p1='training speed (first pass)', decode_speed_p1='decode speed (first pass)',
              prefill_speed_p1='prefill speed (first pass)', train_speed_p2='training speed (second pass)',
              decode_speed_p2='decode speed (second pass)', prefill_speed_p2='prefill speed (second pass)',
              decode_speed_il='decode speed (interleaved)', prefill_speed_il='prefill speed (interleaved)')
SPEED_ARM = dict(m2='m2k')          # m2's compute is timed as m2k (same kernel, no transformers import)
HARD = {'om2g'}                     # scored with the gate as designed (test_hard / test_docs_hard)
WEIGHTS = dict(quality=0.35, recall=0.15, train_speed=0.15, decode_speed=0.15, inference_memory=0.10, prefill_speed=0.05,
               train_memory=0.05)
TIMING0 = dict(WEIGHTS, train_speed=0.0, decode_speed=0.0, prefill_speed=0.0)
SCHEMES = {   # key: (label, weights); used to show how stable each model's rank is
    'equal': ('equal weights', {k: 1.0 for k in WEIGHTS}),
    'q50': ('quality 50%', dict(quality=0.50, recall=0.10, train_speed=0.10, decode_speed=0.10, inference_memory=0.10,
                               prefill_speed=0.05, train_memory=0.05)),
    'speed': ('speed-heavy', dict(quality=0.25, recall=0.10, train_speed=0.25, decode_speed=0.20, inference_memory=0.10,
                                 prefill_speed=0.05, train_memory=0.05)),
    'norec': ('without repeat-span accuracy', {k: w for k, w in WEIGHTS.items() if k != 'recall'}),
    'ttq': ('training cost as time to quality', dict(WEIGHTS, train_speed=0.0, time_to_quality=0.15)),
    'infonly': ('inference only (no training cost)', dict(quality=0.45, recall=0.15, decode_speed=0.20, inference_memory=0.15,
                                                          prefill_speed=0.05)),
    'bf16': ('bf16 weights in inference memory', dict(WEIGHTS, inference_memory=0.0, inference_memory_bf16=0.10)),
    'b8': ('inference memory at batch 8', dict(WEIGHTS, inference_memory=0.0, inference_memory_b8=0.10)),
    'stateonly': ('decode state only as memory (OM2 definition)', dict(WEIGHTS, inference_memory=0.0, state_only_memory=0.10)),
    'rsswhole': ('whole-process training memory (OM2 definition)', dict(WEIGHTS, train_memory=0.0, train_memory_whole=0.05)),
    'statetotal': ('whole decode cache as memory (incl. input-independent tables)',
                   dict(WEIGHTS, inference_memory=0.0, inference_memory_total=0.10)),
    'latbest': ('best round instead of the median for prefill and decode', dict(WEIGHTS, decode_speed=0.0, prefill_speed=0.0,
                                                                                 decode_speed_best=0.15, prefill_speed_best=0.05)),
    'pass1': ('timings from the first pass only', dict(TIMING0, train_speed_p1=0.15, decode_speed_p1=0.15, prefill_speed_p1=0.05)),
    'pass2': ('timings from the second pass only', dict(TIMING0, train_speed_p2=0.15, decode_speed_p2=0.15, prefill_speed_p2=0.05)),
    'latil': ('prefill and decode timed interleaved in one process', dict(WEIGHTS, decode_speed=0.0, prefill_speed=0.0,
                                                                          decode_speed_il=0.15, prefill_speed_il=0.05)),
}
NWAYS = 1 + len(SCHEMES)
GATE = 80.0                          # quality index below this is shown but not ranked


def mean(xs):
    return statistics.mean(xs)


runs = {}
for d in RUN_DIRS:
    for f in sorted(list(d.glob('*/result.json')) + list(d.glob('*.json'))):
        r = json.loads(f.read_text())
        if not isinstance(r, dict) or r.get('status') != 'ok' or 'test' not in r or 'arm' not in r:
            continue
        if r['arm'] in HARD:
            r['test_soft'] = r['test']
            r['test'] = r['test_hard']
        runs.setdefault(r['arm'], []).append(r)


def spd(x):
    """speed-table fields used by the index, from the medians (arms block) or one timing round (raw); in a raw round the
    single-pass and interleaved variants fall back to that round's own values"""
    d = dict(train_s_per_step=x['train_s_per_step'], train_rss=x.get('train_model_rss_mib', x['train_peak_rss_mib']),
             train_rss_whole=x['train_peak_rss_mib'], prefill=x['prefill_tok_s'], decode_ms=x['decode_ms_median'],
             decode_p95=x['decode_ms_p95'],
             state_bytes=x.get('decode_state_bytes_per_seq_at_256', x['decode_state_bytes_at_256']),
             state_bytes_total=x['decode_state_bytes_at_256'],
             decode_ms_best=x.get('decode_ms_median_best', x['decode_ms_median']), prefill_best=x.get('prefill_tok_s_best', x['prefill_tok_s']))
    for tag in ('p1', 'p2'):
        src = x.get('pass1' if tag == 'p1' else 'pass2') or {}
        d[f'train_{tag}'] = src.get('train_s_per_step', d['train_s_per_step'])
        d[f'decode_{tag}'] = src.get('decode_ms_median', d['decode_ms'])
        d[f'prefill_{tag}'] = src.get('prefill_tok_s', d['prefill'])
    il = x.get('interleaved') or {}
    d['decode_il'] = il.get('decode_ms_median', d['decode_ms'])
    d['prefill_il'] = il.get('prefill_tok_s', d['prefill'])
    return d


rows = {}
for arm, rs in runs.items():
    sp = SPEED.get(SPEED_ARM.get(arm, arm), {})
    if 'train_s_per_step' not in sp:
        continue
    gaps = sorted(rs[0]['copy'], key=int)
    rows[arm] = dict(
        name=NAMES.get(arm, arm), family=FAMILY.get(arm, arm), seeds=[r['seed'] for r in rs], params=rs[0]['params'],
        bpb=mean([r['test']['macro'] for r in rs]), bpb_per_seed=[r['test']['macro'] for r in rs],
        domains={k: mean([r['test'][k] for r in rs]) for k in ('article', 'fiction', 'dialogue')},
        recall=mean([mean([r['copy'][g]['second_acc'] for g in gaps]) for r in rs]),
        recall_by_gap={g: mean([r['copy'][g]['second_acc'] for r in rs]) for g in gaps},
        copy_gain_bpb_range=[min(r['copy'][g]['copy_gain_bpb'] for r in rs for g in gaps),
                             max(r['copy'][g]['copy_gain_bpb'] for r in rs for g in gaps)],
        dev_start=mean([r['curve'][0]['dev_bpb'] for r in rs]), dev_final=mean([r['curve'][-1]['dev_bpb'] for r in rs]),
        steps=rs[0]['steps'], curves=[[(c['step'], c['dev_bpb']) for c in r['curve']] for r in rs],
        training_host_torch=sorted({r.get('torch', '?') for r in rs}), parity_max=max(r['parity_max_abs'] for r in rs),
        timing_rounds=sp.get('timing_rounds'), decode_ms_range=sp.get('decode_ms_range'), train_s_range=sp.get('train_s_range'),
        pass1_from_second=sp.get('pass1_from_second', False), **spd(sp))
    if arm in HARD:
        rows[arm]['bpb_soft'] = mean([r['test_soft']['macro'] for r in rs])
        rows[arm]['gate_read_rate_hard'] = mean([r['gate_hard']['read_rate'] for r in rs])
        rows[arm]['gate_mean_soft'] = [r['gate_soft']['mean_gate'] for r in rs]

tf = rows['tf']
target = max(c[-1][1] for r in rows.values() for c in r['curves'])   # worst final dev score of any single run
target_arm = next(a for a, r in rows.items() if any(c[-1][1] == target for c in r['curves']))
tf_target = max(c[-1][1] for c in tf['curves'])                     # the Transformer's own final dev score (weaker seed)
for a, r in rows.items():
    r['quality_rank'] = 1 + sum(o['bpb'] < r['bpb'] for o in rows.values())


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


def minutes_to(r, tgt):
    st = [steps_to(c, tgt) for c in r['curves']]
    return mean(st) * r['train_s_per_step'] / 60 if all(s is not None for s in st) else None


for arm, r in rows.items():
    r['train_min_to_common_target'] = minutes_to(r, target)
    r['train_min_to_tf_final'] = minutes_to(r, tf_target)


def indices(r, s, ref, refs):
    """per-metric indices of row r with speed fields s, against the Transformer row ref with speed fields refs"""
    P, Pt = r['params'], ref['params']
    d = dict(quality=100 * 2 ** (ref['bpb'] - r['bpb']), recall=100 * r['recall'] / ref['recall'],
             train_speed=100 * refs['train_s_per_step'] / s['train_s_per_step'],
             decode_speed=100 * refs['decode_ms'] / s['decode_ms'], prefill_speed=100 * s['prefill'] / refs['prefill'],
             decode_speed_best=100 * refs['decode_ms_best'] / s['decode_ms_best'],
             prefill_speed_best=100 * s['prefill_best'] / refs['prefill_best'],
             decode_speed_il=100 * refs['decode_il'] / s['decode_il'], prefill_speed_il=100 * s['prefill_il'] / refs['prefill_il'],
             inference_memory=100 * (4 * Pt + refs['state_bytes']) / (4 * P + s['state_bytes']),
             inference_memory_bf16=100 * (2 * Pt + refs['state_bytes']) / (2 * P + s['state_bytes']),
             inference_memory_b8=100 * (4 * Pt + 8 * refs['state_bytes']) / (4 * P + 8 * s['state_bytes']),
             inference_memory_total=100 * (4 * Pt + refs['state_bytes_total']) / (4 * P + s['state_bytes_total']),
             state_only_memory=100 * refs['state_bytes'] / s['state_bytes'],
             train_memory=100 * refs['train_rss'] / s['train_rss'],
             train_memory_whole=100 * refs['train_rss_whole'] / s['train_rss_whole'])
    for tag in ('p1', 'p2'):
        d[f'train_speed_{tag}'] = 100 * refs[f'train_{tag}'] / s[f'train_{tag}']
        d[f'decode_speed_{tag}'] = 100 * refs[f'decode_{tag}'] / s[f'decode_{tag}']
        d[f'prefill_speed_{tag}'] = 100 * s[f'prefill_{tag}'] / refs[f'prefill_{tag}']
    if r.get('train_min_to_common_target') and ref.get('train_min_to_common_target'):
        ratio = s['train_s_per_step'] / r['train_s_per_step']          # a timing round scales the minutes as well
        d['time_to_quality'] = 100 * ref['train_min_to_common_target'] / (r['train_min_to_common_target'] * ratio)
    return d


def combine(idx, w):
    return 100 * math.exp(sum(wk * math.log(idx[k] / 100) for k, wk in w.items() if wk) / sum(w.values()))


for arm, r in rows.items():
    r['weight_bytes'] = 4 * r['params']
    r['inference_bytes'] = r['weight_bytes'] + r['state_bytes']
    idx = indices(r, r, tf, tf)
    r['index'] = idx
    r['combined'] = combine(idx, WEIGHTS)
    r['ranked'] = idx['quality'] >= GATE
    r['alt_combined'] = {k: combine(idx, w) for k, (lab, w) in SCHEMES.items()}
    r['multipliers'] = {k: (idx[k] / 100) ** w for k, w in WEIGHTS.items()}
    r['bits_gained_per_train_min'] = (r['dev_start'] - r['dev_final']) / (r['steps'] * r['train_s_per_step'] / 60)

# seed x timing-round pairs: that seed's quality against the Transformer's at the same seed, that round's own timings
# against the Transformer's medians; the spread under the stated weights and under every alternative
for arm, r in rows.items():
    pairs = []
    for run in runs[arm]:
        tfrun = next((t for t in runs['tf'] if t['seed'] == run['seed']), None)
        if tfrun is None:
            continue
        g = sorted(run['copy'], key=int)
        a_row = dict(r, bpb=run['test']['macro'], recall=mean([run['copy'][k]['second_acc'] for k in g]))
        t_row = dict(tf, bpb=tfrun['test']['macro'], recall=mean([tfrun['copy'][k]['second_acc'] for k in g]))
        for x in RAW.get(SPEED_ARM.get(arm, arm), []):
            if 'error' in x:
                continue
            ix = indices(a_row, spd(x), t_row, tf)
            pairs.append(dict(seed=run['seed'], timing_pass=x['timing_pass'], timing_round=x['timing_round'],
                              combined=combine(ix, WEIGHTS), alt={k: combine(ix, w) for k, (lab, w) in SCHEMES.items()}))
    r['combined_pairs'] = [{k: v for k, v in p.items() if k != 'alt'} for p in pairs]
    r['combined_spread'] = [min(p['combined'] for p in pairs), max(p['combined'] for p in pairs)]
    r['alt_spread'] = {k: [min(p['alt'][k] for p in pairs), max(p['alt'][k] for p in pairs)] for k in SCHEMES}


def overlap(lower, upper):
    return lower[1] >= upper[0]


order = sorted(rows, key=lambda a: (not rows[a]['ranked'], -rows[a]['combined']))
ranked = [a for a in order if rows[a]['ranked']]
for i, a in enumerate(ranked):
    rows[a]['rank'] = i + 1
    rows[a]['level_with_above'] = bool(i) and overlap(rows[a]['combined_spread'], rows[ranked[i - 1]]['combined_spread'])
    rows[a]['rank_label'] = f"{i + 1}{'≈' if rows[a]['level_with_above'] else ''}"
alternatives = {}
for k, (lab, w) in SCHEMES.items():
    o = sorted(ranked, key=lambda a: -rows[a]['alt_combined'][k])
    for a in ranked:
        rows[a].setdefault('alt_rank', {})[k] = o.index(a) + 1
    lead, ru = o[0], o[1]
    alternatives[k] = dict(label=lab, weights=w, leader=lead, runner_up=ru,
                           level=overlap(rows[ru]['alt_spread'][k], rows[lead]['alt_spread'][k]))
for a in ranked:
    rk = [rows[a]['rank']] + list(rows[a]['alt_rank'].values())
    rows[a]['rank_range'] = [min(rk), max(rk)]
    rows[a]['top3_count'] = sum(x <= 3 for x in rk)
    rows[a]['top4_count'] = sum(x <= 4 for x in rk)

# what-if: the state-gated store scored with its soft read instead of its gate as designed
whatif = {}
if 'om2g' in rows and 'bpb_soft' in rows['om2g']:
    g = rows['om2g']
    ix = dict(g['index'], quality=100 * 2 ** (tf['bpb'] - g['bpb_soft']))
    c = combine(ix, WEIGHTS)
    whatif['om2g_soft'] = dict(combined=c, rank=1 + sum(rows[a]['combined'] > c for a in ranked if a != 'om2g'))

# control: the Transformer with its attention written out in ordinary ops (no fused kernel); its factors are ratios of
# second-pass medians (same pass as the Transformer's second-pass rounds), applied to the Transformer's headline values and
# scored against the same reference as every row. Other attention-using rows are not re-timed.
controls = {}
if 'tfplain' in SPEED:
    p_tf, p_pl = SPEED['tf']['pass2'], SPEED['tfplain']['pass2']
    fac = dict(train=p_pl['train_s_per_step'] / p_tf['train_s_per_step'], decode=p_pl['decode_ms_median'] / p_tf['decode_ms_median'],
               prefill=p_tf['prefill_tok_s'] / p_pl['prefill_tok_s'],
               train_memory=SPEED['tfplain']['train_model_rss_mib'] / SPEED['tf']['train_model_rss_mib'])
    s = dict(tf, train_s_per_step=tf['train_s_per_step'] * fac['train'], decode_ms=tf['decode_ms'] * fac['decode'],
             prefill=tf['prefill'] / fac['prefill'], train_rss=tf['train_rss'] * fac['train_memory'])
    c = combine(indices(tf, s, tf, tf), WEIGHTS)
    eq = None
    for p in (SPEED_PATH.parent / 'checks' / 'tfplain_equivalence.json', SPEED_PATH.parent / 'tfplain_equivalence.json'):
        if p.exists():
            eq = json.loads(p.read_text())
    dec_tf = [x['decode_ms_median'] for x in RAW.get('tf', []) if x.get('timing_pass') == 2 and 'error' not in x]
    dec_pl = [x['decode_ms_median'] for x in RAW.get('tfplain', []) if 'error' not in x]
    near = [rows[a]['rank'] for a in ranked if a != 'tf' and rows[a]['combined_spread'][0] <= c <= rows[a]['combined_spread'][1]]
    controls['tfplain'] = dict(factors=fac, combined=c, rank_if_plain=1 + sum(rows[a]['combined'] > c for a in ranked if a != 'tf'),
                               within_spread_of_ranks=near, decode_ms_rounds_tf_pass2=dec_tf, decode_ms_rounds_plain=dec_pl,
                               equivalence=eq)
jit = dict(decode_max=max(r['decode_ms_range'][1] / r['decode_ms_range'][0] - 1 for r in rows.values() if r.get('decode_ms_range')),
           train_max=max(r['train_s_range'][1] / r['train_s_range'][0] - 1 for r in rows.values() if r.get('train_s_range')))
(OUT / 'index.json').write_text(json.dumps(dict(
    weights=WEIGHTS, labels=LABELS, alternatives=alternatives, gate_quality_index=GATE, common_dev_target_bpb=target,
    common_target_arm=target_arm, tf_final_dev_bpb=tf_target, controls=controls, whatif=whatif, timing_jitter=jit,
    speed_machine=dict(cpu=SPEED_DOC['cpu'], threads=1, time=SPEED_DOC['time'], timing=SPEED_DOC.get('timing')),
    order=order, rows={a: {k: v for k, v in r.items() if k != 'curves'} for a, r in rows.items()}), indent=1) + '\n')
lab = lambda k: LABELS.get(k, k)
L = [f'| rank | model | family | **combined** (spread) | rank under {NWAYS} ways of counting | quality (bpb) | repeat-span accuracy | '
     'training speed (s/step) | decode (ms/token) | prefill (tok/s) | inference memory: fp32 weights + state at 256 tokens (state) | '
     'training memory above baseline | minutes to common dev target |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for arm in order:
    r, i = rows[arm], rows[arm]['index']
    rr = f"{r['rank_range'][0]}-{r['rank_range'][1]}" if r['ranked'] else '-'
    L.append(f"| {r.get('rank_label', '-')} | {r['name']} | {r['family']} | **{r['combined']:.1f}** ({r['combined_spread'][0]:.1f}-"
             f"{r['combined_spread'][1]:.1f}) | {rr} | {i['quality']:.0f} ({r['bpb']:.3f}) | "
             f"{i['recall']:.0f} ({r['recall'] * 100:.0f}%) | {i['train_speed']:.0f} ({r['train_s_per_step']:.2f}) | "
             f"{i['decode_speed']:.0f} ({r['decode_ms']:.1f}) | {i['prefill_speed']:.0f} ({r['prefill'] / 1000:.1f}k) | {i['inference_memory']:.0f} "
             f"({r['inference_bytes'] / 2 ** 20:.2f} MiB; {r['state_bytes'] / 1024:.0f} KiB) | {i['train_memory']:.0f} ({r['train_rss']:.0f} MiB) | "
             f"{r['train_min_to_common_target']:.1f} |")
L += ['', 'Each column is an index with the Transformer = 100 (higher is better), raw value in brackets. Combined = weighted '
      'geometric mean of the columns: ' + ', '.join(f'{lab(k)} {w:.0%}' for k, w in WEIGHTS.items()) + '. The spread is the lowest '
      'and highest combined score over the seed x timing-round pairs (each round\'s own timings against the Transformer\'s medians; '
      'each seed\'s quality against the Transformer\'s at that seed). "≈" marks a row whose spread overlaps the row above (not '
      f'separated from it). All models with a quality index of at least {GATE:.0f} are ranked. Speed and memory: one machine '
      f'({SPEED_DOC["cpu"]}), one thread, every arm in its own process; every timing is the median over the rounds of two passes. '
      f'"Rank under {NWAYS} ways of counting" is the best and worst rank under the stated weights and {NWAYS - 1} alternatives: ' + '; '.join(
          f"{lb} ({', '.join(f'{lab(k)} {w / sum(ws.values()):.0%}' for k, w in ws.items() if w)})" for k, (lb, ws) in SCHEMES.items()) + '.']
if controls:
    c = controls['tfplain']
    L += ['', f"Control: the Transformer with its attention written out in ordinary PyTorch ops instead of the fused kernel "
              f"takes {c['factors']['train']:.2f}x as long per training step, {c['factors']['train_memory']:.2f}x the training memory, "
              f"{c['factors']['decode']:.2f}x the decode time (not resolved: its decode rounds overlap the Transformer's) and "
              f"{c['factors']['prefill']:.2f}x the prefill time (second-pass medians), and would score {c['combined']:.1f} against "
              f"the same reference."]
(OUT / 'INDEX.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
