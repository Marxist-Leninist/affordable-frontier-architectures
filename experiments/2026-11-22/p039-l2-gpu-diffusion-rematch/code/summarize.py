#!/usr/bin/env python3
"""P039-L summary: seed means, document-level bootstrap CIs (resampling documents within each domain), paired
differences against attn_ar, a stated-weight aggregate index and a Markdown report. Reads results/ and private/;
writes results/summary.json and results/REPORT.md (aggregates only, no raw text)."""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
RES, PRIV, RUNS = ROOT / f'results{TAG}', ROOT / f'private{TAG}', ROOT / f'runs{TAG}'
ARMS = ['attn_ar', 'attn_diff', 'hybrid_ar', 'hybrid_diff']
SPACE = 32
REPS = 2000
WEIGHTS = {'likelihood': 0.25, 'nonspace_acc': 0.25, 'latency_b1': 0.25, 'train_time': 0.25}   # owner request, 22 Nov 2026: speed and training time count, after a quality gate


def load(p):
    return json.loads(Path(p).read_text())


meta = np.load(PRIV / 'case_meta.npz')
dom, doc, Y = meta['dom'], meta['doc'], meta['target'].astype(np.int64)
docs = sorted(set(zip(dom.tolist(), doc.tolist())))
doc_index = {d: i for i, d in enumerate(docs)}
case_doc = np.array([doc_index[(a, b)] for a, b in zip(dom.tolist(), doc.tolist())])
doc_dom = np.array([d for d, _ in docs])
DOMS = ['article', 'fiction', 'dialogue']
rng = np.random.default_rng(20261122)
BOOT = {d: rng.integers(0, int((doc_dom == d).sum()), size=(REPS, int((doc_dom == d).sum()))) for d in DOMS}
DOM_DOCS = {d: np.where(doc_dom == d)[0] for d in DOMS}


def per_doc(v):
    """Per-document mean of a per-case vector, NaN-aware."""
    s = np.zeros(len(docs)); n = np.zeros(len(docs))
    ok = ~np.isnan(v)
    np.add.at(s, case_doc[ok], v[ok]); np.add.at(n, case_doc[ok], 1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def macro_doc(dv):
    return float(np.mean([np.nanmean(dv[DOM_DOCS[d]]) for d in DOMS]))


def boot(dv):
    """Bootstrap distribution of the equal-domain macro of per-document values."""
    out = np.zeros(REPS)
    for d in DOMS:
        vals = dv[DOM_DOCS[d]][BOOT[d]]
        out += np.nanmean(vals, 1)
    return out / len(DOMS)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def case_gen(P):
    P = P.astype(np.int64); eq = P == Y; ns = Y != SPACE
    return {'byte_acc': eq.mean(1),
            'nonspace_acc': np.where(ns.sum(1) > 0, (eq & ns).sum(1) / np.maximum(ns.sum(1), 1), np.nan),
            'exact': eq.all(1).astype(float), 'space_pred_rate': (P == SPACE).mean(1),
            'degenerate': np.array([np.bincount(r, minlength=256).max() >= 12 for r in P], float)}


def seeds_of(arm):
    return sorted(int(p.stem.rsplit('_', 1)[1]) for p in RES.glob(f'eval_{arm}_*.json'))


summary = {'weights': WEIGHTS, 'bootstrap': {'reps': REPS, 'unit': 'documents resampled within each domain'},
           'arms': {}, 'baselines': {}, 'paired_vs_attn_ar': {}}
cases = load(RES / 'cases.json'); summary['cases'] = {k: cases[k] for k in ('n_cases', 'n_docs', 'by_domain', 'prompt', 'suffix', 'stride')}
pc = {}   # per-case seed-averaged vectors used for bootstrap: pc[arm][metric]
for arm in ARMS:
    seeds = seeds_of(arm)
    if not seeds:
        continue
    evs = [load(RES / f'eval_{arm}_{s}.json') for s in seeds]
    npz = [np.load(PRIV / f'cases_{arm}_{s}.npz') for s in seeds]
    states = [load(RUNS / f'{arm}_{s}' / 'state.json') for s in seeds]
    a = {'seeds': seeds, 'params': evs[0]['params'], 'best_steps': [e['best_step'] for e in evs],
         'best_dev_bits': [e['best_dev_bits'] for e in evs],
         'train_wall_s': [s_['train_s'] for s_ in states], 'train_peak_mem_bytes': [s_.get('peak_mem_bytes') for s_ in states],
         'foreign_gpu_seen': [sorted({p for f in s_['foreign_gpu'] for p in f['pids']}) for s_ in states],
         'scan_impl': [s_.get('scan_impl') for s_ in states], 'dev_curves': [s_['dev'] for s_ in states]}
    v = {}
    if arm.endswith('_ar'):
        v['bits'] = np.mean([z['nll_ar'] for z in npz], 0)
        decs = {'greedy': [z['pred_greedy'] for z in npz]}
        a['bits_kind'] = 'exact AR NLL'
    else:
        v['bits'] = np.mean([z['nll_l2r'] for z in npz], 0); v['bits_elbo'] = np.mean([z['nelbo'] for z in npz], 0)
        decs = {k.replace('pred_', ''): [z[k] for z in npz] for k in npz[0].files if k.startswith('pred_')}
        a['bits_kind'] = 'exact left-to-right NLL of the diffusion network (NELBO also reported)'
    a['bits'] = {'macro': macro_doc(per_doc(v['bits'])), 'ci95': ci(boot(per_doc(v['bits']))),
                 'per_seed': [e.get('suffix_bits', e.get('suffix_bits_l2r'))['macro'] for e in evs]}
    if 'bits_elbo' in v:
        a['bits_elbo'] = {'macro': macro_doc(per_doc(v['bits_elbo'])), 'ci95': ci(boot(per_doc(v['bits_elbo']))),
                          'per_seed': [e['suffix_bits_elbo']['macro'] for e in evs]}
    a['decoders'] = {}
    for name, Ps in decs.items():
        mets = [case_gen(P) for P in Ps]
        dd = {}
        for k in mets[0]:
            vec = np.mean([m[k] for m in mets], 0) if k != 'nonspace_acc' else np.nanmean(np.stack([m[k] for m in mets]), 0)
            dd[k] = {'macro': macro_doc(per_doc(vec)), 'per_seed': [macro_doc(per_doc(m[k])) for m in mets]}
            if k in ('byte_acc', 'nonspace_acc', 'exact'):
                dd[k]['ci95'] = ci(boot(per_doc(vec))); v[f'{name}:{k}'] = vec
        a['decoders'][name] = dd
    summary['arms'][arm] = a; pc[arm] = v

# baselines
if (RES / 'baselines.json').exists():
    b = load(RES / 'baselines.json'); bz = np.load(PRIV / 'cases_baselines.npz'); N = b['best_order_on_dev']
    bb = {'kn_order_selected_on_dev': N, 'kn_dev_bits_by_order': b['dev_bits'],
          'kn_test_bits': {'macro': macro_doc(per_doc(bz['nll_kn'])), 'ci95': ci(boot(per_doc(bz['nll_kn'])))},
          'decoders': {}}
    for name, P in (('all_space', np.full_like(Y, 32)), (f'kn{N}_greedy', bz['pred_kn_greedy'])):
        m = case_gen(P)
        bb['decoders'][name] = {k: {'macro': macro_doc(per_doc(m[k]))} for k in m}
    summary['baselines'] = bb

# paired differences vs attn_ar (same documents in every bootstrap replicate)
if 'attn_ar' in pc:
    ref = pc['attn_ar']
    for arm, v in pc.items():
        if arm == 'attn_ar':
            continue
        out = {}
        d = boot(per_doc(v['bits'])) - boot(per_doc(ref['bits']))
        out['bits'] = {'diff': macro_doc(per_doc(v['bits'])) - macro_doc(per_doc(ref['bits'])), 'ci95': ci(d), 'p_arm_better': float((d < 0).mean())}
        dec = 'greedy' if arm.endswith('_ar') else 'conf16'
        for k in ('nonspace_acc', 'exact', 'byte_acc'):
            if f'{dec}:{k}' in v:
                x, r_ = v[f'{dec}:{k}'], ref[f'greedy:{k}']
                d = boot(per_doc(x)) - boot(per_doc(r_))
                out[f'{dec}_vs_greedy:{k}'] = {'diff': macro_doc(per_doc(x)) - macro_doc(per_doc(r_)), 'ci95': ci(d), 'p_arm_better': float((d > 0).mean())}
        summary['paired_vs_attn_ar'][arm] = out

# timing and aggregate index
if (RES / 'timing.json').exists():
    t = load(RES / 'timing.json'); summary['timing'] = t
    def lat(arm, dec='conf16'):   # GPU device time (profiler) when recorded, else eager wall time
        r = t['arms'][arm]; pre = 'dev_' if 'dev_b1_ar_cached_16' in r or 'dev_b1_diff_16pass_16' in r else ''
        return r[f'{pre}b1_ar_cached_16']['p50_ms'] if arm.endswith('_ar') else r[f'{pre}b1_diff_{dec[4:]}pass_16']['p50_ms']
    def tstep(r):
        return (r.get('dev_train_step') or r['train_step'])['p50_ms']
    if 'attn_ar' in summary['arms']:
        base = summary['arms']['attn_ar']; tb = t['arms']['attn_ar']
        idx = {}
        bb = summary.get('baselines') or {}
        kn_bits = (bb.get('kn_test_bits') or {}).get('macro')
        kn_ns = max((v['nonspace_acc']['macro'] for k, v in (bb.get('decoders') or {}).items() if k.startswith('kn')), default=None)
        for arm, a in summary['arms'].items():
            dec = 'greedy' if arm.endswith('_ar') else 'conf16'
            ra = t['arms'][arm]; dd = a['decoders'][dec]
            gate = {'likelihood_beats_kn': kn_bits is not None and a['bits']['macro'] < kn_bits,
                    'nonspace_beats_kn_greedy': kn_ns is not None and dd['nonspace_acc']['macro'] > kn_ns,
                    'degenerate_below_half': dd['degenerate']['macro'] < 0.5}
            gate['pass'] = all(gate.values())
            steps = 20000
            clean_min = tstep(ra) * steps / 60000
            ratios = {'likelihood': base['bits']['macro'] / a['bits']['macro'],
                      'nonspace_acc': dd['nonspace_acc']['macro'] / base['decoders']['greedy']['nonspace_acc']['macro'],
                      'latency_b1': lat('attn_ar', 'greedy') / lat(arm, dec),
                      'train_time': tstep(tb) / tstep(ra)}
            agg = 100 * math.exp(sum(WEIGHTS[k] * math.log(ratios[k]) for k in WEIGHTS)) if gate['pass'] else None
            q = 100 * math.exp(0.5 * math.log(ratios['likelihood']) + 0.5 * math.log(max(ratios['nonspace_acc'], 1e-6)))
            gains = []
            for curve in a['dev_curves']:
                first = curve[0]; best = min(curve, key=lambda d: d['dev_bits'])
                mins = (best['step'] - first['step']) * tstep(ra) / 60000
                gains.append((first['dev_bits'] - best['dev_bits']) / mins if mins > 0 else 0.0)
            idx[arm] = {'decoder': dec, 'gate': gate, 'ratios_vs_attn_ar': ratios, 'aggregate_index': agg, 'quality_index': q,
                        'clean_train_minutes': clean_min, 'dev_bits_gained_per_clean_minute_after_step2000': sum(gains) / len(gains),
                        'train_peak_mem_bytes': ra['train_peak_mem_bytes']}
        target = max(min(d['dev_bits'] for d in c) for a in summary['arms'].values() for c in a['dev_curves'] if a['bits_kind'].startswith('exact AR'))
        for arm, a in summary['arms'].items():
            ra = t['arms'][arm]; hits = []
            for c in a['dev_curves']:
                reach = [d['step'] for d in c if d['dev_bits'] <= target]
                hits.append(reach[0] * tstep(ra) / 60000 if reach else None)
            idx[arm]['clean_minutes_to_common_dev_target'] = hits
        summary['common_dev_target_bits'] = target
        summary['index'] = idx

try:   # one ranked table of every model and metric (owner request, 22 Nov 2026); must never stop the report
    import index_table_p039
    HEAD, summary['index_table'] = index_table_p039.build(summary, WEIGHTS)
except Exception as e:
    import traceback
    HEAD, summary['index_table'] = [f'One-table view failed: {e!r}'], {'error': traceback.format_exc()}
(RES / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')

# Markdown report
CASES_TXT = ', '.join(f'{k} {v:,}' for k, v in summary['cases']['by_domain'].items())
DOCS_TXT = ', '.join(f'{d} {int((doc_dom == d).sum())}' for d in DOMS)
L = ['# P039-L2: matched GPU-scale rematch of P039 (GTX 1660 SUPER)', '',
     f"Cases: {summary['cases']['n_cases']:,} prompt-240 / suffix-16 completions ({CASES_TXT}) from {summary['cases']['n_docs']} held-out P045 test documents "
     f"({DOCS_TXT}), equal-domain then equal-document averages, 95% CIs from {REPS} document bootstrap replicates.", ''] + ['## All models and metrics, best first', ''] + HEAD + ['', '## Detail by arm and decoder', '',
     '| Arm | Params | Suffix bits/byte (lower better) | Decoder | Non-space acc | Byte acc | Exact 16/16 | Space rate | Degenerate |',
     '|---|---|---|---|---|---|---|---|---|']
for arm, a in summary['arms'].items():
    for dec, dd in a['decoders'].items():
        bits = f"{a['bits']['macro']:.3f} [{a['bits']['ci95'][0]:.3f}, {a['bits']['ci95'][1]:.3f}]"
        if 'bits_elbo' in a:
            bits += f" (NELBO {a['bits_elbo']['macro']:.3f})"
        L.append(f"| {arm} | {a['params']:,} | {bits} | {dec} | {dd['nonspace_acc']['macro']:.4f} | {dd['byte_acc']['macro']:.4f} | "
                 f"{dd['exact']['macro']:.4f} | {dd['space_pred_rate']['macro']:.3f} | {dd['degenerate']['macro']:.3f} |")
if summary['baselines']:
    bb = summary['baselines']
    L += ['', f"Baselines: KN byte {bb['kn_order_selected_on_dev']}-gram (order chosen on dev) suffix bits {bb['kn_test_bits']['macro']:.3f}; "
          + '; '.join(f"{k}: non-space {v['nonspace_acc']['macro']:.4f}, byte {v['byte_acc']['macro']:.4f}, exact {v['exact']['macro']:.4f}" for k, v in bb['decoders'].items())]
if 'index' in summary:
    L += ['', f'Aggregate index (attn_ar = 100; weights {WEIGHTS}):', '']
    for arm, r in summary['index'].items():
        agg = 'not ranked (gate failed: ' + ', '.join(k for k, v in r['gate'].items() if k != 'pass' and not v) + ')' if r['aggregate_index'] is None else f"{r['aggregate_index']:.1f}"
        L.append(f"- {arm} ({r['decoder']}): aggregate {agg}, quality-only {r['quality_index']:.1f}, {r['dev_bits_gained_per_clean_minute_after_step2000']:.4f} dev bits gained per clean training minute, clean training {r['clean_train_minutes']:.1f} min, ratios "
                 + ', '.join(f'{k} {v:.3f}' for k, v in r['ratios_vs_attn_ar'].items()))
L += ['', 'Paired differences vs attn_ar (arm minus attn_ar):', '']
for arm, out in summary['paired_vs_attn_ar'].items():
    L.append(f'- {arm}: ' + '; '.join(f"{k} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in out.items()))
(RES / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L))
