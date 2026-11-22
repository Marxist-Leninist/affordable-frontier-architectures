#!/usr/bin/env python3
"""P039-L2 positions check (22 Nov 2026). Compares the v1 attention AR runs (P045's learned absolute positions, runs/)
with the v2 rotary AR runs (runs_v2/) on the SAME 4,768 cases, using summarize.py's document-level bootstrap exactly
(2,000 replicates, documents resampled within each domain, rng seed 20261122). It first reproduces summary.json's v2
numbers to 1e-9, then reports v1 and the paired differences. Writes results_v2/positions_check.json (aggregates only)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
P1, P2 = ROOT / 'private', ROOT / 'private_v2'
SPACE, REPS, SEEDS = 32, 2000, (17, 43)
m1, m2 = np.load(P1 / 'case_meta.npz'), np.load(P2 / 'case_meta.npz')
assert m1.files == m2.files and all(np.array_equal(m1[k], m2[k]) for k in m1.files), 'v1 and v2 cases differ'
dom, doc, Y = m2['dom'], m2['doc'], m2['target'].astype(np.int64)
docs = sorted(set(zip(dom.tolist(), doc.tolist())))
doc_index = {d: i for i, d in enumerate(docs)}
case_doc = np.array([doc_index[(a, b)] for a, b in zip(dom.tolist(), doc.tolist())])
doc_dom = np.array([d for d, _ in docs])
DOMS = ['article', 'fiction', 'dialogue']
rng = np.random.default_rng(20261122)
BOOT = {d: rng.integers(0, int((doc_dom == d).sum()), size=(REPS, int((doc_dom == d).sum()))) for d in DOMS}
DOM_DOCS = {d: np.where(doc_dom == d)[0] for d in DOMS}


def per_doc(v):
    s = np.zeros(len(docs)); n = np.zeros(len(docs)); ok = ~np.isnan(v)
    np.add.at(s, case_doc[ok], v[ok]); np.add.at(n, case_doc[ok], 1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def macro_doc(dv):
    return float(np.mean([np.nanmean(dv[DOM_DOCS[d]]) for d in DOMS]))


def boot(dv):
    out = np.zeros(REPS)
    for d in DOMS:
        out += np.nanmean(dv[DOM_DOCS[d]][BOOT[d]], 1)
    return out / len(DOMS)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def case_gen(P):
    """Exactly summarize.py's per-case generation metrics."""
    P = P.astype(np.int64); eq = P == Y; ns = Y != SPACE
    return {'byte_acc': eq.mean(1),
            'nonspace_acc': np.where(ns.sum(1) > 0, (eq & ns).sum(1) / np.maximum(ns.sum(1), 1), np.nan),
            'exact': eq.all(1).astype(float), 'space_pred_rate': (P == SPACE).mean(1),
            'degenerate': np.array([np.bincount(r, minlength=256).max() >= 12 for r in P], float)}


def arm(priv, name):
    zs = [np.load(priv / f'cases_{name}_{s}.npz') for s in SEEDS]
    mets = [case_gen(z['pred_greedy']) for z in zs]
    v = {'bits': np.mean([z['nll_ar'] for z in zs], 0)}
    for k in mets[0]:
        v[k] = np.mean([m[k] for m in mets], 0) if k != 'nonspace_acc' else np.nanmean(np.stack([m[k] for m in mets]), 0)
    return v


def summ(v):
    return {k: {'macro': macro_doc(per_doc(x)), 'ci95': ci(boot(per_doc(x)))} for k, x in v.items()}


def paired(a, b, lower_better):
    out = {}
    for k in ('bits', 'nonspace_acc', 'exact'):
        d = boot(per_doc(a[k])) - boot(per_doc(b[k]))
        out[k] = {'diff': macro_doc(per_doc(a[k])) - macro_doc(per_doc(b[k])), 'ci95': ci(d),
                  'p_first_better': float(((d < 0) if k == 'bits' else (d > 0)).mean())}
    return out


v1 = arm(P1, 'attn_ar'); a2 = arm(P2, 'attn_ar'); h2 = arm(P2, 'hybrid_ar')
S = json.loads((ROOT / 'results_v2' / 'summary.json').read_text())
chk = {}
for name, v in (('attn_ar', a2), ('hybrid_ar', h2)):
    want = S['arms'][name]; got = summ(v)
    ok = abs(got['bits']['macro'] - want['bits']['macro']) < 1e-9 and all(abs(x - y) < 1e-9 for x, y in zip(got['bits']['ci95'], want['bits']['ci95']))
    for k in ('nonspace_acc', 'exact', 'space_pred_rate', 'degenerate', 'byte_acc'):
        ok = ok and abs(got[k]['macro'] - want['decoders']['greedy'][k]['macro']) < 1e-9
    chk[name] = ok
out = {'what': 'v1 = learned absolute positions (P045 parity), v2 = rotary positions; same 4,768 cases, same bootstrap as summarize.py',
       'cases': int(len(Y)), 'documents': len(docs), 'seeds': list(SEEDS), 'reproduces_summary_json_v2_bits': chk,
       'v1_abs_attn_ar': summ(v1), 'v2_rope_attn_ar': summ(a2), 'v2_rope_hybrid_ar': summ(h2),
       'paired_v1abs_attn_minus_v2rope_attn': paired(v1, a2, True),
       'paired_v1abs_attn_minus_v2rope_hybrid': paired(v1, h2, True)}
(ROOT / 'results_v2' / 'positions_check.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
