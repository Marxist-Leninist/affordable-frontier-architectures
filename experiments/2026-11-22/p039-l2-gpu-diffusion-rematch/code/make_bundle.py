#!/usr/bin/env python3
"""Assemble the P039-L2 publication folder (GitHub + Hugging Face mirror) from the pulled result files.
Every number in README.md is computed from the result files, except the two pilot dev scores, which are copied from the
unpublished pilot state files. Ends with a privacy scan that fails on IP addresses,
tokens, key material, raw-text fields, email addresses and host-specific terms listed in an unpublished file.
v2 (22 Nov 2026) applies the five-lens pre-publication review. Usage: python3 make_bundle.py <publish dir>"""
import hashlib, json, math, os, re, shutil, sys
from pathlib import Path

PUB = Path(sys.argv[1]).resolve(); PULL = PUB / 'pull'; CK = PUB / 'ckpt'
STAGE = Path(os.environ.get('P039L_STAGE') or PUB.parent)      # where p039l.v2a.py / p039l.v2b.py are kept
NAME = 'p039-l2-gpu-diffusion-rematch'
OUT = PUB / 'bundle' / NAME
HF = f'https://huggingface.co/ScAGI/affordable-frontier-architectures/tree/main/experiments/2026-11-22/{NAME}'
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(src, rel):
    dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)


# ---------------------------------------------------------------- files
for f in ('p039l.py', 'p039l.v1.py', 'fused_scan.py', 'baselines.py', 'summarize.py', 'index_table_p039.py', 'positions_check.py'):
    put(PULL / f, f'code/{f}')
for f in ('p039l.v2a.py', 'p039l.v2b.py'):
    put(STAGE / f, f'code/{f}')
for f in ('make_card.py', 'make_bundle.py'):
    put(PUB / f, f'code/{f}')
pipe = (PULL / 'run_pipeline.sh').read_text()
pipe, n_sub = re.subn(r'^# P039-L v2 pipeline on the 1660 \(\d+\)\.', '# P039-L v2 pipeline on one GTX 1660 SUPER.', pipe, count=1, flags=re.M)
assert n_sub == 1                                               # the one comment that named the rental instance
(OUT / 'code').mkdir(exist_ok=True)
(OUT / 'code' / 'run_pipeline.sh').write_text(pipe)
put(PULL / 'results_v2' / 'REPORT.md', 'REPORT.md')
for p in sorted((PULL / 'results_v2').glob('*.json')):
    put(p, f'results/v2/{p.name}')
for p in sorted((PULL / 'results').glob('*.json')):
    put(p, f'results/v1/{p.name}')
for ver, d in (('v1', 'runs'), ('v2', 'runs_v2')):
    for p in sorted((PULL / d).glob('*/*.json')):
        put(p, f'runs/{ver}/{p.parent.name}/{p.name}')
put(PULL / 'logs' / 'pipeline.log', 'logs/pipeline.log')
for f in ('manifest.json', 'evaluation_manifest.json'):
    put(PULL / 'data_manifests' / f, f'data/{f}')

# ---------------------------------------------------------------- provenance + checkpoints
code_by_sha = {sha_file(OUT / 'code' / f): f for f in ('p039l.py', 'p039l.v1.py', 'p039l.v2a.py', 'p039l.v2b.py')}
prov = {'code_files': {v: k for k, v in code_by_sha.items()}, 'runs': {},
        'published_edits': ['code/run_pipeline.sh: one comment no longer names the rental instance (no code change)',
                            'code/summarize.py and code/index_table_p039.py: comments reworded and report wording corrected '
                            'after review; REPORT.md was regenerated with these files and summary.json came out byte-identical']}
ck_rows = []
for ver, d in (('v1', 'runs'), ('v2', 'runs_v2')):
    for sp in sorted((PULL / d).glob('*/state.json')):
        st = json.loads(sp.read_text()); run = sp.parent.name
        best = st.get('best') or {}
        rec = {'version': ver, 'positions': st.get('pos') or 'abs (learned table)', 'finished': st.get('finished'),
               'steps_done': st.get('step'), 'stopped_early': st.get('finished') is None,
               'code_file': code_by_sha.get(st['code_sha256'], 'unknown'), 'code_sha256': st['code_sha256'],
               'resumed': [{'at_step': r.get('at_step'), 'code_file': code_by_sha.get(r.get('code_sha256'), 'unknown')}
                           for r in st.get('resumed', [])],
               'fused_scan_sha256': st.get('fused_scan_sha256'), 'scan_impl': st.get('scan_impl'),
               'diff_loss': st.get('diff_loss'), 'params': st.get('params'), 'gpu': st.get('gpu'), 'torch': st.get('torch'),
               'best': best, 'final_sha256': st.get('final_sha256'), 'oom_retries': len(st.get('oom_retries', [])),
               'run_peak_mem_mib': round(st['peak_mem_bytes'] / 2 ** 20, 1) if st.get('peak_mem_bytes') else None}
        assert rec['code_file'] != 'unknown', (run, st['code_sha256'])
        ckp = CK / d / run / 'best.safetensors'
        if ckp.exists() and not rec['stopped_early']:
            h = sha_file(ckp); assert h == best.get('sha256'), (run, 'checkpoint hash differs from state.json')
            rec['checkpoint'] = f'checkpoints/{ver}/{run}/best.safetensors'
            ck_rows.append((rec['checkpoint'], ckp.stat().st_size, h, best.get('step'), best.get('dev_bits')))
        prov['runs'][f'{ver}/{run}'] = rec
(OUT / 'PROVENANCE.json').write_text(json.dumps(prov, indent=1) + '\n')
assert len(ck_rows) == 10, len(ck_rows)
CREDITS = ('Data credits: WikiText-2 (Merity et al., 2016; Wikipedia text, CC BY-SA 3.0), Project Gutenberg books (public '
           'domain in the US) and OpenAssistant Conversations (Köpf et al., 2023; Apache-2.0). The models were trained on this '
           'text; none of it is included here.')
ck_md = ['# Checkpoints', '', f'The ten best-dev checkpoints (safetensors, float32, no metadata) are on Hugging Face: [{HF}]({HF})',
         'under `checkpoints/`. They are not on GitHub. Final-step checkpoints and optimizer states are not published. The two',
         'v1 diffusion runs were stopped near update 2,600 and are not published either.', '',
         '| file | bytes | sha256 | best update | best dev bits/byte |', '|---|---|---|---|---|']
ck_md += [f'| `{p}` | {n:,} | `{h}` | {s:,} | {b:.4f} |' for p, n, h, s, b in ck_rows]
ck_md += ['', 'Dev bits/byte are exact NLL for AR runs and an 8-sample NELBO for diffusion runs.', '', CREDITS]
(OUT / 'CHECKPOINTS.md').write_text('\n'.join(ck_md) + '\n')

# ---------------------------------------------------------------- numbers for the README (same checks as make_card.py)
J = lambda *p: json.loads(PULL.joinpath(*p).read_text())
S, PC, T = J('results_v2', 'summary.json'), J('results_v2', 'positions_check.json'), J('results_v2', 'timing.json')
MAN = J('data_manifests', 'evaluation_manifest.json'); TRAIN_BYTES = J('data_manifests', 'manifest.json')['train_bytes']
assert all(PC['reproduces_summary_json_v2_bits'].values()) and S['index_table']['timing_clean'] is True
rows = S['index_table']['rows']
R = {(r['model'], r['decoder']): r for r in rows}
AA, HA, HD, AD = R['attn_ar', 'greedy'], R['hybrid_ar', 'greedy'], R['hybrid_diff', 'conf16'], R['attn_diff', 'conf16']
B = S['baselines']; N = B['kn_order_selected_on_dev']; KG, SP = B['decoders'][f'kn{N}_greedy'], B['decoders']['all_space']
V1 = PC['v1_abs_attn_ar']; P_HA = S['paired_vs_attn_ar']['hybrid_ar']
P1A, P1H = PC['paired_v1abs_attn_minus_v2rope_attn'], PC['paired_v1abs_attn_minus_v2rope_hybrid']
diff_rows = [r for r in rows if r['model'].endswith('_diff')]
assert all(r['bits'] > B['kn_test_bits']['macro'] and r['nonspace_acc'] < KG['nonspace_acc']['macro'] for r in diff_rows)
seed_gap = [h - a for h, a in zip(S['arms']['hybrid_ar']['bits']['per_seed'], S['arms']['attn_ar']['bits']['per_seed'])]
assert all(g < 0 for g in seed_gap)
tk = lambda arm, key: T['arms'][arm][key]['p50_ms']
dk = {a: tk(a, 'dev_b1_ar_cached_16') for a in ('attn_ar', 'hybrid_ar')}
wall = {a: tk(a, 'b1_ar_cached_16') for a in ('attn_ar', 'hybrid_ar')}
for arm in ('attn_diff', 'hybrid_diff'):
    for p in (1, 2, 4, 8, 16):
        dk[f'{arm}{p}'] = tk(arm, f'dev_b1_diff_{p}pass_16'); wall[f'{arm}{p}'] = tk(arm, f'b1_diff_{p}pass_16')
assert abs(dk['attn_ar'] - AA['decode_ms_16_bytes_b1']) < 1e-9 and abs(dk['hybrid_diff16'] - HD['decode_ms_16_bytes_b1']) < 1e-9
assert dk['hybrid_ar'] < dk['attn_ar'] and wall['hybrid_ar'] > wall['attn_ar']
ar_fast = min(dk['attn_ar'], dk['hybrid_ar'])
fast = {arm: [p for p in (1, 2, 4, 8, 16) if dk[f'{arm}{p}'] < ar_fast] for arm in ('attn_diff', 'hybrid_diff')}
fast_rows = [R[arm, f'conf{p}'] for arm, ps in fast.items() for p in ps]
pct_up = lambda x: math.floor(100 * x + 0.5)
W = S['index_table']['weights']; R0 = S['index']['hybrid_ar']['ratios_vs_attn_ar']
index = lambda r: 100 * math.exp(sum(W[k] * math.log(r[k]) for k in W))
assert abs(index(R0) - HA['index']) < 1e-6
tt = {a: [m for m in S['index'][a]['clean_minutes_to_common_dev_target'] if m is not None] for a in ('attn_ar', 'hybrid_ar')}
idx_target = index({**R0, 'train_time': (sum(tt['attn_ar']) / 2) / (sum(tt['hybrid_ar']) / 2)})
idx_wall = index({**R0, 'latency_b1': wall['attn_ar'] / wall['hybrid_ar']})
assert idx_target > 100 > HA['index'] > idx_wall
p45 = {r: J('results', f'p045protocol_{r}.json')['macro_bits'] for r in ('attn_ar_17', 'attn_ar_43')}
p45v2 = {r: J('results_v2', f'p045protocol_{r}.json')['macro_bits'] for r in ('attn_ar_17', 'attn_ar_43', 'hybrid_ar_17', 'hybrid_ar_43')}
v1d = [J('runs', f'attn_diff_{s}', 'state.json') for s in (17, 43)]
v1a = [J('runs', f'attn_ar_{s}', 'state.json') for s in (17, 43)]
v2s = {f'{a}_{s}': J('runs_v2', f'{a}_{s}', 'state.json') for a in ('attn_ar', 'hybrid_ar', 'attn_diff', 'hybrid_diff') for s in (17, 43)}
assert all(v2s[f'{a}_{s}']['best']['step'] == 20000 for a in ('attn_diff', 'hybrid_diff') for s in (17, 43))
run_peak = {a: max(v2s[f'{a}_{s}']['peak_mem_bytes'] for s in (17, 43)) / 2 ** 20 for a in ('attn_ar', 'hybrid_ar', 'attn_diff', 'hybrid_diff')}
docs = {d: MAN['test']['domains'][d]['documents'] for d in ('article', 'fiction', 'dialogue')}
kn_dev = B['kn_dev_bits_by_order']
passes = 20000 * 16 * 256 / TRAIN_BYTES
best_ns = max(diff_rows, key=lambda r: r['nonspace_acc'])
one_pass_byte = max(R['attn_diff', 'conf1']['byte_acc'], R['hybrid_diff', 'conf1']['byte_acc'])
smoke = J('results', 'smoke.json')


def f3(x): return f'{x:.3f}'
def ci3(c): return f'[{c[0]:.3f}, {c[1]:.3f}]'
def pct(x, d=1): return f'{100 * x:.{d}f}%'
def sgn(x, d=3): return f'{x:+.{d}f}'.replace('-', '−')
def span(ps): return f'{ps[0]}' if len(ps) == 1 else f'{ps[0]}–{ps[-1]}'


def mrow(rank, name, r, wall_ms):
    gate = 'pass' if r['gate_pass'] else 'fail (' + ', '.join(
        t for k, t in (('likelihood_beats_kn', 'bits'), ('nonspace_beats_kn_greedy', 'non-space'),
                       ('degenerate_below_half', 'degenerate')) if not r['gate'][k]) + ')'
    tg = [m for m in S['index'][r['model']]['clean_minutes_to_common_dev_target'] if m is not None]
    return (f"| {rank} | {name} | {'**%.1f**' % r['index'] if r['index'] is not None else 'not ranked'} | {r['quality_index']:.1f} | "
            f"{f3(r['bits'])} {ci3(r['bits_ci95'])} | {pct(r['nonspace_acc'])} | {pct(r['exact'], 2)} | {pct(r['space_rate'])} | "
            f"{pct(r['degenerate'])} | {r['decode_ms_16_bytes_b1']:.2f} | {wall_ms:.1f} | {r['train_minutes']:.1f} | "
            f"{(sum(tg) / len(tg)) if len(tg) == 2 else 'not reached'}{'' if len(tg) != 2 else ''} | "
            f"{r['dev_bits_gained_per_train_minute']:.4f} | {r['train_peak_mem_mb']:.0f} | {r['params']:,} | {gate} |")


def fixmin(s):     # one decimal for the minutes-to-target cell
    return re.sub(r'\| (\d+\.\d+) \| (\d+\.\d{4}) \|', lambda m: f'| {float(m.group(1)):.1f} | {m.group(2)} |', s, count=1)


table = ['| # | model | combined index (attention AR = 100) | quality index | held-out bits/byte [95% CI] | non-space acc | exact 16/16 | space rate | degenerate | decode ms, GPU kernel | decode ms, wall-clock | train GPU min | GPU min to common dev target | dev bits gained / train min | train-step peak MiB | params | gate |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
         fixmin(mrow(1, 'attention AR ★', AA, wall['attn_ar'])), fixmin(mrow(2, 'hybrid AR ★', HA, wall['hybrid_ar'])),
         mrow('–', 'hybrid diffusion, 16 passes ★', HD, wall['hybrid_diff16']),
         mrow('–', 'attention diffusion, 16 passes ★', AD, wall['attn_diff16']),
         f"| ref | byte {N}-gram (Kneser-Ney) | – | – | {f3(B['kn_test_bits']['macro'])} {ci3(B['kn_test_bits']['ci95'])} | {pct(KG['nonspace_acc']['macro'])} | {pct(KG['exact']['macro'], 2)} | {pct(KG['space_pred_rate']['macro'])} | {pct(KG['degenerate']['macro'])} | – | – | – | – | – | – | – | reference |",
         f"| ref | attention AR, learned absolute positions (v1) | – | – | {f3(V1['bits']['macro'])} {ci3(V1['bits']['ci95'])} | {pct(V1['nonspace_acc']['macro'])} | {pct(V1['exact']['macro'], 2)} | {pct(V1['space_pred_rate']['macro'])} | {pct(V1['degenerate']['macro'])} | – | – | – | – | – | – | 3,356,416 | reference, not timed |",
         f"| ref | always a space | – | – | – | {pct(SP['nonspace_acc']['macro'])} | {pct(SP['exact']['macro'], 2)} | {pct(SP['space_pred_rate']['macro'])} | {pct(SP['degenerate']['macro'])} | – | – | – | – | – | – | – | reference |"]
pos_tab = ['| model (same 4,768 cases) | positions | params | bits/byte [95% CI] | non-space acc | exact 16/16 |', '|---|---|---|---|---|---|']
for nm, k, pos, prm in (('attention AR (v1)', 'v1_abs_attn_ar', 'learned absolute table', '3,356,416'),
                        ('attention AR (v2)', 'v2_rope_attn_ar', 'rotary', f"{AA['params']:,}"),
                        ('hybrid AR (v2)', 'v2_rope_hybrid_ar', 'rotary', f"{HA['params']:,}")):
    x = PC[k]
    pos_tab.append(f"| {nm} | {pos} | {prm} | {f3(x['bits']['macro'])} {ci3(x['bits']['ci95'])} | {pct(x['nonspace_acc']['macro'])} | {pct(x['exact']['macro'], 2)} |")
fused_err = max(v for k in ('fused_typical', 'fused_extreme') for vv in (smoke.get(k) or {}).values()
                for v in (vv if isinstance(vv, list) else [vv]) if isinstance(v, (int, float)))
pilot = {'elbo': (3.424, 1077), 'mean': (3.418, 1069)}      # runs_test_elbo / runs_test_mean state.json (kept on the box)

readme = f"""# P039-L2: matched GPU rematch of P039 (autoregression vs masked diffusion, attention vs a Mamba-attention hybrid)

22 November 2026 · one NVIDIA GeForce GTX 1660 SUPER (6 GB) · PyTorch {v1a[0]['torch']} · site card:
[zqx.lat/agi-architectures/#p039-l2](https://zqx.lat/agi-architectures/#p039-l2)

P039 was a small GTX 1660 pilot comparing autoregressive (AR) and masked-diffusion byte models; its CPU follow-up P039-L1
(476,736 parameters, 3,200 updates) found that diffusion still produced only spaces. P039-L2 repeats the comparison on a GPU
at the scale of our P045 byte-model benchmark: about 3.29M parameters and 20,000 updates of 16 × 256 bytes per run, two
seeds, four matched arms, the same data for every arm.

**Verdict:** both AR models pass the quality gate and both masked-diffusion models fail it. The two AR models are close, and
which one ranks first depends on how cost is counted.

## Result

{chr(10).join(table)}

★ = decoder set before any network was scored on the test set (AR greedy; diffusion = 16 passes, each committing the most
confident masked byte). All 14 network × decoder rows (every diffusion sampler), the per-arm detail tables and the paired
differences are in [REPORT.md](REPORT.md). 95% intervals resample test documents, not training seeds.

- **Masked diffusion fails the quality gate.** Scored by its exact left-to-right likelihood (one byte revealed per step),
  it reaches {f3(HD['bits'])} bits/byte (hybrid) and {f3(AD['bits'])} (attention), worse than the byte {N}-gram's
  {f3(B['kn_test_bits']['macro'])} (NELBO: {S['arms']['hybrid_diff']['bits_elbo']['macro']:.3f} and
  {S['arms']['attn_diff']['bits_elbo']['macro']:.3f}). Its best non-space accuracy, {pct(best_ns['nonspace_acc'])}
  ({best_ns['model'].split('_')[0]}, decoder {best_ns['decoder']}), is below the {N}-gram's {pct(KG['nonspace_acc']['macro'])}.
  With 16 passes it no longer collapses to spaces as in P039-L1 ({pct_up(min(HD['degenerate'], AD['degenerate']))}–{pct_up(max(HD['degenerate'], AD['degenerate']))}% of
  answers degenerate), but it needs more GPU time than cached AR. With {span(fast['attn_diff'])} passes (attention) or
  {span(fast['hybrid_diff'])} (hybrid) it needs less, but {pct_up(min(r['degenerate'] for r in fast_rows))}–{pct_up(max(r['degenerate'] for r in fast_rows))}%
  of those answers repeat one byte, mostly spaces. On raw byte accuracy one-pass diffusion scores {pct(one_pass_byte, 0)},
  above AR's {pct(max(AA['byte_acc'], HA['byte_acc']), 0)}, only because spaces are common; that is why the gate uses non-space accuracy.
- **Among the rotary-position models, the hybrid predicts best:** {f3(HA['bits'])} against {f3(AA['bits'])} bits/byte
  for attention (paired {sgn(P_HA['bits']['diff'])}, 95% CI {sgn(P_HA['bits']['ci95'][0])} to {sgn(P_HA['bits']['ci95'][1])};
  per seed the hybrid leads by {abs(seed_gap[0]):.3f} and {abs(seed_gap[1]):.3f}). Non-space accuracy is level
  ({sgn(P_HA['greedy_vs_greedy:nonspace_acc']['diff'], 4)}, CI {sgn(P_HA['greedy_vs_greedy:nonspace_acc']['ci95'][0], 4)} to
  {sgn(P_HA['greedy_vs_greedy:nonspace_acc']['ci95'][1], 4)}). It uses slightly less GPU kernel time per answer
  ({dk['hybrid_ar']:.2f} against {dk['attn_ar']:.2f} ms) but is slower in eager wall-clock time ({wall['hybrid_ar']:.1f}
  against {wall['attn_ar']:.1f} ms). It reaches the common dev target sooner ({sum(tt['hybrid_ar']) / 2:.1f} against
  {sum(tt['attn_ar']) / 2:.1f} GPU minutes), but its full 20,000 updates take {100 * (HA['train_minutes'] / AA['train_minutes'] - 1):.0f}%
  more GPU time ({HA['train_minutes']:.1f} against {AA['train_minutes']:.1f} minutes).
- **Rotary positions cost attention AR quality here** (next section). The hybrid was not trained with absolute positions,
  so these runs cannot tell whether it is the better design or only less hurt by rotary.
- **Ranking:** the preset combined index gives attention AR 100 and the hybrid {HA['index']:.1f}. Counting GPU time to the
  common dev target instead of the full 20,000-update run gives the hybrid {idx_target:.1f}; counting wall-clock decode time
  instead of GPU kernel time gives it {idx_wall:.1f}. On quality alone the hybrid leads rotary attention ({HA['quality_index']:.1f}).
- **"Dev bits gained per train minute"** is highest for attention diffusion ({AD['dev_bits_gained_per_train_minute']:.4f}) only
  because its dev curve starts far higher; it is not comparable between AR and diffusion. No diffusion run reached the common
  dev target ({S['common_dev_target_bits']:.3f} bits/byte) within 20,000 updates.

## Absolute vs rotary positions

`code/positions_check.py` scores the v1 attention AR runs (P045's learned absolute position table, otherwise the same model,
optimizer, schedule, data and seeds) on the same {S['cases']['n_cases']:,} cases with the same document bootstrap as
`summarize.py`. It first reproduces summary.json's v2 numbers exactly (`reproduces_summary_json_v2_bits`).

{chr(10).join(pos_tab)}

- Absolute minus rotary, attention: {sgn(P1A['bits']['diff'])} bits/byte (95% CI {sgn(P1A['bits']['ci95'][0])} to
  {sgn(P1A['bits']['ci95'][1])}); non-space accuracy {sgn(P1A['nonspace_acc']['diff'], 4)} (CI
  {sgn(P1A['nonspace_acc']['ci95'][0], 4)} to {sgn(P1A['nonspace_acc']['ci95'][1], 4)}).
- Absolute attention minus rotary hybrid: {sgn(P1H['bits']['diff'])} bits/byte (CI {sgn(P1H['bits']['ci95'][0])} to
  {sgn(P1H['bits']['ci95'][1])}); non-space accuracy {sgn(P1H['nonspace_acc']['diff'], 4)} (CI
  {sgn(P1H['nonspace_acc']['ci95'][0], 4)} to {sgn(P1H['nonspace_acc']['ci95'][1], 4)}).
- On P045's own test protocol the ordering is the same: v1 attention {p45['attn_ar_17']:.3f} / {p45['attn_ar_43']:.3f}
  bits/byte (seeds 17 / 43), v2 attention {p45v2['attn_ar_17']:.3f} / {p45v2['attn_ar_43']:.3f}, v2 hybrid
  {p45v2['hybrid_ar_17']:.3f} / {p45v2['hybrid_ar_43']:.3f}.
- Why every matched arm uses rotary: with absolute positions, attention diffusion was at {v1d[0]['dev'][0]['dev_bits']:.2f}
  and {v1d[1]['dev'][0]['dev_bits']:.2f} dev bits/byte (NELBO) at update 2,000 (a byte unigram scores {kn_dev['1']:.2f} on
  dev), while the absolute-position AR runs were at {v1a[0]['dev'][0]['dev_bits']:.2f} and {v1a[1]['dev'][0]['dev_bits']:.2f}.
  We stopped the two v1 diffusion runs at updates {v1d[0]['step']:,} and {v1d[1]['step']:,}, still on that plateau, so it is
  not shown that they could never have left it. With rotary positions the same diffusion models were at
  {v2s['attn_diff_17']['dev'][0]['dev_bits']:.2f} (attention) and {v2s['hybrid_diff_17']['dev'][0]['dev_bits']:.2f} (hybrid)
  at update 2,000 (seed 17). Only attention diffusion was tried with absolute positions; the hybrid was not.

## Setup

- **Arms** (`ARMS` in `code/p039l.py`): `attn_ar` = 4 attention layers, causal; `hybrid_ar` = Mamba, attention, Mamba,
  attention, causal; `attn_diff` and `hybrid_diff` = the same stacks, bidirectional, trained as masked diffusion (the
  hybrid's Mamba layers average a forward and a backward scan).
- **Model:** width 256, 4 layers, 8 heads, FFN 1024, context 256 bytes, byte vocabulary plus a mask token (input only; the
  output head has 256 byte classes), rotary positions. Mamba mixer: Mamba-1-style selective SSM (input projection, causal
  depthwise conv k=4, selective scan with state 16 and dt rank 16, SiLU gate, output projection), inner width 308 so the
  mixer has 263,340 parameters against attention's 263,168 (+0.07%). The scan is a fused CUDA kernel (`code/fused_scan.py`)
  that matches an fp64 reference to within {math.ceil(fused_err * 1e8) / 1e8:.1e} including gradients (`results/v1/smoke.json`).
- **Training:** AdamW, lr 1e-3, 500 warm-up updates, cosine to 1e-4, weight decay 0.01 (none on A_log and D), gradient clip
  1.0, 20,000 updates × batch 16 × 256 bytes (about {passes:.1f} passes over the training bytes), seeds 17 and 43, dev check
  every 2,000 updates, best dev checkpoint kept. Diffusion: absorbing-state masked diffusion with the 1/t-weighted ELBO loss.
  Every diffusion run had its best dev score at the final update, so all of them were still improving.
- **Pilots:** before the v2 runs, two ~1,070-update pilots of rotary attention diffusion (seed 17, scored on dev only)
  compared the 1/t ELBO loss with a per-token mean loss: {pilot['elbo'][0]} against {pilot['mean'][0]} dev bits/byte at
  update 1,000, within 0.006 bits/byte on one seed. The ELBO, the proper likelihood bound, was kept. Note: `p039l.py`'s `--diff-loss` flag defaults to
  `mean`, so pass `--diff-loss elbo` to reproduce these runs.
- **Data:** the P045 byte corpus ({TRAIN_BYTES:,} training bytes; articles from WikiText-2, public-domain fiction from
  Project Gutenberg, dialogue from OpenAssistant), split by source entity with exact and near-duplicate exclusion between
  training and evaluation (`data/manifest.json`: "heuristic not proof against all paraphrase overlap"). Held out:
  {docs['article']} article, {docs['fiction']} fiction and {docs['dialogue']} dialogue test documents. File hashes are pinned in
  `PINS` in `code/p039l.py`. The corpus text is not redistributed here. {CREDITS}
- **Cases:** a 240-byte prompt and the next 16 bytes at every 64th position after each test window's 256-byte context
  ({S['cases']['n_cases']:,} cases: {S['cases']['by_domain']['article']} article, {S['cases']['by_domain']['fiction']} fiction,
  {S['cases']['by_domain']['dialogue']:,} dialogue). Scores are averaged per document, then per domain with equal weight, so
  dialogue's share of the cases does not dominate (but article and fiction rest on only {docs['article']} and
  {docs['fiction']} documents). 95% CIs come from 2,000 bootstrap replicates that resample documents within each domain;
  paired differences use the same replicates. Each case's score is the mean of the two seeds' scores (each model is scored
  on its own predictions), so the intervals do not include seed-to-seed variation.
- **Likelihood:** AR = exact next-byte NLL of the 16 answer bytes; diffusion = exact left-to-right NLL of the same network
  (one masked byte revealed at a time, left to right), with the NELBO also reported.
- **Decoders:** AR greedy with a KV/state cache. Diffusion `confN` = N passes that commit the most confident masked bytes
  first (N = 1, 2, 4, 8, 16); `l2r16` = 16 passes left to right.
- **Metrics:** non-space accuracy (correct non-space answer bytes over non-space answer bytes), exact 16/16, space rate,
  degenerate (one byte 12 or more times in 16).
- **Gate and index:** bits/byte below the byte Kneser-Ney n-gram (order {N}, chosen on dev from orders 1–12), non-space
  accuracy above the n-gram's greedy answers, and fewer than half of the answers degenerate. The gate and the four 25%
  weights were added at the owner's request at 04:01 UTC on 22 Nov, 38 minutes after the first v2 runs started and before any
  network was scored on the test set; an earlier draft of the index also weighted training memory. The ★ decoders were
  already fixed in that draft. Combined index = weighted geometric mean of four ratios to attention AR (likelihood,
  non-space accuracy, decode time, training time; 25% each; inverted where lower is better), only for rows that pass the
  gate. Quality index = likelihood and non-space accuracy only.
- **Timing:** batch 1, three rounds in rotated arm order with no other GPU process (checked before and after each
  measurement). Decode = one 16-byte answer after a 240-byte prompt. "GPU kernel" = summed CUDA kernel time from
  torch.profiler (the index uses this); "wall-clock" = eager PyTorch time with synchronisation. Train GPU min = 20,000 × the
  median over three rounds of the mean kernel time per training step. All timings are in `results/v2/timing.json`
  (`dev_*` keys = kernel time, the rest wall-clock).
- **Dev scores and training curves:** exact NLL for AR, an 8-sample NELBO (an upper bound) for diffusion, every 2,000
  updates. Common dev target = the worst AR run's best dev score ({S['common_dev_target_bits']:.3f}); GPU min to target is the
  mean over the two seeds. Dev bits gained per train minute = drop from the first dev check to the best one, per GPU minute
  between them. Train-step peak MiB = one training step in the timing pass; the full runs, with dev evaluation, peaked at
  {run_peak['attn_ar']:.0f} (attention AR), {run_peak['hybrid_ar']:.0f} (hybrid AR), {run_peak['attn_diff']:.0f} (attention
  diffusion) and {run_peak['hybrid_diff']:.0f} MiB (hybrid diffusion).

## Limits

- One small scale (3.3M parameters, 4 layers, 256-byte context), one corpus, two seeds, one 20,000-update budget (every
  diffusion run was still improving at its last dev check), one GPU type and one eager implementation.
- One diffusion formulation (absorbing-state masked diffusion, 1/t ELBO, confidence-ordered or left-to-right unmasking of a
  16-byte answer). Other objectives, samplers, longer training and larger models are untested here.
- GPU kernel time leaves out Python and kernel-launch overhead. At batch 1 that overhead weighs most on cached AR's
  single-byte steps: in eager wall-clock one answer takes {wall['attn_ar']:.1f} ms (attention AR), {wall['hybrid_ar']:.1f} ms
  (hybrid AR), {wall['attn_diff16']:.1f} ms (attention diffusion, 16 passes) and {wall['hybrid_diff16']:.1f} ms (hybrid
  diffusion, 16 passes), against {dk['attn_ar']:.2f}, {dk['hybrid_ar']:.2f}, {dk['attn_diff16']:.2f} and {dk['hybrid_diff16']:.2f} ms
  of kernel time. Kernel time therefore favours AR.
- The hybrid's quality lead is over rotary attention only; attention with learned absolute positions beats both rotary AR
  models on likelihood. The hybrid was not run with absolute positions, and the v1 runs were not part of the timing pass.
- v2 attn_ar seed 17 died in its first launch with a CUDA out-of-memory error while other runs shared the GPU, was
  restarted from scratch, stopped at update 424 and resumed from that checkpoint with `p039l.v2b.py` (PROVENANCE.json).
  No other v2 run was interrupted. The code versions differ only in evaluation batching, out-of-memory retries and timing;
  the model, loss and data are the same.
- The intervals cover test documents only, not training seeds or other data.

## Files

- `REPORT.md`: the full table (all 14 network × decoder rows), detail tables, paired differences.
- `results/v2/`: `summary.json` (everything in REPORT.md), `eval_*.json` (per run), `timing.json` (raw timings, three
  rounds), `p045protocol_*.json`, `baselines.json` (Kneser-Ney orders 1–12 on dev and test), `targets_descriptive.json`,
  `cases.json` (case counts and document ids, which are SHA-256 hashes), `positions_check.json`.
- `results/v1/`: the absolute-position runs, the v1 baselines and the smoke tests.
- `runs/{{v1,v2}}/<arm>_<seed>/state.json`: training curves (loss every 100 updates, dev every 2,000), training wall
  time, peak memory, code and checkpoint hashes; `done.json`.
- `code/`: `p039l.py` (final; it also trained v2 attn_ar seed 43), `p039l.v2a.py` and `p039l.v2b.py` (trained the other v2
  runs, see PROVENANCE.json), `p039l.v1.py` (v1), `fused_scan.py`, `baselines.py`, `summarize.py`, `index_table_p039.py`, `run_pipeline.sh`, `positions_check.py`,
  `make_card.py` (builds the site card from these files), `make_bundle.py` (builds this folder). Comments that named the
  owner were reworded; the report text written by `summarize.py` and `index_table_p039.py` was corrected after review
  (REPORT.md regenerated with the published files, summary.json byte-identical); one comment in `run_pipeline.sh` no
  longer names the rental instance. No training, evaluation or scoring code changed (PROVENANCE.json `published_edits`).
  `results/v1/smoke_v1_logspace.json` and `smoke_v1b_fused_abs.json` come from intermediate code drafts that are not
  included.
- `data/`: the P045 `manifest.json` and `evaluation_manifest.json` (counts and hashes, no text).
- `PROVENANCE.json`: which code trained and resumed each run, checkpoint SHA-256, full-run peak memory.
- `CHECKPOINTS.md`: the ten best-dev checkpoints on Hugging Face ({HF}).
- Not included: per-case predictions and answer bytes (they contain corpus text), the corpus itself, final-update
  checkpoints, the two dev-only pilots.

## Reproduce

```bash
cd code                                     # the scripts keep data/, runs*/, results*/ and private*/ next to themselves
ln -s /path/to/p045/data data               # the P045 corpus files with the hashes in PINS (p039l.py)
python3 p039l.py smoke                      # data hashes, fused scan vs fp64, parameter matching
for s in 17 43; do for a in attn_ar hybrid_ar attn_diff hybrid_diff; do
  python3 p039l.py train --arm $a --seed $s --out runs_v2 --pos rope --diff-loss elbo   # --diff-loss defaults to mean
done; done
python3 baselines.py results_v2 private_v2  # Kneser-Ney orders 1-12, order chosen on dev
python3 p039l.py evaluate --out runs_v2
python3 p039l.py p045 --out runs_v2 --only attn_ar_17,attn_ar_43,hybrid_ar_17,hybrid_ar_43
python3 p039l.py timing --out runs_v2 --rounds 3
python3 summarize.py _v2                    # writes results_v2/summary.json and REPORT.md
# v1 reference: train attn_ar with --pos abs --out runs, evaluate it with --out runs, then
python3 positions_check.py
```

`code/run_pipeline.sh` is the pipeline that produced these files (it runs the trainers concurrently on one GPU when memory
allows; timing is measured separately on a quiet GPU).
"""
(OUT / 'README.md').write_text(readme)

# ---------------------------------------------------------------- privacy scan + manifest
BAD = [(re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), 'ip address'), (re.compile(r'hf_[A-Za-z0-9]{20,}'), 'hf token'),
       (re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'), 'github token'),
       (re.compile(r'"(?:bytes|prefix)_b64"\s*:\s*"'), 'raw text field (data form)'),
       (re.compile(r'\bssh -[pio]\b|ssh:/{2}|BEGIN [A-Z ]*PRIVATE KEY|(?i:\bvault\b)|\.ssh[/]'), 'ssh / key material'),
       (re.compile(r'[\w.+-]+@[\w-]+\.[\w.]{2,}'), 'email')]
terms_file = PUB / 'private_terms.txt'                      # host-specific terms, never published
TERMS = [t.strip() for t in terms_file.read_text().splitlines() if t.strip()] if terms_file.exists() else []
assert TERMS, 'private_terms.txt missing'
hits = []
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        rel = str(p.relative_to(OUT))
        for ln in p.read_text(errors='replace').splitlines():
            for rx, what in BAD:
                for m in rx.finditer(ln):
                    if what == 'ip address' and not all(int(x) <= 255 for x in m.group(0).split('.')):
                        continue
                    hits.append((rel, what, ln[max(0, m.start() - 40):m.end() + 40]))
            low = ln.lower()
            for t in TERMS:
                if t.lower() in low:
                    hits.append((rel, 'private term', '(term not shown)'))
lines = [f'{sha_file(p)}  {p.relative_to(OUT)}' for p in sorted(OUT.rglob('*')) if p.is_file()]
(OUT / 'MANIFEST.sha256').write_text('\n'.join(lines) + '\n')
total = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
print(f'bundle {OUT}: {len(lines)} files, {total:,} bytes; checkpoints {len(ck_rows)}; privacy hits {len(hits)}; terms {len(TERMS)}')
for h in hits[:40]:
    print('  HIT', h)
sys.exit(1 if hits else 0)
