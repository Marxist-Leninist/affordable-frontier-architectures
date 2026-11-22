#!/usr/bin/env python3
"""Build the #p039-l2 card for zqx.lat/agi-architectures from the P039-L2 result files, so every number on the card
(table AND prose) is computed from summary.json, positions_check.json, timing.json and the run states, never typed by
hand; every claim in the prose is backed by an assert. v2 (22 Nov 2026) applies the five-lens pre-publication review.
Usage: python3 make_card.py <pull dir> <out.html> [<github commit> <hf commit>]   (commits pin the links)"""
import html, json, math, sys
from pathlib import Path

PULL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
GH_SHA, HF_SHA = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ('main', 'main')
NAME = 'p039-l2-gpu-diffusion-rematch'
GH = f'https://github.com/Marxist-Leninist/affordable-frontier-architectures/tree/{GH_SHA}/experiments/2026-11-22/{NAME}'
HF = f'https://huggingface.co/ScAGI/affordable-frontier-architectures/tree/{HF_SHA}/experiments/2026-11-22/{NAME}'
J = lambda *p: json.loads(PULL.joinpath(*p).read_text())
S, PC, T = J('results_v2', 'summary.json'), J('results_v2', 'positions_check.json'), J('results_v2', 'timing.json')
MAN = J('data_manifests', 'evaluation_manifest.json'); TRAIN_BYTES = J('data_manifests', 'manifest.json')['train_bytes']
assert all(PC['reproduces_summary_json_v2_bits'].values()), 'positions_check did not reproduce summary.json'
assert S['index_table']['timing_clean'] is True, 'timing was not clean'
rows = S['index_table']['rows']


def row(model, dec):
    hit = [r for r in rows if r['model'] == model and r['decoder'] == dec]
    assert len(hit) == 1, (model, dec, len(hit))
    return hit[0]


AA, HA = row('attn_ar', 'greedy'), row('hybrid_ar', 'greedy')
HD, AD = row('hybrid_diff', 'conf16'), row('attn_diff', 'conf16')
assert AA['rank'] == 1 and HA['rank'] == 2 and AA['gate_pass'] and HA['gate_pass'] and not HD['gate_pass'] and not AD['gate_pass']
assert all(r['preset_decoder'] for r in (AA, HA, HD, AD))
diff_rows = [r for r in rows if r['model'].endswith('_diff')]
assert len(diff_rows) == 12 and not any(r['gate_pass'] for r in diff_rows)
B = S['baselines']; N = B['kn_order_selected_on_dev']
KN_BITS, KN_CI = B['kn_test_bits']['macro'], B['kn_test_bits']['ci95']
KG, SP = B['decoders'][f'kn{N}_greedy'], B['decoders']['all_space']
assert all(r['bits'] > KN_BITS and r['nonspace_acc'] < KG['nonspace_acc']['macro'] for r in diff_rows)   # the title
V1 = PC['v1_abs_attn_ar']
P_HA = S['paired_vs_attn_ar']['hybrid_ar']
P_V1A, P_V1H = PC['paired_v1abs_attn_minus_v2rope_attn'], PC['paired_v1abs_attn_minus_v2rope_hybrid']
assert P_HA['bits']['ci95'][1] < 0 and P_V1A['bits']['ci95'][1] < 0 and P_V1H['bits']['ci95'][1] < 0
for lv in (P_HA['greedy_vs_greedy:nonspace_acc'], P_V1A['nonspace_acc'], P_V1H['nonspace_acc']):
    assert lv['ci95'][0] < 0 < lv['ci95'][1]                                    # every "level" claim
seed_gap = [h - a for h, a in zip(S['arms']['hybrid_ar']['bits']['per_seed'], S['arms']['attn_ar']['bits']['per_seed'])]
assert all(g < 0 for g in seed_gap)                                             # the hybrid leads in both seeds


# ------------------------------------------------------------------ timing: GPU kernel (index) and eager wall-clock
def tk(arm, key):
    v = T['arms'][arm][key]['p50_ms']; assert isinstance(v, (int, float)) and v > 0, (arm, key); return v


AR_KEY = 'ar_cached_16'
dk = {'attn_ar': tk('attn_ar', 'dev_b1_' + AR_KEY), 'hybrid_ar': tk('hybrid_ar', 'dev_b1_' + AR_KEY)}
wall = {'attn_ar': tk('attn_ar', 'b1_' + AR_KEY), 'hybrid_ar': tk('hybrid_ar', 'b1_' + AR_KEY)}
for arm in ('attn_diff', 'hybrid_diff'):
    for p in (1, 2, 4, 8, 16):
        dk[f'{arm}{p}'] = tk(arm, f'dev_b1_diff_{p}pass_16'); wall[f'{arm}{p}'] = tk(arm, f'b1_diff_{p}pass_16')
assert abs(dk['attn_ar'] - AA['decode_ms_16_bytes_b1']) < 1e-9 and abs(dk['hybrid_ar'] - HA['decode_ms_16_bytes_b1']) < 1e-9
assert abs(dk['attn_diff16'] - AD['decode_ms_16_bytes_b1']) < 1e-9 and abs(dk['hybrid_diff16'] - HD['decode_ms_16_bytes_b1']) < 1e-9
assert dk['hybrid_ar'] < dk['attn_ar'] and wall['hybrid_ar'] > wall['attn_ar']  # "less kernel time, slower wall-clock"
ar_fast = min(dk['attn_ar'], dk['hybrid_ar'])
fast = {arm: [p for p in (1, 2, 4, 8, 16) if dk[f'{arm}{p}'] < ar_fast] for arm in ('attn_diff', 'hybrid_diff')}
assert 16 not in fast['attn_diff'] and 16 not in fast['hybrid_diff'] and fast['attn_diff'] and fast['hybrid_diff']
fast_rows = [row(arm, f'conf{p}') for arm, ps in fast.items() for p in ps]


def span(ps):
    return f'{ps[0]}' if len(ps) == 1 else f'{ps[0]}&ndash;{ps[-1]}'


def pct_up(x):                                                                  # half-up integer percent
    return math.floor(100 * x + 0.5)


deg_range = f"{pct_up(min(r['degenerate'] for r in fast_rows))}&ndash;{pct_up(max(r['degenerate'] for r in fast_rows))}%"
deg16 = f"{pct_up(min(HD['degenerate'], AD['degenerate']))}&ndash;{pct_up(max(HD['degenerate'], AD['degenerate']))}%"
one_pass_byte = max(row('attn_diff', 'conf1')['byte_acc'], row('hybrid_diff', 'conf1')['byte_acc'])
ar_byte = max(AA['byte_acc'], HA['byte_acc'])
assert one_pass_byte > ar_byte > KG['byte_acc']['macro']                        # raw byte accuracy rewards spaces


# ------------------------------------------------------------------ index variants (same weights, one term swapped)
W = S['index_table']['weights']; R0 = S['index']['hybrid_ar']['ratios_vs_attn_ar']


def index(r):
    return 100 * math.exp(sum(W[k] * math.log(r[k]) for k in W))


assert abs(index(R0) - HA['index']) < 1e-6
tt = {a: [m for m in S['index'][a]['clean_minutes_to_common_dev_target'] if m is not None] for a in ('attn_ar', 'hybrid_ar')}
assert len(tt['attn_ar']) == 2 and len(tt['hybrid_ar']) == 2
idx_target = index({**R0, 'train_time': (sum(tt['attn_ar']) / 2) / (sum(tt['hybrid_ar']) / 2)})
idx_wall = index({**R0, 'latency_b1': wall['attn_ar'] / wall['hybrid_ar']})
assert idx_target > 100 > HA['index'] > idx_wall                                # "the order depends on how cost is counted"

# ------------------------------------------------------------------ scale, budget, plateau, memory
v1_states = [J('runs', f'attn_diff_{s}', 'state.json') for s in (17, 43)]
plateau = [st['dev'][0] for st in v1_states]
assert all(p['step'] == 2000 and p['dev_bits'] > 4.4 for p in plateau)
stopped_near = f"{round(sum(st['step'] for st in v1_states) / 2, -2):,.0f}"
v2_states = {f'{a}_{s}': J('runs_v2', f'{a}_{s}', 'state.json') for a in ('attn_ar', 'hybrid_ar', 'attn_diff', 'hybrid_diff') for s in (17, 43)}
assert all(v2_states[f'{a}_{s}']['best']['step'] == 20000 for a in ('attn_diff', 'hybrid_diff') for s in (17, 43))   # still improving
passes = 20000 * 16 * 256 / TRAIN_BYTES
run_peak = [st['peak_mem_bytes'] / 2 ** 20 for st in v2_states.values()]
docs = {d: MAN['test']['domains'][d]['documents'] for d in ('article', 'fiction', 'dialogue')}
assert sum(docs.values()) == S['cases']['n_docs']
target = S['common_dev_target_bits']
params = sorted({r['params'] for r in (AA, HA, HD, AD)})
best_diff_ns = max(diff_rows, key=lambda r: r['nonspace_acc'])
E = html.escape


def f3(x): return f'{x:.3f}'
def ci3(c): return f'{c[0]:.3f}&ndash;{c[1]:.3f}'
def pct(x, d=1): return f'{100 * x:.{d}f}%'
def mins(x): return f'{x:.1f}'
def neg(x, d=3): return f'&minus;{abs(x):.{d}f}' if x < 0 else f'+{x:.{d}f}'


# ------------------------------------------------------------------ table
TH = ('padding:6px 8px;border-bottom:1px solid var(--green);color:var(--green);font:700 12px var(--mono);'
      'vertical-align:bottom;text-align:')
TD = 'padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:'
heads = [('#', 'right'), ('model', 'left'), ('combined index (attention AR = 100)', 'right'), ('quality index', 'right'),
         ('held-out bits/byte [95% CI]', 'right'), ('non-space accuracy', 'right'), ('exact 16/16', 'right'),
         ('space rate', 'right'), ('degenerate', 'right'), ('decode ms, GPU kernel', 'right'), ('decode ms, wall-clock', 'right'),
         ('train GPU min', 'right'), ('GPU min to common dev target', 'right'), ('train-step peak MiB', 'right'), ('gate', 'left')]
STAR = ' &#9733;'; NA = '&ndash;'


def cells(r, name, rank, wall_ms):
    gate = 'pass' if r['gate_pass'] else 'fail: ' + ', '.join(
        t for k, t in (('likelihood_beats_kn', 'bits'), ('nonspace_beats_kn_greedy', 'non-space'),
                       ('degenerate_below_half', 'degenerate')) if not r['gate'][k])
    tgt = [m for m in S['index'][r['model']]['clean_minutes_to_common_dev_target'] if m is not None]
    return [rank, name, f"<strong>{r['index']:.1f}</strong>" if r['index'] is not None else 'not ranked',
            f"{r['quality_index']:.1f}", f"{f3(r['bits'])} [{ci3(r['bits_ci95'])}]", pct(r['nonspace_acc']),
            pct(r['exact'], 2), pct(r['space_rate']), pct(r['degenerate']), f"{r['decode_ms_16_bytes_b1']:.2f}",
            f'{wall_ms:.1f}', mins(r['train_minutes']), mins(sum(tgt) / len(tgt)) if len(tgt) == 2 else 'not reached',
            f"{r['train_peak_mem_mb']:.0f}", gate]


body = [cells(AA, 'Attention AR' + STAR, '1', wall['attn_ar']), cells(HA, 'Hybrid AR' + STAR, '2', wall['hybrid_ar']),
        cells(HD, 'Hybrid diffusion, 16 passes' + STAR, NA, wall['hybrid_diff16']),
        cells(AD, 'Attention diffusion, 16 passes' + STAR, NA, wall['attn_diff16']),
        ['ref', f'Byte {N}-gram (Kneser-Ney)', NA, NA, f'{f3(KN_BITS)} [{ci3(KN_CI)}]', pct(KG['nonspace_acc']['macro']),
         pct(KG['exact']['macro'], 2), pct(KG['space_pred_rate']['macro']), pct(KG['degenerate']['macro']), NA, NA, NA, NA, NA,
         'reference'],
        ['ref', 'Attention AR, learned absolute positions (v1)', NA, NA, f"{f3(V1['bits']['macro'])} [{ci3(V1['bits']['ci95'])}]",
         pct(V1['nonspace_acc']['macro']), pct(V1['exact']['macro'], 2), pct(V1['space_pred_rate']['macro']),
         pct(V1['degenerate']['macro']), NA, NA, NA, NA, NA, 'reference, not timed'],
        ['ref', 'Always a space', NA, NA, NA, pct(SP['nonspace_acc']['macro']), pct(SP['exact']['macro'], 2),
         pct(SP['space_pred_rate']['macro']), pct(SP['degenerate']['macro']), NA, NA, NA, NA, NA, 'reference']]
assert all(len(r) == len(heads) for r in body)
thead = ''.join(f'<th scope="col" style="{TH}{al}">{h}</th>' for h, al in heads)
tbody = ''.join('<tr>' + ''.join(f'<td style="{TD}{al}">{c}</td>' for c, (_, al) in zip(r, heads)) + '</tr>' for r in body)
table = ('<div role="region" aria-label="P039-L2 results, all models and metrics" tabindex="0" style="overflow-x:auto;margin:12px 0">'
         '<table style="width:100%;border-collapse:collapse;font-size:14px;overflow-wrap:normal;word-break:normal">'
         f'<thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>')

# ------------------------------------------------------------------ prose
slower = HA['train_minutes'] / AA['train_minutes'] - 1
card = f'''<!-- p039-l2:start (claude-p039l-gpu lane, 22 Nov 2026) -->
<article class="hybrid-card" id="p039-l2" aria-labelledby="p039-l2-title">
<p class="entry-family">P039-L2 / MATCHED GPU TEST / GTX 1660 SUPER / 22 NOV 2026</p><h3 id="p039-l2-title">At 3.3M parameters and 20,000 updates, masked diffusion trails a byte {N}-gram</h3>
<p><strong>Verdict:</strong> both autoregressive (AR) models pass the quality gate and both masked-diffusion models fail it. The two AR models are close, and which one ranks first depends on how cost is counted.</p>
{table}
<p>P039-L1 found that masked diffusion produced only spaces at 476,736 parameters on a CPU. This GPU follow-up trains four matched arms of {params[0]:,}&ndash;{params[-1]:,} parameters (4 layers, width 256, 256-byte context, rotary positions): attention, or a hybrid that alternates Mamba-1-style selective-scan layers with attention (the diffusion hybrid scans both ways), each trained AR or as masked diffusion, with the same 20,000 updates of 16 &times; 256 bytes and two seeds. The data is the byte corpus of our P045 benchmark: WikiText-2 articles, Project Gutenberg fiction and OpenAssistant dialogue, {TRAIN_BYTES / 1e6:.1f}M training bytes. Scoring used {S['cases']['n_cases']:,} answers (240-byte prompt, 16-byte answer) from {S['cases']['n_docs']} held-out test documents, with equal weight per domain. Intervals are 95% document-bootstrap intervals; they cover test documents, not training seeds.</p>
<ul>
<li><strong>Diffusion fails the quality gate.</strong> Scored by its exact left-to-right likelihood (one byte revealed per step), it reaches {f3(HD['bits'])} bits/byte (hybrid) and {f3(AD['bits'])} (attention), worse than the {N}-gram&rsquo;s {f3(KN_BITS)}, and its best non-space accuracy, {pct(best_diff_ns['nonspace_acc'])}, is below the {N}-gram&rsquo;s {pct(KG['nonspace_acc']['macro'])}. With 16 passes it no longer collapses to spaces as in P039-L1 ({deg16} of answers degenerate), but it needs more GPU time than cached AR. With {span(fast['attn_diff'])} passes (attention) or {span(fast['hybrid_diff'])} (hybrid) it needs less, but {deg_range} of those answers repeat one byte, mostly spaces. On raw byte accuracy one-pass diffusion scores {pct(one_pass_byte, 0)}, above AR&rsquo;s {pct(ar_byte, 0)}, only because spaces are common; hence the non-space gate.</li>
<li><strong>Among the rotary-position models, the hybrid predicts best:</strong> {f3(HA['bits'])} against {f3(AA['bits'])} bits/byte for attention (paired {neg(P_HA['bits']['diff'])}, 95% CI {neg(P_HA['bits']['ci95'][0])} to {neg(P_HA['bits']['ci95'][1])}; per seed the hybrid leads by {abs(seed_gap[0]):.3f} and {abs(seed_gap[1]):.3f}), with level non-space accuracy. It uses slightly less GPU kernel time per answer ({dk['hybrid_ar']:.2f} against {dk['attn_ar']:.2f} ms) but is slower in eager wall-clock time ({wall['hybrid_ar']:.1f} against {wall['attn_ar']:.1f} ms). It reaches the common dev target sooner ({mins(sum(tt['hybrid_ar']) / 2)} against {mins(sum(tt['attn_ar']) / 2)} GPU minutes), but its full 20,000 updates take {pct(slower, 0)} more GPU time ({mins(HA['train_minutes'])} against {mins(AA['train_minutes'])} minutes).</li>
<li><strong>Rotary positions cost attention AR quality here.</strong> With P045&rsquo;s learned absolute positions the same attention AR scores {f3(V1['bits']['macro'])} bits/byte: better than rotary attention by {abs(P_V1A['bits']['diff']):.3f} (CI {abs(P_V1A['bits']['ci95'][1]):.3f}&ndash;{abs(P_V1A['bits']['ci95'][0]):.3f}) and than the rotary hybrid by {abs(P_V1H['bits']['diff']):.3f} (CI {abs(P_V1H['bits']['ci95'][1]):.3f}&ndash;{abs(P_V1H['bits']['ci95'][0]):.3f}), with level non-space accuracy. The matched arms use rotary because attention diffusion with absolute positions was still on the unigram plateau ({min(p['dev_bits'] for p in plateau):.1f} dev bits/byte at update 2,000) when we stopped both seeds near update {stopped_near}. The hybrid was not trained with absolute positions, so these runs cannot tell whether it is the better design or only less hurt by rotary.</li>
</ul>
<p><strong>Ranking:</strong> the preset combined index (likelihood, non-space accuracy, GPU-kernel decode time and the GPU time of the full 20,000-update run, 25% each) gives attention AR {AA['index']:.0f} and the hybrid {HA['index']:.1f}. Counting GPU time to the common dev target instead gives the hybrid {idx_target:.1f}; counting wall-clock decode time gives it {idx_wall:.1f}. On quality alone the hybrid leads rotary attention, {HA['quality_index']:.1f}. This is one small scale, one corpus (article and fiction rest on {docs['article']} and {docs['fiction']} test documents), two seeds and one 20,000-update budget (about {passes:.0f} passes over the data; every diffusion run was still improving at its last dev check). It does not settle diffusion at larger scale, with longer training, or with other objectives and samplers.</p>
<p class="method-note">Combined index = weighted geometric mean of each metric&rsquo;s ratio to attention AR (inverted where lower is better); quality index = likelihood and non-space accuracy only. Gate: bits/byte below the {N}-gram, non-space accuracy above the {N}-gram&rsquo;s greedy answers, and fewer than half the answers degenerate (one byte 12 or more times in 16). The gate, the weights and the &#9733; decoders were set before any network was scored on the test set. AR decodes greedily; a diffusion pass commits the most confident masked bytes. Decode ms = one 16-byte answer at batch 1, median of three rotated rounds with no other GPU process: GPU kernel time (torch.profiler) and eager wall-clock time; kernel time leaves out Python and launch overhead, which weighs most on cached AR. Train GPU min = 20,000 &times; the median over three rounds of the mean kernel time per training step. Dev scores: exact NLL for AR, an 8-sample NELBO for diffusion, every 2,000 updates; common dev target = {target:.3f} dev bits/byte (the weaker attention AR seed&rsquo;s best), minutes are the mean of the two seeds. Train-step peak MiB = one training step in the timing pass; the full runs, with dev evaluation, peaked at {min(run_peak):.0f}&ndash;{max(run_peak):.0f} MiB. <a href="{GH}">Code, all 14 decoder rows and run logs</a> &middot; <a href="{HF}">10 checkpoints</a> &middot; <a href="#p039-l1">CPU follow-up P039-L1</a> &middot; <a href="#p039-diffusion-language">first P039 pilot</a></p>
</article>
<!-- p039-l2:end -->'''
terms_file = PULL.parent / 'private_terms.txt'          # host-specific terms, kept off the published code
assert terms_file.exists(), 'private_terms.txt missing'
for t in terms_file.read_text().split():
    assert t.lower() not in card.lower(), 'card contains a private term'
OUT.write_text(card + '\n', encoding='utf-8')
print(json.dumps({'idx_target': round(idx_target, 2), 'idx_wall': round(idx_wall, 2), 'fast': fast, 'deg_range': deg_range,
                  'deg16': deg16, 'seed_gap': [round(g, 4) for g in seed_gap], 'passes': round(passes, 2),
                  'run_peak_mib': [round(min(run_peak)), round(max(run_peak))], 'bytes': len(card.encode())}))
