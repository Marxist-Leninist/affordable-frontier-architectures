#!/usr/bin/env python3
"""Render the site card (HTML fragment) for the one-index view of every OM2 + OM3 model.
Inputs: index.json (index_all.py), contrasts.json (contrasts_all.py), links.json, and a template (card_findings.html):
an at-a-glance paragraph, a line '<!-- list -->', then the findings as <li> lines. Its placeholders are filled from the
data by render_findings.py. All links must be present unless OM3_DRAFT=1.
Usage: make_site_card.py INDEX_JSON CONTRASTS_JSON LINKS_JSON TEMPLATE_HTML OUT_HTML"""
import html, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_findings import make_render  # noqa: E402

idx = json.loads(Path(sys.argv[1]).read_text())
C = json.loads(Path(sys.argv[2]).read_text())['contrasts']
links = json.loads(Path(sys.argv[3]).read_text()) if Path(sys.argv[3]).exists() else {}
need = ('om3_github', 'om3_hf', 'study_github', 'study_hf', 'kernel_github', 'kernel_hf')
missing = [k for k in need if not links.get(k)]
if missing and os.environ.get('OM3_DRAFT') != '1':
    raise SystemExit(f'missing links {missing}: publish first, or set OM3_DRAFT=1 for a draft')
glance, findings = make_render(idx, C, as_html=True)(Path(sys.argv[4]).read_text()).split('<!-- list -->')
rows, order = idx['rows'], idx['order']
# facts the template states in words: check them against the data so the prose cannot drift from it
_top4 = [a for a in order if rows[a].get('rank_range') and rows[a]['rank_range'][1] <= 4]
assert _top4 == ['m2'], f'template says only Mamba-2 is in the top four under every way of counting: {_top4}'
assert rows['m2']['rank'] == 3 and not rows['m2']['level_with_above'], 'template says Mamba-2 is a separated third'
assert rows['hyb']['rank'] == 2 and rows['hyb']['level_with_above'] and rows['tf']['rank'] == 1, 'template says tf and hyb are level at the top'
assert [a for a in order[3:5]] == ['om2pos', 'retnet'], 'template says v2 positive decay and RetNet follow'
assert all(rows[a]['bpb'] < rows['tf']['bpb'] for a in rows if a not in ('tf', 'retnet')) and rows['retnet']['bpb'] > rows['tf']['bpb']
nways = 1 + len(idx['alternatives'])
td = 'style="padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap"'
tl = 'style="padding:6px 8px;border-bottom:1px solid var(--line);text-align:left"'
th = 'style="padding:6px 8px;border-bottom:1px solid var(--green);text-align:right;color:var(--green);font:700 12px var(--mono)"'
thl = th.replace('text-align:right', 'text-align:left')
sub = 'style="display:block;color:var(--quiet);font-size:12px"'
head = ['#', 'model', 'combined (spread)', f'rank under {nways} ways of counting', 'quality (bits/byte)', 'repeat-span accuracy',
        'training (s/step)', 'decode (ms/token)', 'prefill (k tok/s)',
        'inference memory (fp32 weights + decode state at 256 tokens; state in brackets)', 'training memory (MiB above baseline)']
T = ['<div style="overflow-x:auto;margin:12px 0"><table style="width:100%;border-collapse:collapse;font-size:14px">',
     '<thead><tr>' + ''.join(f'<th {thl if i == 1 else th}>{h}</th>' for i, h in enumerate(head)) + '</tr></thead><tbody>']
for arm in order:
    r, i = rows[arm], rows[arm]['index']
    fam = f'<span {sub}>{html.escape(r["family"])}</span>' if r['family'] not in r['name'] else ''
    spr = f'<span {sub}>{r["combined_spread"][0]:.1f}&ndash;{r["combined_spread"][1]:.1f}</span>'
    T.append('<tr>' + ''.join([
        f'<td {td}>{r.get("rank_label", "-")}</td>', f'<td {tl}>{html.escape(r["name"])}{fam}</td>',
        f'<td {td}><strong>{r["combined"]:.1f}</strong>{spr}</td>',
        f'<td {td}>{"&ndash;".join(map(str, r["rank_range"])) if r["ranked"] else "-"}</td>',
        f'<td {td}>{i["quality"]:.0f} ({r["bpb"]:.3f})</td>', f'<td {td}>{i["recall"]:.0f} ({r["recall"] * 100:.0f}%)</td>',
        f'<td {td}>{i["train_speed"]:.0f} ({r["train_s_per_step"]:.2f})</td>',
        f'<td {td}>{i["decode_speed"]:.0f} ({r["decode_ms"]:.1f})</td>',
        f'<td {td}>{i["prefill_speed"]:.0f} ({r["prefill"] / 1000:.1f})</td>',
        f'<td {td}>{i["inference_memory"]:.0f} ({r["state_bytes"] / 1024:.0f} KiB)</td>',
        f'<td {td}>{i["train_memory"]:.0f} ({r["train_rss"]:.0f})</td>']) + '</tr>')
T.append('</tbody></table></div>')
W = idx['weights']
lab = idx['labels']
alts = '; '.join(html.escape(a['label']) for a in idx['alternatives'].values())
lk = lambda u, t: f'<a href="{u}">{t}</a>' if u else t
n = len(order)
sm = idx['speed_machine']
card = f"""<div class="project-card" id="om2-one-index" style="margin:0 0 22px">
<p class="entry-family">OM2 + OM3 · one index, all metrics · {n} matched models · 22 Nov 2026</p>
<h3>Which model is best overall, at a glance</h3>
{glance.strip()}
{''.join(T)}
<p class="method-note">{n} models covering ten of the twelve families ranked on this page (all except discrete diffusion and
Perceiver AR), each 0.47&ndash;0.48M parameters (the three exact-store and sidecar variants 0.50M; 4 layers, bytes in, tied
embeddings), trained on the same 2,000 updates of 8 &times; 256 bytes from the P056 corpus (encyclopedia articles, public-domain
fiction, OpenAssistant dialogue), with the same data stream for every model at each of two seeds, scored at the final checkpoint on
303 held-out test windows. Each column is an index where Transformer = 100 and higher is better; raw values are in brackets. Combined = weighted
geometric mean of the columns: {', '.join(f'{lab[k]} {w:.0%}' for k, w in W.items())}. Quality index = 100 &times; 2^(Transformer
bits/byte &minus; the model's), the ratio of per-byte perplexities; bits/byte is the mean of the three domains' held-out scores (31
article, 32 fiction and 240 dialogue windows). Speed and memory: one machine ({html.escape(sm['cpu'])}), one thread, every model in
its own process; every timing is the median over the rounds of two passes (the state-gated store and the control: three rounds of the
second pass). Inference memory = fp32 weights plus the per-sequence decode state after 256 tokens at batch 1; training memory is the
training step's peak above the process baseline (PyTorch and data, about 0.46 GiB). The spread under each combined score is its lowest
and highest value over the seed &times; timing-round pairs (each round's own timings against the Transformer's medians, each seed's
quality against the Transformer's at that seed); "&asymp;" in the first column marks a row whose spread overlaps the row above, i.e. not
separated from it. "Rank under {nways} ways of counting" is the best and worst rank under these weights and
{nways - 1} alternatives: {alts}. Each alternative's weights are listed in the README. Brackets in the findings are paired
differences in bits/byte, mean of both seeds (negative = first model better), with descriptive 95% intervals from resampling test
documents, which leave out seed-to-seed variation. All {n} models pass the quality gate (index &ge;
{idx['gate_quality_index']:.0f}) and are ranked.</p>
<h4>What it shows</h4>
<ul class="project-tests">
{findings.strip()}
<li>Code, per-run results, checkpoints and the review record: OM3 {lk(links.get('om3_github'), 'GitHub')} ·
{lk(links.get('om3_hf'), 'ScAGI Hugging Face')}; OM2 {lk(links.get('study_github'), 'GitHub')} · {lk(links.get('study_hf'), 'ScAGI Hugging Face')};
the fast exact CPU Mamba-2 kernel {lk(links.get('kernel_github'), 'GitHub')} · {lk(links.get('kernel_hf'), 'ScAGI Hugging Face')}.</li>
</ul>
<p class="method-note">Each of the eight new families is one implementation written from its paper and scaled down to the
shared budget, with its simplifications listed in its file; the Transformer, Mamba-2 (Hugging Face's definition and initialisation)
and our Mamba arms are OM2's. Five of the new ones (Gated DeltaNet, Titans, xLSTM, RetNet, RWKV-7) were reviewed independently against
their papers: RWKV-7's initialisation was fixed before training; Titans' major finding (linear instead of deep memory) was not, so that
arm is Titans' MAG design with a linear memory, a combination the paper does not report, with one minor fix (a bounded inner step
size). The hybrid, Griffin and Hyena were checked by tests only. Every model's whole-sequence and token-by-token outputs agree after
training (largest logit difference {max(r['parity_max'] for r in rows.values()):.1e} on two 96-byte test sequences). Training ran on
two machines (Mamba-2, our v1 and the two exact-store arms on PyTorch 2.10, the other twelve on 2.7.1) with the same data stream,
seeds and settings. The attention layers (Transformer, hybrid, Griffin, Titans-style, the exact store) use PyTorch's fused CPU
attention kernel, while the recurrent mixers run as ordinary PyTorch ops; Hyena runs its forward FFT in float64 and the Titans-style
arm takes an exact inner gradient step per token, both slower than their papers' recipes. Not included: discrete diffusion language
models (trained and scored with an upper bound on the likelihood, so their bits per byte would not be a matched number) and Perceiver
AR (its latents predict only the last positions of each window, and at 256 bytes its long-input bottleneck has little to compress).
Small CPU models, one corpus: a matched comparison of these designs at this size, not a ranking of the families at scale. The #
column ranks models inside this benchmark only: it places no design against the twelve-family ranking above, which is unchanged, and
the "Mamba-4" proposal (row {rows['om2g']['rank']}) stays unranked there, its first test not passed.</p>
</div>"""
Path(sys.argv[5]).write_text(card + '\n')
print(card)
