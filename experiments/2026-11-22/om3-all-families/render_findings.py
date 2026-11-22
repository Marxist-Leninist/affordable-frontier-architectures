#!/usr/bin/env python3
"""Fill data placeholders in a findings text, so the numbers in the prose come from the results files.
Placeholders (A, B = arm names; K = an alternative-counting key from index.json 'alternatives'; N = a rank):
  {ci:A:B}   paired test bits/byte difference A minus B, mean of both seeds, with its descriptive 95% interval
  {ps:A:B}   the two per-seed differences A minus B        {bpb:A}  test bits/byte     {bpbs:A}  soft-read bits/byte (om2g)
  {comb:A}   combined index      {sp:A}  spread over the seed x timing-round pairs      {rl:A}  rank label ('1=' if tied)
  {win:A:B}  in how many common seed x timing-round pairs A's combined score beats B's ("4 of 12")
  {rank:A} rank   {ro:A} rank as an ordinal   {qro:A} quality rank as an ordinal (1st = best bits/byte)
  {rr:A}     best-worst rank under all the ways of counting   {top3:A} / {top4:A}  how many of them put A in the top 3 / 4
  {ar:A:K} / {aro:A:K}  rank under alternative K (number / ordinal)   {ac:A:K}  combined score under K
  {lead:K}   name of the leader under K   {leadx:K}  the leader, or the leader and runner-up with scores when < 1 point apart
  {nth:N} / {combn:N}  name / combined score of the model ranked N
  {q:A} quality index   {rec:A} repeat-span accuracy %   {sps:A} training s/step   {dec:A} decode ms/token (median)
  {pre:A} prefill thousand tok/s   {state:A} per-sequence decode state KiB at 256 tokens   {statet:A} whole decode cache KiB
  {tabs:A} KiB of input-independent tables in A's decode cache   {mem:A} inference memory MiB   {rss:A} training memory MiB
  {ttt:A} training minutes to the common dev target   {tttf:A} minutes to the Transformer's own final dev score
  {xs:A:B}   how many times faster A trains per step than B   {xm:A:B}  how many times more training memory A needs than B
  {name:A}   display name   {read:A}  gate read rate % at the designed threshold   {gate:A}  mean soft gate per seed
  {spct:A}   how much the state-only memory column alone multiplies A's score (x)   {mult:A:M}  same for column M
  {ctl:F}    Transformer-without-fused-attention control: F = train, decode, prefill, train_memory (factors), comb, rank
  {qpct} the most the quality column moves any score (%)   {recrange} lowest-highest repeat-span accuracy %
  {cg} lowest-highest copy gain   {target} common dev target   {tffinal} the Transformer's own final dev score
  {nways} number of ways of counting   {alltop4rr} models in the top four under all of them, with their ranges
  {maxjit} / {trjit} largest round-to-round spread of a model's decode / training-step time (%)   {tfeq} control equivalence
Usage: render_findings.py INDEX_JSON CONTRASTS_JSON IN OUT [--html]"""
import html as H
import json, re, sys
from pathlib import Path

TOKENS = ('ci|ps|bpb|bpbs|comb|sp|rl|win|rank|ro|qro|rr|top3|top4|ar|aro|ac|leadx|leadc|lead|nth|combn|q|rec|sps|dec|pre|state|statet|'
          'tabs|mem|rss|ttt|tttf|xs|xm|name|read|gate|spct|mult|ctl|maxmove|wi')
# {maxmove:K}: the most places any model's rank changes between the stated weights and alternative K


def ordinal(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def make_render(idx, contrasts, as_html=True):
    rows, alts = idx['rows'], idx['alternatives']
    ranked = [a for a in idx['order'] if 'rank' in rows[a]]
    dash = '&ndash;' if as_html else '–'
    times = '&times;' if as_html else 'x'
    esc = H.escape if as_html else (lambda s: s)

    def pairs(a):
        return {(p['seed'], p['timing_pass'], p['timing_round']): p['combined'] for p in rows[a]['combined_pairs']}

    def one(m):
        kind, *a = m.group(1).split(':')
        if kind == 'ci':      # contrasts may name pseudo-arms (om2gsoft) that have no row
            c = contrasts[f'{a[0]} minus {a[1]}']
            return f"{c['mean_bpb']:+.3f} [{c['interval95'][0]:+.3f}, {c['interval95'][1]:+.3f}]"
        if kind == 'ps':
            c = contrasts[f'{a[0]} minus {a[1]}']
            return ' and '.join(f'{x:+.3f}' for x in c['per_seed'])
        if kind in ('lead', 'leadx', 'leadc'):     # leadc: a clause, "X leads" or "X and Y are level at the top (a and b)"
            al = alts[a[0]]
            l, ru = al['leader'], al['runner_up']
            sl, sr = rows[l]['alt_combined'][a[0]], rows[ru]['alt_combined'][a[0]]
            if kind == 'lead' or not al.get('level', sl - sr < 1):
                return esc(rows[l]['name']) + (f' leads ({sl:.1f}; next {esc(rows[ru]["name"])} {sr:.1f})' if kind == 'leadc' else '')
            if kind == 'leadc':
                return f"{esc(rows[l]['name'])} and {esc(rows[ru]['name'])} are level at the top ({sl:.1f} and {sr:.1f})"
            return f"{esc(rows[l]['name'])} and {esc(rows[ru]['name'])} level ({sl:.1f} and {sr:.1f})"
        if kind == 'maxmove':
            return str(max(abs(rows[x]['rank'] - rows[x]['alt_rank'][a[0]]) for x in ranked))
        if kind in ('nth', 'combn'):
            r = rows[ranked[int(a[0]) - 1]]
            return esc(r['name']) if kind == 'nth' else f"{r['combined']:.1f}"
        if kind == 'ctl':
            c = idx['controls']['tfplain']
            if a[0] in c['factors']:
                return f"{c['factors'][a[0]]:.2f}"
            nr = c.get('within_spread_of_ranks') or []
            return {'comb': f"{c['combined']:.1f}", 'rank': str(c['rank_if_plain']),
                    'near': (f"{min(nr)}{dash}{max(nr)}" if len(nr) > 1 else str(nr[0])) if nr else str(c['rank_if_plain']),
                    'dtf': f"{min(c['decode_ms_rounds_tf_pass2']):.2f}{dash}{max(c['decode_ms_rounds_tf_pass2']):.2f}",
                    'dpl': f"{min(c['decode_ms_rounds_plain']):.2f}{dash}{max(c['decode_ms_rounds_plain']):.2f}"}[a[0]]
        if kind == 'wi':      # what-if values: {wi:om2g_soft_comb} / {wi:om2g_soft_rank}
            w = idx['whatif']['om2g_soft']
            return f"{w['combined']:.1f}" if a[0].endswith('comb') else ordinal(w['rank'])
        r = rows[a[0]]
        if kind == 'win':
            A, B = pairs(a[0]), pairs(a[1])
            keys = sorted(set(A) & set(B))
            return f"{sum(A[k] > B[k] for k in keys)} of {len(keys)}"
        if kind in ('ar', 'aro', 'ac'):
            v = r['alt_rank'][a[1]] if kind != 'ac' else r['alt_combined'][a[1]]
            return str(v) if kind == 'ar' else (ordinal(v) if kind == 'aro' else f'{v:.1f}')
        if kind == 'mult':
            return f"{r['multipliers'][a[1]]:.2f}"
        if kind == 'xs':
            return f"{rows[a[1]]['train_s_per_step'] / r['train_s_per_step']:.1f}{times}"
        if kind == 'xm':
            return f"{r['train_rss'] / rows[a[1]]['train_rss']:.1f}{times}"
        f = {'bpb': lambda: f"{r['bpb']:.3f}", 'bpbs': lambda: f"{r['bpb_soft']:.3f}", 'comb': lambda: f"{r['combined']:.1f}",
             'sp': lambda: f"{r['combined_spread'][0]:.1f}{dash}{r['combined_spread'][1]:.1f}", 'rl': lambda: r['rank_label'],
             'rank': lambda: str(r['rank']), 'ro': lambda: ordinal(r['rank']), 'qro': lambda: ordinal(r['quality_rank']),
             'rr': lambda: dash.join(map(str, r['rank_range'])), 'top3': lambda: str(r['top3_count']),
             'top4': lambda: str(r['top4_count']), 'q': lambda: f"{r['index']['quality']:.0f}",
             'rec': lambda: f"{r['recall'] * 100:.0f}", 'sps': lambda: f"{r['train_s_per_step']:.2f}",
             'dec': lambda: f"{r['decode_ms']:.1f}", 'pre': lambda: f"{r['prefill'] / 1000:.1f}",
             'state': lambda: f"{r['state_bytes'] / 1024:.0f}", 'statet': lambda: f"{r['state_bytes_total'] / 1024:.0f}",
             'tabs': lambda: f"{(r['state_bytes_total'] - r['state_bytes']) / 1024:.0f}",
             'mem': lambda: f"{r['inference_bytes'] / 2 ** 20:.2f}", 'rss': lambda: f"{r['train_rss']:.0f}",
             'ttt': lambda: f"{r['train_min_to_common_target']:.1f}", 'tttf': lambda: f"{r['train_min_to_tf_final']:.1f}",
             'name': lambda: esc(r['name']), 'read': lambda: f"{r['gate_read_rate_hard'] * 100:.2f}",
             'gate': lambda: ' and '.join(f'{g:.2f}' for g in r['gate_mean_soft']),
             'spct': lambda: f"{(r['index']['state_only_memory'] / 100) ** idx['weights']['inference_memory']:.2f}"}
        return f[kind]()

    recs = [r['recall'] * 100 for r in rows.values()]
    cgs = [x for r in rows.values() for x in r['copy_gain_bpb_range']]
    qmove = max(abs(r['multipliers']['quality'] - 1) for r in rows.values()) * 100
    top4 = [a for a in ranked if rows[a]['rank_range'][1] <= 4]
    top4_txt = ', '.join(f"{esc(rows[a]['name'])} (ranks {dash.join(map(str, rows[a]['rank_range']))})" for a in top4) or 'no model'
    eq = (idx['controls'].get('tfplain') or {}).get('equivalence') or {}
    special = {'{nways}': str(1 + len(alts)), '{recrange}': f"{min(recs):.0f}{dash}{max(recs):.0f}", '{qpct}': f'{qmove:.0f}',
               '{cg}': f"{min(cgs):+.2f} to {max(cgs):+.2f}", '{target}': f"{idx['common_dev_target_bpb']:.3f}",
               '{tffinal}': f"{idx['tf_final_dev_bpb']:.3f}", '{alltop4rr}': top4_txt,
               '{maxjit}': f"{idx['timing_jitter']['decode_max'] * 100:.0f}", '{trjit}': f"{idx['timing_jitter']['train_max'] * 100:.0f}",
               '{tfeq}': (f"{max(eq.get('forward_max_abs_logit_diff', 0), eq.get('step_max_abs_logit_diff', 0)):.1e}"
                          .replace('e-0', 'e-')) if eq else '?',
               '{paritymax}': f"{max(r['parity_max'] for r in rows.values()):.1e}"}

    def render(text):
        assert special['{tfeq}'] != '?' or '{tfeq}' not in text, 'equivalence file missing'
        for k, v in special.items():
            text = text.replace(k, v)
        out = re.sub(r'\{((?:' + TOKENS + r'):[a-z0-9_]+(?::[a-z0-9_]+)?)\}', one, text)
        left = re.findall(r'\{[a-z0-9_:]+\}', re.sub(r'<[^>]+>', '', out))
        assert not left, f'unfilled placeholders: {left}'
        return out
    return render


if __name__ == '__main__':
    idx = json.loads(Path(sys.argv[1]).read_text())
    C = json.loads(Path(sys.argv[2]).read_text())['contrasts']
    out = make_render(idx, C, as_html='--html' in sys.argv)(Path(sys.argv[3]).read_text())
    Path(sys.argv[4]).write_text(out)
    print(out)
