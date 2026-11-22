"""P039-L2 one-table view (owner request, 22 Nov 2026: one ranked table of every model and every metric). One row per network and decoder carrying every measured metric, ranked by the stated-weight
aggregate index (attn_ar greedy = 100). Rows that fail the quality gate follow, unranked, then the KN and all-space
references. A pure function of the summary dict, so it can be re-run on any summary.json."""
import math

MAIN = {'ar': 'greedy', 'diff': 'conf16'}   # decoders set before any network was scored on the test set
GATE_TEXT = {'likelihood_beats_kn': 'bits not below KN', 'nonspace_beats_kn_greedy': 'non-space acc not above KN',
             'degenerate_below_half': 'half or more degenerate'}


def fnum(x, spec='.3f'):
    return 'n/a' if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, spec)


def passes(dec):
    """Network passes used for the 16-byte suffix (AR greedy: 16 cached single-byte steps)."""
    if dec == 'greedy':
        return 16
    return int(dec[3:]) if dec.startswith('l2r') else int(dec[4:])


def build(summary, weights, steps=20000):
    arms = summary.get('arms') or {}
    t = (summary.get('timing') or {}).get('arms') or {}
    bb = summary.get('baselines') or {}
    kn_bits = (bb.get('kn_test_bits') or {}).get('macro')
    kn_rows = {k: v for k, v in (bb.get('decoders') or {}).items() if k.startswith('kn')}
    kn_ns = max((v['nonspace_acc']['macro'] for v in kn_rows.values()), default=None)
    common = summary.get('index') or {}

    def p50(arm, key):
        r = t.get(arm) or {}
        v = r.get('dev_' + key) or r.get(key)
        return v.get('p50_ms') if isinstance(v, dict) else None

    def lat(arm, dec):
        return p50(arm, 'b1_ar_cached_16' if dec == 'greedy' else f'b1_diff_{passes(dec)}pass_16')

    base = arms.get('attn_ar')
    b_bits = base['bits']['macro'] if base else None
    b_ns = base['decoders']['greedy']['nonspace_acc']['macro'] if base else None
    b_lat, b_step = lat('attn_ar', 'greedy'), p50('attn_ar', 'train_step')
    rows = []
    for arm, a in arms.items():
        kind = 'ar' if arm.endswith('_ar') else 'diff'
        step_ms = p50(arm, 'train_step')
        gains = []
        for curve in a.get('dev_curves') or []:
            if len(curve) < 2 or not step_ms:
                continue
            first, best = curve[0], min(curve, key=lambda d: d['dev_bits'])
            mins = (best['step'] - first['step']) * step_ms / 60000
            if mins > 0:
                gains.append((first['dev_bits'] - best['dev_bits']) / mins)
        hits = (common.get(arm) or {}).get('clean_minutes_to_common_dev_target') or []
        reached = [h for h in hits if h is not None]
        if not hits:
            target_txt = 'n/a'
        elif reached:
            target_txt = f'{sum(reached) / len(reached):.1f} ({len(reached)}/{len(hits)} seeds)'
        else:
            target_txt = f'not reached (0/{len(hits)})'
        peak = (t.get(arm) or {}).get('train_peak_mem_bytes')
        for dec, dd in a['decoders'].items():
            ns, L = dd['nonspace_acc']['macro'], lat(arm, dec)
            gate = {'likelihood_beats_kn': kn_bits is not None and a['bits']['macro'] < kn_bits,
                    'nonspace_beats_kn_greedy': kn_ns is not None and ns > kn_ns,
                    'degenerate_below_half': dd['degenerate']['macro'] < 0.5}
            ok = all(gate.values())
            ratios = q = agg = None
            if base and b_bits and b_ns:
                ratios = {'likelihood': b_bits / a['bits']['macro'], 'nonspace_acc': max(ns, 1e-6) / b_ns,
                          'latency_b1': b_lat / L if b_lat and L else None,
                          'train_time': b_step / step_ms if b_step and step_ms else None}
                q = 100 * math.exp(0.5 * math.log(ratios['likelihood']) + 0.5 * math.log(ratios['nonspace_acc']))
                if ok and all(ratios.get(k) for k in weights):
                    agg = 100 * math.exp(sum(w * math.log(ratios[k]) for k, w in weights.items()))
            rows.append({'model': arm, 'decoder': dec, 'preset_decoder': dec == MAIN[kind], 'seeds': a.get('seeds'),
                         'index': agg, 'quality_index': q, 'gate': gate, 'gate_pass': ok, 'ratios_vs_attn_ar_greedy': ratios,
                         'bits': a['bits']['macro'], 'bits_ci95': a['bits'].get('ci95'), 'bits_kind': a.get('bits_kind'),
                         'nonspace_acc': ns, 'exact': dd['exact']['macro'], 'byte_acc': dd['byte_acc']['macro'],
                         'space_rate': dd['space_pred_rate']['macro'], 'degenerate': dd['degenerate']['macro'],
                         'passes': passes(dec), 'decode_ms_16_bytes_b1': L,
                         'train_minutes': step_ms * steps / 60000 if step_ms else None,
                         'dev_bits_gained_per_train_minute': sum(gains) / len(gains) if gains else None,
                         'minutes_to_common_dev_target': sum(reached) / len(reached) if reached else None,
                         'target_txt': target_txt, 'train_peak_mem_mb': peak / 2 ** 20 if peak else None,
                         'params': a.get('params')})
    key = 'index' if any(r['index'] is not None for r in rows) else 'quality_index'
    ranked = sorted((r for r in rows if r['gate_pass'] and r[key] is not None), key=lambda r: -r[key])
    rest = sorted((r for r in rows if r not in ranked),
                  key=lambda r: -(r['quality_index'] if r['quality_index'] is not None else -1e9))
    for i, r in enumerate(ranked, 1):
        r['rank'] = i
    refs = []
    if kn_bits is not None:
        for k, v in kn_rows.items():
            refs.append({'model': f"KN byte {bb.get('kn_order_selected_on_dev', '?')}-gram", 'decoder': 'greedy',
                         'bits': kn_bits, 'bits_ci95': (bb.get('kn_test_bits') or {}).get('ci95'),
                         'nonspace_acc': v['nonspace_acc']['macro'], 'exact': v['exact']['macro'],
                         'space_rate': v['space_pred_rate']['macro'], 'degenerate': v['degenerate']['macro']})
    if 'all_space' in (bb.get('decoders') or {}):
        v = bb['decoders']['all_space']
        refs.append({'model': 'always a space', 'decoder': '-', 'bits': None, 'bits_ci95': None,
                     'nonspace_acc': v['nonspace_acc']['macro'], 'exact': v['exact']['macro'],
                     'space_rate': v['space_pred_rate']['macro'], 'degenerate': v['degenerate']['macro']})

    hdr = ['Rank', 'Model · decoder', 'Index (attn_ar = 100)', 'Quality index', 'Held-out bits/byte [95% CI]', 'Non-space acc',
           'Exact 16/16', 'Space rate', 'Degenerate', 'Decode ms (16 bytes, GPU kernel)', 'Train GPU min',
           'Dev bits gained / train min', 'GPU min to common dev target', 'Train-step peak MiB', 'Params', 'Gate']
    out = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]

    def bits_txt(r):
        ci = r.get('bits_ci95')
        return fnum(r['bits']) + (f' [{fnum(ci[0])}, {fnum(ci[1])}]' if ci else '')

    for r in ranked + rest:
        gate = 'pass' if r['gate_pass'] else 'fail: ' + ', '.join(GATE_TEXT[k] for k, v in r['gate'].items() if not v)
        out.append('| ' + ' | '.join([
            str(r.get('rank', '–')), f"{r['model']} · {r['decoder']}" + (' ★' if r['preset_decoder'] else ''),
            fnum(r['index'], '.1f'), fnum(r['quality_index'], '.1f'), bits_txt(r), fnum(r['nonspace_acc'], '.4f'),
            fnum(r['exact'], '.4f'), fnum(r['space_rate'], '.3f'), fnum(r['degenerate'], '.3f'),
            fnum(r['decode_ms_16_bytes_b1'], '.2f'), fnum(r['train_minutes'], '.1f'),
            fnum(r['dev_bits_gained_per_train_minute'], '.4f'), r['target_txt'], fnum(r['train_peak_mem_mb'], '.0f'),
            f"{r['params']:,}" if r['params'] else 'n/a', gate]) + ' |')
    for r in refs:
        out.append('| ' + ' | '.join(['ref', f"{r['model']} · {r['decoder']}", 'n/a', 'n/a', bits_txt(r),
                                      fnum(r['nonspace_acc'], '.4f'), fnum(r['exact'], '.4f'), fnum(r['space_rate'], '.3f'),
                                      fnum(r['degenerate'], '.3f')] + ['n/a'] * 6 + ['reference']) + ' |')
    w = ', '.join(f"{k.replace('_', ' ')} {v:g}" for k, v in weights.items())
    foreign = sorted({p for a in t.values() for p in (a.get('foreign_gpu_seen') or [])})
    how = ('Ranked by the Index.' if key == 'index' else
           'Ranked by the Quality index: the full Index needs the timing file, which is missing.')
    notes = ['',
             f'How to read: one row per network and decoder, best first. {how} Index = weighted geometric mean of each metric\'s '
             f'ratio to attn_ar greedy, inverted where lower is better ({w}); 100 = attn_ar greedy, higher is better. Only rows that pass the quality gate are '
             f'ranked: held-out bits/byte below the KN byte n-gram ({fnum(kn_bits)}), non-space accuracy above KN greedy '
             f'({fnum(kn_ns, ".4f")}), and fewer than half of the completions degenerate (one byte 12 or more times in 16). '
             'So a fast model that outputs spaces cannot rank. The gate and the weights were set before any network was scored '
             'on the test set. Quality index = held-out likelihood and non-space accuracy only, '
             'equal weights.',
             '',
             'Held-out bits/byte: exact next-byte NLL for AR; for diffusion, the exact left-to-right NLL of the same network '
             '(one masked byte revealed per step). Decode ms = summed GPU kernel time (torch.profiler) for one 16-byte '
             'completion after a 240-byte prompt, batch 1, '
             f"{(summary.get('timing') or {}).get('gpu') or 'one GPU'}, median of three rotated rounds; it leaves out Python "
             'and kernel-launch overhead, which weighs most on cached AR (eager wall-clock times are in timing.json under the '
             'b1_* keys). AR reuses its KV/state cache, diffusion re-reads the whole window on every pass (confN = N passes, '
             f'most confident bytes first; l2r16 = 16 passes, left to right). Train GPU min = {steps:,} steps × the median '
             'over three rounds of the mean GPU kernel time per training step. Dev scores are exact NLL for AR and an '
             '8-sample NELBO (an upper bound) for diffusion, checked every 2,000 steps. Dev bits gained / train min = drop in '
             'dev bits/byte from the first dev check to the best one, per GPU minute between them; it rewards a model that '
             'starts worse, so it is not comparable between AR and diffusion. GPU min to common dev target = GPU minutes '
             "until the dev curve first reaches the worst AR run's best dev bits (mean over seeds). Train-step peak MiB = "
             'peak allocated memory of one training step in the timing pass; the full runs record their own peaks, which '
             "include dev evaluation, in state.json. ★ = decoder set before any network was scored on the test set.",
             '',
             'Timing clean (no other GPU processes seen while timing): '
             + ('yes' if t and not foreign else 'no timing file' if not t else f'NO, PIDs {foreign}; speed terms are provisional') + '.']
    return out + notes, {'ranked_by': key, 'rows': ranked + rest, 'references': refs, 'weights': weights,
                         'timing_clean': bool(t) and not foreign}
