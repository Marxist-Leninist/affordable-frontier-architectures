#!/usr/bin/env python3
"""build_index.py -- the one combined index for ASI-Arch DLGM #458 vs Fusion #879 (claude-fable52, 2026-11-12, for Scott).
Reads the AGILLM-objective simulator runs that asi-agillm-sync pulls from the GPU queue every 2 minutes, applies the frozen formula
(index_spec_v1.json) and writes index.html + index.json into /var/www/zqx/asi-arch/dlgm-vs-fusion/. Standard library only; it writes nowhere else
except its own folder. Off switch: systemctl disable --now asi-dlgm-fusion-index.timer"""
import hashlib, html, json, math, os, random, subprocess, sys, time

HOME = '/root/asi_dlgm_fusion_index'
D = '/root/asi_agillm_dev'
OUT = '/var/www/zqx/asi-arch/dlgm-vs-fusion'
sys.path.insert(0, D)
import summarize as S          # the site's own run loader and equal-time arithmetic; used read-only

SPEC_PATH = HOME + '/index_spec_v1.json'
SPEC = json.load(open(SPEC_PATH))
SPEC_SHA = hashlib.sha256(open(SPEC_PATH, 'rb').read()).hexdigest()
MIX, CONST, CANDS = SPEC['quality_mix'], SPEC['constants'], SPEC['candidates']
KEYS = ('dlgm', 'fusion')
LABEL = {k: CANDS[k]['label'] for k in KEYS}
esc = html.escape


def q_of(e):
    return MIX['ar'] * e['ar'] + MIX['sat'] * e['sat'] + MIX['nat'] * e['nat']


def final(run):
    return next(c for c in run['curve'] if c.get('step') == 1200)


def stats(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    n = len(xs)
    if not n:
        return None, None, 0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else None
    return m, sd, n


def group(by, *pids):
    runs = [r for p in pids for r in by.get(p, [])]
    good = S.aggregate(runs)
    if not good:
        return None
    m = S.row_metrics(good)
    m['runs'] = good
    m['started'] = S.mean([r.get('started') for r in good])
    m['ids'] = sorted({str(r.get('plugin')) for r in good})
    return m


def own_objective(by):
    base = group(by, *SPEC['baseline_runs'])
    if not base:
        return None
    day = {p: g for p in SPEC['baseline_runs'] for g in [group(by, p)] if g}
    bq, bq_sd, bn = stats([q_of(final(r)) for r in base['runs']])
    last = base['curve'][-1]
    b_at = {k: S.interp(base['curve'], k, S.REF_TOKENS) for k in ('ar', 'sat', 'nat')}
    out = dict(base=dict(ids=base['ids'], seeds=base['seeds'], ar=last['ar'], sat=last['sat'], nat=last['nat'], q=bq, q_sd=bq_sd, ctx_gain=last['ctx_gain'], ms_per_step=base['ms_per_step'],
                         per_day={p: dict(seeds=g['seeds'], ms_per_step=g['ms_per_step'], started=g['started']) for p, g in day.items()}), forms={})
    for key in KEYS:
        for form in ('faithful', 'lowrank', 'hn'):
            pid = CANDS[key][form] if form != 'hn' else CANDS[key]['faithful'] + '_hn'
            m = group(by, pid)
            if not m:
                out['forms'][pid] = None
                continue
            gains = [bq - q_of(final(r)) for r in m['runs']]
            g, g_sd, n = stats(gains)
            sb = min(day.values(), key=lambda b: abs((b['started'] or 0) - (m['started'] or 0)))
            byte_ratio = ((m.get('touched_frac') or 1.0) * m['census'].get('latent_bytes', 1)) / ((sb.get('touched_frac') or 1.0) * sb['census'].get('latent_bytes', 1))
            ms_ratio = (m.get('ms_per_step') or 1.0) / (sb.get('ms_per_step') or 1.0)
            step_ratio = S.clamp(0.75 * byte_ratio + 0.25 * ms_ratio, 0.05, 10.0)
            tokens_eq = S.REF_TOKENS / step_ratio
            d = {k: b_at[k] - S.interp(m['curve'], k, tokens_eq) for k in b_at}
            lm = m['curve'][-1]
            out['forms'][pid] = dict(id=pid, cand=key, form=form, seeds=n, seeds_list=m['seeds_list'], ar=lm['ar'], sat=lm['sat'], nat=lm['nat'], ctx_gain=lm['ctx_gain'],
                                     gain_steps=g, gain_sd=g_sd, gain_se=(g_sd / math.sqrt(n)) if (g_sd is not None and n > 1) else None, gain_seeds=gains,
                                     ctx_vs_base=lm['ctx_gain'] - last['ctx_gain'], ms_per_step=m['ms_per_step'], base_same_day=sb['ids'], base_ms=sb['ms_per_step'],
                                     ms_ratio=ms_ratio, byte_ratio=byte_ratio, step_ratio=step_ratio, tokens_eq=tokens_eq,
                                     gain_time=MIX['ar'] * d['ar'] + MIX['sat'] * d['sat'] + MIX['nat'] * d['nat'], d_time=d, census=m['census'],
                                     max_gpu_mib=m['max_gpu_mib'], causal_ok=m['causal_ok'], sim_sha=m['sim_sha'], ld_mean=m['ld_mean'], started=m['started'], skipped=m['skipped'])
    return out


def share(ga, gb, floor):
    if ga is None or gb is None:
        return None
    top = max(ga, gb, floor or 0.0)
    if top <= 0:
        return dict(dlgm=0.0, fusion=0.0)
    return dict(dlgm=100.0 * max(ga, 0.0) / top, fusion=100.0 * max(gb, 0.0) / top)


def avg_shares(parts):
    parts = [p for p in parts if p is not None]
    return {k: sum(p[k] for p in parts) / len(parts) for k in KEYS} if parts else None


def components(own):
    forms = (own or {}).get('forms', {})
    comps = []
    for c in SPEC['components']:
        k, floor = c['key'], c['floor']
        e = dict(key=k, title=c['title'], weight=c['weight'], unit=c['unit'], measure=c['measure'], floor=floor, score=None, parts=[], pending=None)
        if k in ('own_steps', 'own_long_context', 'own_time'):
            form = 'lowrank' if k == 'own_time' else 'faithful'
            field = {'own_steps': 'gain_steps', 'own_long_context': 'ctx_vs_base', 'own_time': 'gain_time'}[k]
            a, b = forms.get(CANDS['dlgm'][form]), forms.get(CANDS['fusion'][form])
            if a and b:
                e['score'] = share(a[field], b[field], floor)
                e['parts'] = [dict(name='simulator, ' + ('low-rank' if form == 'lowrank' else 'faithful') + ' form', dlgm=a[field], fusion=b[field], seeds=dict(dlgm=a['seeds'], fusion=b['seeds']),
                                   se=dict(dlgm=a.get('gain_se'), fusion=b.get('gain_se')) if k == 'own_steps' else None)]
            else:
                e['pending'] = 'waiting for ' + ' and '.join(CANDS[x][form] for x, v in (('dlgm', a), ('fusion', b)) if not v)
        elif k == 'own_small':
            sh = []
            for corpus in ('wikitext', 'fineweb'):
                v = CONST['v65_r6'][corpus]
                ga, gb = v['baseline'] - v['dlgm'], v['baseline'] - v['fusion']
                sh.append(share(ga, gb, floor)); e['parts'].append(dict(name=corpus, dlgm=ga, fusion=gb))
            e['score'] = avg_shares(sh)
        elif k in ('up_tasks', 'up_loss'):
            sh = []
            tab = CONST['upstream']['tasks7' if k == 'up_tasks' else 'train_loss']
            for scale in ('20M', '340M'):
                v = tab[scale]
                ga, gb = (v['dlgm'] - v['baseline'], v['fusion'] - v['baseline']) if k == 'up_tasks' else (v['baseline'] - v['dlgm'], v['baseline'] - v['fusion'])
                sh.append(share(ga, gb, floor)); e['parts'].append(dict(name=scale + ' parameters', dlgm=ga, fusion=gb))
            e['score'] = avg_shares(sh)
        elif k == 'serving':
            e['score'] = dict(dlgm=float(CONST['serving']['dlgm']), fusion=float(CONST['serving']['fusion']))
            e['parts'] = [dict(name='rubric', dlgm=CONST['serving']['dlgm'], fusion=CONST['serving']['fusion'])]
        comps.append(e)
    return comps


def index_of(comps, weights=None, skip=None):
    tot, wsum = {x: 0.0 for x in KEYS}, 0.0
    for c in comps:
        if c['score'] is None or c['key'] == skip:
            continue
        w = c['weight'] if weights is None else weights[c['key']]
        wsum += w
        for x in KEYS:
            tot[x] += w * c['score'][x]
    return {x: tot[x] / wsum for x in KEYS} if wsum > 0 else None


def robustness(comps):
    avail = [c for c in comps if c['score'] is not None]
    rng = random.Random(SPEC['robustness']['seed'])
    n = SPEC['robustness']['random_weightings']
    wins = {x: 0 for x in KEYS}
    for _ in range(n):
        raw = {c['key']: rng.expovariate(1.0) for c in avail}
        i = index_of(avail, raw)
        wins['dlgm' if i['dlgm'] > i['fusion'] else 'fusion'] += 1
    loo = []
    for c in avail:
        i = index_of(avail, skip=c['key'])
        loo.append(dict(without=c['key'], title=c['title'], dlgm=i['dlgm'], fusion=i['fusion'], top='dlgm' if i['dlgm'] > i['fusion'] else 'fusion'))
    return dict(draws=n, share={x: wins[x] / n for x in KEYS}, leave_one_out=loo)


def queue_status(by):
    st = {}
    try:
        st = json.load(open(D + '/results/runs/_status.json'))
    except Exception:
        pass
    prog = {}
    for pid in (CANDS['fusion']['faithful'], CANDS['fusion']['lowrank'], 'base_r3', 'fusion879_hn', 'dlgm458_hn'):
        runs = by.get(pid, [])
        done = sorted(r['seed'] for r in runs if r.get('ok') and r.get('steps') == 1200)
        part = {r['seed']: (r['curve'][-1]['step'] if r.get('curve') else 0) for r in runs if not r.get('ok') and not r.get('error')}
        failed = sorted(r['seed'] for r in runs if r.get('error'))
        prog[pid] = dict(done=done, running=part, failed=failed)
    return dict(state=st.get('state'), current=st.get('current'), last_update=st.get('last_update'), progress=prog)


def pull_state():
    """The plugin's own log lines (retention, temperature, mean gates at the evaluation steps). Best effort over ssh; a failure keeps the last copy."""
    p = HOME + '/fusion_state.json'
    try:
        key = 'SSH_KEY_PATH'
        opt = ['-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=10', '-o', 'LogLevel=ERROR']
        r = subprocess.run(['ssh', '-p', 'SSH_PORT', '-i', key] + opt + ['root@GPU_BOX', 'cd /workspace/asi_agillm/logs && grep -H fusion_state_step fusion879*__s*.log 2>/dev/null'],
                           capture_output=True, text=True, timeout=25)
        rows = []
        for line in r.stdout.splitlines():
            name, _, js = line.partition(':')
            try:
                d = json.loads(js)
            except Exception:
                continue
            d['run'] = name.replace('.log', '')
            rows.append(d)
        if rows:
            json.dump(rows, open(p + '.tmp', 'w')); os.replace(p + '.tmp', p)
    except Exception:
        pass
    try:
        return json.load(open(p))
    except Exception:
        return []


def load_optional(name):
    try:
        return json.load(open(HOME + '/' + name))
    except Exception:
        return None


# ------------------------------------------------------------------ page
CSS = """
.viz-root{color-scheme:light;--plane:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,0.10);--series-1:#2a78d6;--series-2:#eb6834;--good:#006300}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--plane:#0d0d0d;--surface-1:#1a1a19;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,0.10);--series-1:#3987e5;--series-2:#d95926;--good:#0ca30c}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--plane:#0d0d0d;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,0.10);--series-1:#3987e5;--series-2:#d95926;--good:#0ca30c}
*{box-sizing:border-box}html,body{margin:0}
body.viz-root{background:var(--plane);color:var(--text-primary);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:0 20px 64px}
a{color:inherit;text-decoration:underline;text-decoration-color:var(--axis);text-underline-offset:3px}a:hover{text-decoration-color:currentColor}
header{padding:40px 0 8px}.eyebrow{font-size:13px;font-weight:650;letter-spacing:.05em;text-transform:uppercase;color:var(--text-secondary)}
h1{font-size:clamp(28px,4.4vw,42px);line-height:1.1;margin:8px 0 12px;font-weight:750;letter-spacing:-.02em}
h2{font-size:22px;margin:44px 0 6px;letter-spacing:-.01em}h3{font-size:16px;margin:22px 0 6px}
p{margin:8px 0;max-width:780px}.sub{color:var(--text-secondary);font-size:17px}.small{font-size:14px;color:var(--text-secondary)}
.pill{display:inline-flex;gap:8px;align-items:center;border:1px solid var(--border);background:var(--surface-1);border-radius:999px;padding:5px 14px;font-size:14px;color:var(--text-secondary);margin-top:6px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:14px;padding:22px 24px;margin-top:18px}
.hero{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:20px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:14px;padding:18px 22px;position:relative}
.tile.top{border:2px solid var(--text-primary)}
.tile .name{display:flex;align-items:center;gap:8px;font-weight:650}.sw{width:12px;height:12px;border-radius:3px;flex:none}.sw.s1{background:var(--series-1)}.sw.s2{background:var(--series-2)}
.tile .num{font-size:clamp(44px,8vw,64px);font-weight:750;letter-spacing:-.03em;line-height:1.05;margin-top:6px}
.tile .of{font-size:14px;color:var(--text-secondary)}
.chip{position:absolute;top:14px;right:16px;font-size:13px;font-weight:650;border:1px solid var(--border);border-radius:999px;padding:2px 10px;background:var(--plane)}
.verdict{font-size:19px;line-height:1.45;margin:18px 0 0;max-width:820px}
.legend{display:flex;gap:18px;font-size:14px;color:var(--text-secondary);margin:4px 0 12px}.legend span{display:inline-flex;gap:7px;align-items:center}
.row{display:grid;grid-template-columns:minmax(200px,300px) 1fr;gap:10px 22px;padding:12px 0;border-top:1px solid var(--grid)}
.row:first-of-type{border-top:0}.rt{font-weight:600;line-height:1.3}.rw{font-size:13px;color:var(--text-secondary);margin-top:2px}
.bl{display:grid;grid-template-columns:58px 1fr 48px;align-items:center;gap:8px;height:16px;position:relative;margin:2px 0}
.bl .who{font-size:12.5px;color:var(--text-secondary)}.track{height:10px;border-left:1px solid var(--axis);position:relative}
.bar{display:block;height:10px;border-radius:0 4px 4px 0;min-width:2px}.bar.s1{background:var(--series-1)}.bar.s2{background:var(--series-2)}
.val{font-size:13.5px;font-variant-numeric:tabular-nums;text-align:right}
.pend{font-size:14px;color:var(--text-secondary);font-style:italic;padding-top:4px}
[data-tip]:hover::after{content:attr(data-tip);position:absolute;left:58px;bottom:calc(100% + 3px);background:var(--text-primary);color:var(--surface-1);padding:4px 9px;border-radius:6px;
font-size:12.5px;white-space:nowrap;z-index:5;pointer-events:none}
.dv{display:grid;grid-template-columns:minmax(200px,300px) 1fr 1px 1fr 150px;align-items:center;gap:0 0;padding:7px 0;font-size:14.5px}
.dv .lab{padding-right:22px}.dv .l{display:flex;justify-content:flex-end}.dv .mid{background:var(--axis);height:22px}.dv .r{display:flex;justify-content:flex-start}
.dv .l .bar{border-radius:4px 0 0 4px}.dv .txt{padding-left:12px;font-variant-numeric:tabular-nums;color:var(--text-secondary)}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14.5px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
th{font-size:12.5px;font-weight:650;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.04em}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
ul{padding-left:20px;max-width:800px}li{margin:6px 0}code{font:13px ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--plane);border:1px solid var(--border);border-radius:5px;padding:1px 5px}
footer{margin-top:48px;font-size:13.5px;color:var(--text-secondary)}
@media (max-width:720px){.hero{grid-template-columns:1fr}.row{grid-template-columns:1fr}.dv{grid-template-columns:1fr 1px 1fr 120px}.dv .lab{grid-column:1/-1;padding:0 0 3px}}
"""


def f(x, n=1, plus=False):
    if x is None:
        return 'n/a'
    if round(x, n) == 0:
        return ('%.' + str(n) + 'f') % 0.0
    return (('%+.' + str(n) + 'f') % x) if plus else (('%.' + str(n) + 'f') % x)


def bars(c):
    if c['score'] is None:
        return '<div class="pend">%s</div>' % esc(c['pending'] or 'pending')
    out = []
    raw = c['parts']
    for key, cls in (('dlgm', 's1'), ('fusion', 's2')):
        s = c['score'][key]
        detail = '; '.join('%s: %s %s' % (p['name'], f(p[key], 4 if c['unit'] != 'rubric' and c['unit'] != 'accuracy points' else 2), c['unit']) for p in raw)
        tip = '%s: %s points (%s)' % (LABEL[key], f(s, 1), detail)
        out.append('<div class="bl" data-tip="%s"><span class="who">%s</span><span class="track"><span class="bar %s" style="width:%.2f%%"></span></span><span class="val">%s</span></div>'
                   % (esc(tip, quote=True), esc(LABEL[key].split(' ')[0]), cls, max(0.0, min(100.0, s)), f(s, 0)))
    return ''.join(out)


def raw_text(c):
    if c['score'] is None:
        return esc(c['pending'] or 'pending')
    bits = []
    for p in c['parts']:
        n = 2 if c['unit'] in ('accuracy points', 'rubric') else 4
        if c['unit'] == 'rubric':
            bits.append('DLGM %s, Fusion %s' % (p['dlgm'], p['fusion']))
        else:
            extra = ''
            if p.get('seeds'):
                extra = ' (%d and %d seeds)' % (p['seeds']['dlgm'], p['seeds']['fusion'])
            bits.append('%s: DLGM %s, Fusion %s %s%s' % (p['name'], f(p['dlgm'], n, True), f(p['fusion'], n, True), c['unit'], extra))
    return esc('; '.join(bits))


def build():
    by = S.load_runs()
    own = own_objective(by)
    comps = components(own)
    idx = index_of(comps)
    avail = [c for c in comps if c['score'] is not None]
    provisional = len(avail) < len(comps)
    forms = (own or {}).get('forms', {})
    fus = forms.get(CANDS['fusion']['faithful'])
    have_own = any(c['key'] == 'own_steps' and c['score'] is not None for c in comps)
    rob = robustness(comps) if idx else None
    st = queue_status(by)
    state = pull_state() if '--no-ssh' not in sys.argv else (load_optional('fusion_state.json') or [])
    census = load_optional('census560.json')
    fidelity = load_optional('fidelity_test_result.json')
    now = time.gmtime()
    top = None
    margin = None
    if idx and have_own:
        top = 'dlgm' if idx['dlgm'] > idx['fusion'] else 'fusion'
        margin = abs(idx['dlgm'] - idx['fusion'])
    other = {'dlgm': 'fusion', 'fusion': 'dlgm'}
    contrib = []
    wsum = sum(c['weight'] for c in avail) or 1.0
    for c in avail:
        contrib.append(dict(key=c['key'], title=c['title'], points=c['weight'] * (c['score']['fusion'] - c['score']['dlgm']) / wsum))
    # ---- words
    fp = st['progress'].get(CANDS['fusion']['faithful'], {})
    lp = st['progress'].get(CANDS['fusion']['lowrank'], {})
    prog_txt = 'Fusion faithful form: %d of 3 seeds done; low-rank form: %d of 3' % (len(fp.get('done', [])), len(lp.get('done', [])))
    if st.get('current'):
        prog_txt += '; GPU queue now running %s' % st['current']
    if not have_own:
        verdict = 'No name on top yet: the Fusion run under your objective is still training (%s). The page fills itself in as the seeds land.' % prog_txt
    else:
        w, l = top, other[top]
        seeds_note = '' if (fus and fus['seeds'] >= 3 and not provisional) else ' Provisional: %s.' % prog_txt
        if margin < 3.0:
            verdict = 'Level. %s is nominally ahead, %s to %s, a gap of %s points, which is inside the 3-point band where this index cannot separate them.%s' % (
                LABEL[w], f(idx[w]), f(idx[l]), f(margin), seeds_note)
        else:
            verdict = '%s is on top: %s to %s, a lead of %s points. It wins %d%% of %s random re-weightings of the same evidence.%s' % (
                LABEL[w], f(idx[w]), f(idx[l]), f(margin), round(100 * rob['share'][w]), '{:,}'.format(rob['draws']), seeds_note)
    # ---- html
    H = []
    H.append('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">')
    H.append('<title>DLGM #458 vs Fusion #879: one index</title>')
    H.append('<meta name="description" content="One combined index for the two ASI-Arch candidates DLGM #458 and Fusion #879: seven lines of evidence, including both under the AGILLM training objective, with the formula fixed before the result.">')
    H.append('<link rel="icon" href="../icon.svg"><style>' + CSS + '</style></head><body class="viz-root"><div class="wrap">')
    H.append('<header><div class="eyebrow">ASI-Arch on the AGILLM objective &middot; research result</div><h1>DLGM #458 vs Fusion #879: one index</h1>')
    H.append('<p class="sub">Two architecture candidates from ASI-Arch, seven lines of evidence, one number each. The weights and the scoring rule were fixed at %s UTC on 12 November 2026, before the Fusion run had produced a result.</p>' % esc(SPEC['frozen_at_utc'][11:16]))
    H.append('<div class="pill">Updated %s UTC &middot; %s</div></header>' % (time.strftime('%H:%M, %d %b %Y', now), esc(prog_txt)))
    H.append('<section class="hero">')
    for key, cls in (('dlgm', 's1'), ('fusion', 's2')):
        num = f(idx[key]) if (idx and have_own) else '&ndash;'
        H.append('<div class="tile%s"><div class="name"><span class="sw %s"></span>%s</div>%s<div class="num">%s</div><div class="of">index, 0 to 100%s</div></div>' % (
            ' top' if top == key else '', cls, esc(LABEL[key]), '<span class="chip">&#9650; On top</span>' if (top == key and margin is not None and margin >= 3.0) else ('<span class="chip">nominally ahead</span>' if top == key else ''),
            num, ' (provisional)' if (provisional and have_own) else ''))
    H.append('</section><p class="verdict">%s</p>' % esc(verdict))
    # the index chart
    H.append('<h2>The seven lines of evidence</h2><p class="small">Each line gives the better candidate 100 points and the other its share of that gain over the shared baseline. Hover a bar for the raw numbers; the table below has all of them.</p>')
    H.append('<div class="card"><div class="legend"><span><span class="sw s1"></span>%s</span><span><span class="sw s2"></span>%s</span></div>' % (esc(LABEL['dlgm']), esc(LABEL['fusion'])))
    for c in comps:
        H.append('<div class="row"><div><div class="rt">%s</div><div class="rw">weight %d%%</div></div><div>%s</div></div>' % (esc(c['title']), round(100 * c['weight']), bars(c)))
    H.append('</div>')
    # contributions
    if idx and have_own:
        mx = max(abs(x['points']) for x in contrib) or 1.0
        H.append('<h2>What moved the result</h2><p class="small">Index points each line adds to one side (its weight times the score gap). Bars to the left favour %s, bars to the right favour %s.</p><div class="card">' % (esc(LABEL['dlgm']), esc(LABEL['fusion'])))
        for x in sorted(contrib, key=lambda x: -abs(x['points'])):
            w = 100.0 * abs(x['points']) / mx
            left = '<span class="bar s1" style="width:%.1f%%"></span>' % w if x['points'] < 0 else ''
            right = '<span class="bar s2" style="width:%.1f%%"></span>' % w if x['points'] > 0 else ''
            who = 'level' if abs(x['points']) < 0.05 else ('%s to %s' % (f(abs(x['points'])), LABEL['fusion' if x['points'] > 0 else 'dlgm'].split(' ')[0]))
            H.append('<div class="dv"><div class="lab">%s</div><div class="l">%s</div><div class="mid"></div><div class="r">%s</div><div class="txt">%s</div></div>' % (esc(x['title']), left, right, esc(who)))
        H.append('</div>')
    # table view
    H.append('<h2>All the numbers</h2><div class="card tablewrap"><table><thead><tr><th>Evidence</th><th class="n">Weight</th><th class="n">DLGM</th><th class="n">Fusion</th><th>Gain over the shared baseline</th></tr></thead><tbody>')
    for c in comps:
        H.append('<tr><td><strong>%s</strong><br><span class="small">%s</span></td><td class="n">%d%%</td><td class="n">%s</td><td class="n">%s</td><td>%s</td></tr>' % (
            esc(c['title']), esc(c['measure']), round(100 * c['weight']), f(c['score']['dlgm']) if c['score'] else '&ndash;', f(c['score']['fusion']) if c['score'] else '&ndash;', raw_text(c)))
    if idx:
        H.append('<tr><td><strong>Index</strong>%s</td><td class="n">100%%</td><td class="n"><strong>%s</strong></td><td class="n"><strong>%s</strong></td><td></td></tr>' % (
            ' <span class="small">(over the lines available so far)</span>' if provisional else '', f(idx['dlgm']), f(idx['fusion'])))
    H.append('</tbody></table></div>')
    # own objective detail
    if own:
        b = own['base']
        H.append('<h2>Under your own training objective</h2><p class="small">The simulator is a small mirror of the live binary-weight block (12 layers, width 256), trained for 1,200 steps with your real mix of objectives and scored on held-out text. Lower loss is better. Candidate layers replace attention on 3 of every 4 layers.</p>')
        H.append('<div class="card tablewrap"><table><thead><tr><th>Run</th><th class="n">Seeds</th><th class="n">AR</th><th class="n">SAT</th><th class="n">NAT</th><th class="n">Gain, equal steps</th><th class="n">4x context</th><th class="n">ms per step</th><th class="n">Gain, equal sim time</th></tr></thead><tbody>')
        H.append('<tr><td>Live block (%s)</td><td class="n">%d</td><td class="n">%s</td><td class="n">%s</td><td class="n">%s</td><td class="n">0</td><td class="n">%s</td><td class="n">%s</td><td class="n">0</td></tr>' % (
            esc(', '.join(b['ids'])), b['seeds'], f(b['ar'], 4), f(b['sat'], 4), f(b['nat'], 4), f(b['ctx_gain'], 3, True), f(b['ms_per_step'], 0)))
        for key, form in [(k, fm) for fm in ('faithful', 'lowrank', 'hn') for k in KEYS]:
            if True:
                pid = CANDS[key][form] if form != 'hn' else CANDS[key]['faithful'] + '_hn'
                r = forms.get(pid)
                name = '%s, %s (<code>%s</code>)' % (esc(LABEL[key]), {'faithful': 'faithful form', 'lowrank': 'low-rank form', 'hn': 'per-head norm check'}[form], esc(pid))
                if not r:
                    if form == 'hn':
                        pr = st['progress'].get(pid, {})
                        if not (pr.get('done') or pr.get('running')):
                            continue
                    H.append('<tr><td>%s</td><td colspan="8" class="small">not finished yet</td></tr>' % name)
                    continue
                se = (' &plusmn; %s' % f(r['gain_se'], 3)) if r.get('gain_se') is not None else ''
                H.append('<tr><td>%s</td><td class="n">%d</td><td class="n">%s</td><td class="n">%s</td><td class="n">%s</td><td class="n">%s%s</td><td class="n">%s</td><td class="n">%s (x%s)</td><td class="n">%s</td></tr>' % (
                    name, r['seeds'], f(r['ar'], 4), f(r['sat'], 4), f(r['nat'], 4), f(r['gain_steps'], 3, True), se, f(r['ctx_gain'], 3, True), f(r['ms_per_step'], 0), f(r['ms_ratio'], 2), f(r['gain_time'], 3, True)))
        H.append('</tbody></table></div><p class="small">Gain = live block minus candidate on 0.6 AR + 0.2 SAT + 0.2 NAT, in nats, so positive is better; &plusmn; is one standard error across seeds. &ldquo;4x context&rdquo; is how much the loss on the last 256 tokens improves when the model also sees the 768 tokens before them (positive helps, negative hurts). &ldquo;Equal sim time&rdquo; charges each candidate for its step cost in this small simulator.</p>')
    # firmness
    H.append('<h2>How firm is it</h2><ul>')
    if rob and have_own:
        H.append('<li><strong>Other weightings.</strong> Across %s random weightings of the available lines, %s comes out ahead in %d%% and %s in %d%%.</li>' % (
            '{:,}'.format(rob['draws']), esc(LABEL['dlgm']), round(100 * rob['share']['dlgm']), esc(LABEL['fusion']), round(100 * rob['share']['fusion'])))
        flips = [x for x in rob['leave_one_out'] if x['top'] != top]
        if flips:
            H.append('<li><strong>Dropping one line.</strong> The name on top changes if you remove: %s. It stays the same without any of the others.</li>' % esc('; '.join(x['title'] for x in flips)))
        else:
            H.append('<li><strong>Dropping one line.</strong> The same name stays on top whichever single line of evidence is removed.</li>')
        a, bb = forms.get(CANDS['dlgm']['faithful']), forms.get(CANDS['fusion']['faithful'])
        if a and bb and a.get('gain_se') is not None and bb.get('gain_se') is not None:
            diff = bb['gain_steps'] - a['gain_steps']
            se = math.sqrt(a['gain_se'] ** 2 + bb['gain_se'] ** 2)
            H.append('<li><strong>Seed noise on your objective.</strong> At equal steps Fusion minus DLGM is %s nats with a standard error of %s (%s standard errors). %s</li>' % (
                f(diff, 3, True), f(se, 3), f(abs(diff) / se if se else 0.0, 1), 'That is a real difference at this sample size.' if se and abs(diff) / se >= 2 else 'That is inside seed noise: on this line the two are not separated.'))
    ha, hb = forms.get('dlgm458_hn'), forms.get('fusion879_hn')
    if ha and hb:
        hd = hb['gain_steps'] - ha['gain_steps']
        hse = math.sqrt((ha.get('gain_se') or 0.0) ** 2 + (hb.get('gain_se') or 0.0) ** 2) if (ha.get('gain_se') is not None and hb.get('gain_se') is not None) else None
        ma, mb = forms.get(CANDS['dlgm']['faithful']), forms.get(CANDS['fusion']['faithful'])
        main = (mb['gain_steps'] - ma['gain_steps']) if (ma and mb) else None
        H.append('<li><strong>Fairness check: the originals&rsquo; per-head output norm restored for both.</strong> Fusion minus DLGM at equal steps is %s nats%s with %d and %d seeds (main port: %s). Gains over the live block in this variant: DLGM %s, Fusion %s.</li>' % (
            f(hd, 3, True), (' &plusmn; %s' % f(hse, 3)) if hse is not None else '', hb['seeds'], ha['seeds'], f(main, 3, True), f(ha['gain_steps'], 3, True), f(hb['gain_steps'], 3, True)))
    else:
        pa, pb = st['progress'].get('dlgm458_hn', {}), st['progress'].get('fusion879_hn', {})
        if pa.get('done') or pb.get('done') or pa.get('running') or pb.get('running'):
            H.append('<li><strong>Fairness check running:</strong> both candidates are being re-run with the originals&rsquo; per-head output norm restored (%d of 3 DLGM seeds, %d of 3 Fusion seeds done). This line fills in when both are complete.</li>' % (len(pa.get('done', [])), len(pb.get('done', []))))
    v = CONST['v65_r6']['per_seed_fusion_minus_dlgm']
    H.append('<li><strong>Your matched small run</strong> is a statistical tie: the per-seed Fusion minus DLGM differences are %s on WikiText and %s on FineWeb, so the average lead of DLGM comes from one seed.</li>' % (
        esc(', '.join(f(x, 4, True) for x in v['wikitext'])), esc(', '.join(f(x, 4, True) for x in v['fineweb']))))
    H.append('<li><strong>The paper numbers</strong> are single runs with no error bars. %s</li>' % esc(CONST['upstream']['note_1p3b']))
    H.append('<li><strong>Serving readiness is a rubric, not a measurement.</strong> %s %s</li>' % (esc(CONST['serving']['fusion_reason']), esc(CONST['serving']['dlgm_reason'])))
    H.append('</ul><h3>Known limits of tonight&rsquo;s Fusion run</h3><ul>')
    if state:
        lam = min(x.get('lambda_min', 1.0) for x in state)
        endrows = [x for x in state if x.get('fusion_state_step') == 1200]
        gates = endrows[-1].get('mean_gate_short_long_delta_value') if endrows else None
        H.append('<li><strong>Retention did nothing in this run.</strong> Fusion&rsquo;s per-head forgetting factor starts at 0.99996 in the original and the lowest value any head reached was %s, so its memory behaved like DLGM&rsquo;s plain delta rule. What separated the two here is the gating and the FIR filters.%s</li>' % (
            f(lam, 5), (' Mean gate at the end (short FIR, long FIR, delta memory, value): %s.' % ', '.join(f(g, 2) for g in gates)) if gates else ''))
    else:
        H.append('<li><strong>Retention is expected to do nothing in this run.</strong> Fusion&rsquo;s per-head forgetting factor starts at 0.99996 in the original and cannot move far in 1,200 steps, so its memory behaves like DLGM&rsquo;s plain delta rule. What separates the two here is the gating and the FIR filters.</li>')
    H.append('<li><strong>One shared port convention is not neutral.</strong> Both ports use the block&rsquo;s own whole-width output norm instead of the per-head norm of the originals. Fusion&rsquo;s four independent gates can then also change how loud a head is; DLGM&rsquo;s gates always sum to one. Part of any Fusion advantage could come from that.</li>')
    H.append('<li><strong>Speed columns are not like for like.</strong> The DLGM port computes its memory in chunks of 16 tokens, Fusion needs 32 (its forgetting is tied to the chunk). Same mathematics for DLGM, different step time.</li>')
    H.append('<li><strong>Three seeds, a 12-layer mirror, 1,200 steps.</strong> This is a ranking signal under your objective, not a full-scale training result.</li></ul>')
    # full scale
    H.append('<h2>At full scale: 560 layers</h2>')
    if census:
        rows = census['rows']
        def gb(n):
            return '%.1f GB' % (n / 1e9)
        b560 = rows.get('base@560')
        H.append('<div class="card tablewrap"><table><thead><tr><th>At 560 blocks, width 8192</th><th class="n">Binary latents</th><th class="n">Extra floating-point parameters for the mixer</th><th class="n">Candidate layers</th></tr></thead><tbody>')
        for name, lab in (('base', 'Live block'), ('dlgm458_lr', 'DLGM #458, low-rank'), ('fusion879_lr', 'Fusion #879, low-rank'), ('dlgm458', 'DLGM #458, faithful'), ('fusion879', 'Fusion #879, faithful')):
            r = rows.get(name + '@560')
            if r:
                H.append('<tr><td>%s</td><td class="n">%s</td><td class="n">%s</td><td class="n">%d</td></tr>' % (esc(lab), '{:,.1f} B'.format(r['latent_bytes'] / 1e9), '{:,}'.format(r['mixer_fp_params']), r['candidate_layers']))
        H.append('</tbody></table></div>')
        lr_d, lr_f, fu_d, fu_f = (rows.get(k + '@560') for k in ('dlgm458_lr', 'fusion879_lr', 'dlgm458', 'fusion879'))
        if lr_d and lr_f and fu_d and fu_f and b560:
            H.append('<p>Both candidates keep every binary matrix of the live block (%s billion latents at 560 blocks), so storage and streaming are the same for both. The faithful forms add about %s and %s billion 32-bit parameters for their routers and gates, which is too much to train at this size; the low-rank forms add %s and %s million, under a tenth of a gigabyte, which is nothing. So the low-rank form is the one that could run, and on size the two are equal.</p>' % (
                '{:,.1f}'.format(b560['latent_bytes'] / 1e9), f(fu_d['mixer_fp_params'] / 1e9, 1), f(fu_f['mixer_fp_params'] / 1e9, 1), f(lr_d['mixer_fp_params'] / 1e6, 1), f(lr_f['mixer_fp_params'] / 1e6, 1)))
    H.append('<p><strong>&ldquo;Equal time&rdquo; is the simulator&rsquo;s model, not a full-scale measurement.</strong> In the small mirror a candidate step costs two to three times a live-block step, because at width 256 the mixer&rsquo;s convolutions and memory updates are large next to tiny matrices. At width 8192 the arithmetic runs the other way: a block spends about 800 million multiply-adds per token in its binary matrices, attention over 2,048 tokens about 34 million, and either mixer about 6 million. By that count both candidates would cost about the same per step as the live block, slightly less, and the two differ from each other by well under 1%. That is an estimate from counting operations; nobody has timed either mixer inside the real trainer.</p>')
    # method and files
    H.append('<h2>How the index is built</h2><p>%s</p><p>%s</p>' % (esc(SPEC['scoring']), esc(SPEC['headline_rule'])))
    H.append('<p class="small">Formula file: <a href="index_spec_v1.json">index_spec_v1.json</a> (sha256 <code>%s</code>), frozen %s. Everything on this page as data: <a href="index.json">index.json</a>.</p>' % (SPEC_SHA[:16] + '&hellip;', esc(SPEC['frozen_at_utc'])))
    H.append('<h2>What was run tonight</h2><ul>')
    H.append('<li>A new simulator plugin, <a href="fusion879.py">fusion879.py</a>, ports ASI-Arch #879 with the same layout and conventions as the existing DLGM port (<a href="../code/dlgm458.py">dlgm458.py</a>). The file name switches it to the low-rank form.</li>')
    if fidelity:
        try:
            c64 = max(x['max_abs_err'] for x in fidelity['phases'][0]['cases']); g32 = max(x['max_abs_err'] for x in fidelity['phases'][1]['cases'])
            H.append('<li>Fidelity check: with the original #879 weights copied in, the port reproduces the original&rsquo;s output to %.1e in 64-bit, and to %.1e in 32-bit on the GPU with the original library&rsquo;s kernels untouched, at six sequence lengths. No future-token leak. <a href="test_fusion879_equiv.py">Test</a>, <a href="fidelity_test_result.json">result</a>.</li>' % (c64, g32))
        except Exception:
            pass
    H.append('<li>Three seeds of each form on the GPU queue, plus a same-night control run of the live block for timing. Raw runs: <a href="../unified.json">unified.json</a> on the <a href="../">main ASI-Arch index</a>.</li>')
    H.append('<li>A three-reader review of the plugin before it was queued found no blocking defect; its caveats are the limits listed above.</li></ul>')
    H.append('<footer>Research project, November 2026. Built by Claude for Scott from the runs on the AGILLM cluster. Sources for the paper rows: ASI-Arch, arXiv 2507.18074v2, Table 9, and data/106.json at commit 3113c51b. This page rebuilds itself every two minutes from the synced runs.</footer>')
    H.append('</div></body></html>')
    os.makedirs(OUT, exist_ok=True)
    page = '\n'.join(H)
    tmp = OUT + '/index.html.tmp'
    open(tmp, 'w').write(page); os.chmod(tmp, 0o644); os.replace(tmp, OUT + '/index.html')
    data = dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', now), spec_sha256=SPEC_SHA, spec=SPEC, index=idx, top=top, margin=margin, provisional=provisional, have_own=have_own,
                verdict=verdict, components=comps, contributions=contrib, robustness=rob, own=own, queue=st, fusion_state=state, census=census, fidelity=fidelity)
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k not in ('runs', 'curve')}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    tmp = OUT + '/index.json.tmp'
    json.dump(clean(data), open(tmp, 'w'), indent=1); os.chmod(tmp, 0o644); os.replace(tmp, OUT + '/index.json')
    open(HOME + '/summary.txt', 'w').write(verdict + '\n' + ('' if not idx else 'DLGM %s, Fusion %s\n' % (f(idx['dlgm']), f(idx['fusion']))) + prog_txt + '\n')
    print(verdict); print(prog_txt)


if __name__ == '__main__':
    build()
