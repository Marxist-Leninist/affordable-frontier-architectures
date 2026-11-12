#!/usr/bin/env python3
"""make_package.py -- assemble the reproducible package of the DLGM #458 vs Fusion #879 index (code, frozen formula, raw runs, results, README) for the
research repo. Reads the same index.json the web page is built from, so the README can never disagree with the page."""
import glob, hashlib, json, os, shutil, sys, time
HOME = '/root/asi_dlgm_fusion_index'
D = '/root/asi_agillm_dev'
WEB = '/var/www/zqx/asi-arch/dlgm-vs-fusion'
NAME = 'asi_arch_dlgm_vs_fusion_index_20261112'
DEST = HOME + '/package/' + NAME
d = json.load(open(WEB + '/index.json'))
shutil.rmtree(DEST, ignore_errors=True)
os.makedirs(DEST + '/runs'); os.makedirs(DEST + '/plugins'); os.makedirs(DEST + '/page')
def cp(src, dst):
    shutil.copyfile(src, DEST + '/' + dst)
for n in ('fusion879', 'fusion879_lr', 'fusion879_lrk', 'dlgm458_lrk', 'fusion879_lg', 'dlgm458_lg', 'fusion879_hn', 'dlgm458_hn'):
    p = D + '/plugins/%s.py' % n
    if os.path.exists(p): cp(p, 'plugins/%s.py' % n)
for n in ('dlgm458', 'dlgm458_lr', 'base', 'base_r2', 'base_r3'):
    p = D + '/results/plugins/%s.py' % n
    if os.path.exists(p): cp(p, 'plugins/%s.py' % n)
cp(D + '/results/src/agillm_mini.py', 'agillm_mini.py')
cp(D + '/plugins/_fusion879_test/test_fusion879_equiv.py', 'test_fusion879_equiv.py')
for n in ('fidelity_test_result.json', 'index_spec_v1.json', 'build_index.py', 'make_package.py', 'census560.json', 'census560.py', 'fusion_state.json'):
    if os.path.exists(HOME + '/' + n):
        cp(HOME + '/' + n, n)
        if n.endswith('.py'):   # no server addresses in the public copy
            t = open(DEST + '/' + n).read().replace('root@GPU_BOX', 'root@GPU_BOX').replace("'SSH_PORT'", "'SSH_PORT'").replace('SSH_KEY_PATH', 'SSH_KEY_PATH')
            open(DEST + '/' + n, 'w').write(t)
cp(WEB + '/index.json', 'index.json'); cp(WEB + '/index.html', 'page/index.html')
runs = []
for pat in ('base__s*', 'base_r2__s*', 'base_r3__s*', 'dlgm458__s*', 'dlgm458_lr__s*', 'dlgm458_hn__s*', 'dlgm458_lrk__s*', 'dlgm458_lg__s*', 'fusion879_lg__s*', 'fusion879__s*', 'fusion879_lr__s*', 'fusion879_lrk__s*', 'fusion879_hn__s*'):
    for p in sorted(glob.glob(D + '/results/runs/' + pat + '.json')):
        r = json.load(open(p))
        if r.get('ok'):
            cp(p, 'runs/' + os.path.basename(p)); runs.append(os.path.basename(p))
f = lambda x, n=1: 'n/a' if x is None else ('%.' + str(n) + 'f') % x
g = lambda x, n=3: 'n/a' if x is None else ('%+.' + str(n) + 'f') % x
idx, comps, own = d.get('index') or {}, d['components'], d.get('own') or {}
L = []
L.append('# DLGM #458 vs Fusion #879: one combined index (12 November 2026)\n')
L.append('Two ASI-Arch candidates compared on seven lines of evidence, including both under the AGILLM training objective, and combined into one number each. The weights and the scoring rule were frozen (`index_spec_v1.json`, sha256 `%s`) at %s, before the Fusion run under this objective had produced a result.\n' % (d['spec_sha256'], d['spec']['frozen_at_utc']))
L.append('Live page (rebuilds itself from the synced runs): https://zqx.lat/asi-arch/dlgm-vs-fusion/\n')
L.append('## Result\n')
L.append(d['verdict'] + '\n')
if idx:
    L.append('| | DLGM #458 | Fusion #879 |\n|---|---:|---:|\n| **Index (0 to 100)** | **%s** | **%s** |\n' % (f(idx.get('dlgm')), f(idx.get('fusion'))))
alt = d.get('alternative_forms') or {}
if alt:
    L.append('**Read this with the result.** The frozen formula scores the two main own-objective lines with each candidate in its faithful form. Only a cut-down form fits at 560 layers. The same index with other forms on those two lines:\n')
    L.append('| Form on the two own-objective lines | DLGM gain (nats) | Fusion gain (nats) | Seeds | Index DLGM | Index Fusion |\n|---|---:|---:|---|---:|---:|')
    for k, x in alt.items():
        L.append('| %s | %s | %s | %d and %d | %s | %s |' % (x['form'], g(x['gain']['dlgm']), g(x['gain']['fusion']), x['seeds']['dlgm'], x['seeds']['fusion'], f(x['index']['dlgm']), f(x['index']['fusion'])))
    L.append('')
L.append('## The seven lines\n')
L.append('| Evidence | Weight | DLGM | Fusion | Gain over the shared baseline |\n|---|---:|---:|---:|---|')
for c in comps:
    if c['score']:
        raw = '; '.join('%s: DLGM %s, Fusion %s' % (p['name'], g(p['dlgm'], 4) if c['unit'] != 'rubric' else p['dlgm'], g(p['fusion'], 4) if c['unit'] != 'rubric' else p['fusion']) for p in c['parts'])
        L.append('| %s | %d%% | %s | %s | %s (%s) |' % (c['title'], round(100 * c['weight']), f(c['score']['dlgm']), f(c['score']['fusion']), raw, c['unit']))
    else:
        L.append('| %s | %d%% | n/a | n/a | %s |' % (c['title'], round(100 * c['weight']), c.get('pending') or 'pending'))
L.append('\nScoring: ' + d['spec']['scoring'] + ' ' + d['spec']['headline_rule'] + '\n')
if own:
    b = own['base']
    L.append('## Under the AGILLM objective (simulator `agillm_mini.py`, 1,200 steps, held-out loss, lower is better)\n')
    L.append('| Run | Seeds | AR | SAT | NAT | Gain, equal steps (nats) | 4x context | ms per step | Gain, equal simulator time |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    L.append('| Live block (%s) | %d | %s | %s | %s | 0 | %s | %s | 0 |' % (', '.join(b['ids']), b['seeds'], f(b['ar'], 4), f(b['sat'], 4), f(b['nat'], 4), g(b['ctx_gain']), f(b['ms_per_step'], 0)))
    for pid, r in own['forms'].items():
        if r:
            se = (' +/- %s' % f(r['gain_se'], 3)) if r.get('gain_se') is not None else ''
            L.append('| `%s` | %d | %s | %s | %s | %s%s | %s | %s (x%s) | %s |' % (pid, r['seeds'], f(r['ar'], 4), f(r['sat'], 4), f(r['nat'], 4), g(r['gain_steps']), se, g(r['ctx_gain']), f(r['ms_per_step'], 0), f(r['ms_ratio'], 2), g(r['gain_time'])))
    L.append('\nGain = live block minus candidate on 0.6 AR + 0.2 SAT + 0.2 NAT; +/- is one standard error across seeds. "4x context" is the change in loss on the last 256 tokens when the 768 tokens before them are also visible. "Equal simulator time" uses the site method (step cost = 0.75 x bytes streamed + 0.25 x measured ms per step against the base run of the same day).\n')
rob = d.get('robustness')
L.append('## How firm\n')
if rob:
    L.append('- Random weightings (%d draws, uniform on the simplex): DLGM ahead in %d%%, Fusion in %d%%.' % (rob['draws'], round(100 * rob['share']['dlgm']), round(100 * rob['share']['fusion'])))
    L.append('- Leave one line out: ' + '; '.join('without "%s": %s on top (%s vs %s)' % (x['title'], d['spec']['candidates'][x['top']]['label'], f(x['dlgm']), f(x['fusion'])) for x in rob['leave_one_out']) + '.')
L.append('- The matched small run (v65 R6) is a statistical tie at 3 paired seeds; the paper rows are single runs; serving readiness is a rubric.')
L.append('- Fusion retention is inert at this run length (upstream init 0.99996; see `fusion_state.json`), so the delta memory behaves like the plain delta rule of DLGM here.')
L.append('- Both main ports use the block-wide output norm instead of the per-head norm of the originals; `fusion879_hn.py` and `dlgm458_hn.py` restore a per-head norm for both as a robustness check (results in `runs/` when present).')
L.append('- `dlgm458.py` computes its memory in chunks of 16, Fusion needs 32: same mathematics for DLGM, different step time, so the speed columns are not strictly like for like.')
L.append('- "Equal time" is the simulator cost model at width 256. At width 8192 an operation count puts either mixer at about 6M multiply-adds per token against about 800M for the binary matrices of a block and about 34M for attention over 2,048 tokens; neither mixer has been timed in the real trainer.\n')
L.append('## Files\n')
L.append('- `plugins/fusion879.py` (and the byte-identical `fusion879_lr.py`; the file name selects the low-rank form): ASI-Arch #879 (delta_net_psfr) ported to the simulator block with the conventions of `plugins/dlgm458.py`.')
L.append('- `test_fusion879_equiv.py`, `fidelity_test_result.json`: the port against the original #879 class with copied weights (float64 on CPU, float32 on the GPU with the real FLA 0.2.2 kernels).')
L.append('- `index_spec_v1.json`: the frozen formula. `build_index.py`: builds the page and `index.json` from the runs. `index.json`, `page/index.html`: snapshot at packaging time.')
L.append('- `runs/`: every raw run JSON used (%d files). `census560.json`: storage bill of each form at 380 and 560 blocks.' % len(runs))
L.append('- `agillm_mini.py`: the simulator as run (sha256 prefix `dc5c3c28ddbf1853`, frozen since 9 Nov).\n')
L.append('## Reproduce\n')
L.append('```\npython3 -I agillm_mini.py --plugin plugins/fusion879.py --seed 1 --out fusion879__s1.json      # needs the simulator data set (train.npy, test.npy)\npython3 test_fusion879_equiv.py                                                              # needs flash-linear-attention 0.2.2, upstream data/106.json, a GPU for the float32 phase\n```\n')
L.append('Upstream sources: GAIR-NLP/ASI-Arch at commit 3113c51b (data/106.json, entries #458 and #879); paper arXiv 2507.18074v2, Table 9.\n')
L.append('Packaged %s by Claude for Scott.\n' % time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()))
open(DEST + '/README.md', 'w').write('\n'.join(L))
man = {}
for root, _, files in os.walk(DEST):
    for fn in sorted(files):
        p = os.path.join(root, fn)
        man[os.path.relpath(p, DEST)] = dict(bytes=os.path.getsize(p), sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest())
json.dump(man, open(DEST + '/FILE_MANIFEST.json', 'w'), indent=1)
print(DEST, len(man), 'files', sum(v['bytes'] for v in man.values()), 'bytes; runs', len(runs))
