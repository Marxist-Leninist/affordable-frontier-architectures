"""chainmix512.py -- CHAINMIX512: a counted Markov chain mixed into the served
model's next-token distribution. OPT-IN, AR generation and /score.

A request field  chain_mix = g  (0 < g <= 0.98) makes the next-token distribution
    p = (1 - g) * p_model + g * p_chain
where p_chain is a counted Markov chain over the same tokenizer: the next token given the last two (interpolated absolute
discounting, orders 1-3, counts from plain text, no training). Absent or 0 = the model alone and nothing in this file runs.

Why: measured offline on the served 698B pack (step 7036), the first 128 targets of 2 held-out windows: model alone 7.84
nats per token, chain alone (116M fineweb-edu tokens) 5.96, the mixture at g = 0.5 5.82. The model at this stage has seen
about 15M tokens; counting 116M tokens takes 83 seconds.

Tables: a folder (env CHAINMIX512_DIR) of .npy files written by `python3 chainmix512.py build ...`, memory-mapped
read-only (a lookup touches a few pages). The count stream uses the worker's convention: documents separated by EOS.
numpy + torch only.
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

VERSION = 'chainmix512-v2-20261112'    # v2: the "measured" sentence comes from current.json / the tables, not from this file
FORMAT = 'chainmix-tables-v1'
FILES = ('c1', 'u2', 'n2', 'c2k', 'c2ptr', 'c2tot', 'c2n1', 'u3', 'n3', 'c3k', 'c3ptr', 'c3tot', 'c3n1')
G_MAX = 0.98
STATE = dict(tables=None, error='not installed', root=None, stamp=None, V=None, reload_error=None, measured=None)
NOT_MEASURED = 'not measured yet for the checkpoint and tables in use'


def _disc(cnt):
    n1 = float(np.count_nonzero(cnt == 1))
    n2 = float(np.count_nonzero(cnt == 2))
    return min(0.95, max(0.05, n1 / (n1 + 2.0 * n2))) if n1 > 0 and n2 > 0 else 0.75


def _merge(ka, ca, kb, cb):
    """sum the counts of two sorted (key, count) tables"""
    k = np.concatenate([ka, kb])
    c = np.concatenate([ca, cb])
    o = np.argsort(k, kind='stable')
    k, c = k[o], c[o]
    first = np.ones(len(k), dtype=bool)
    first[1:] = k[1:] != k[:-1]
    idx = np.flatnonzero(first)
    return k[idx], np.add.reduceat(c, idx)


def _grams(tok, V):
    t = np.asarray(tok).astype(np.int64)
    k2 = t[:-1] * V + t[1:]
    u2, n2 = np.unique(k2, return_counts=True)
    k3 = k2[:-1] * V + t[2:]
    del k2
    u3, n3 = np.unique(k3, return_counts=True)
    return u2, n2.astype(np.int64), u3, n3.astype(np.int64)


def build(tok, V, out, source='', delta=0.5, eos=None, extra=()):
    """Count tables for orders 1-3 from a token stream and write them to folder `out`.
    extra = [(token array, integer weight, label)]: further text whose counts are added `weight` times, to lean the chain
    toward a subject or a style; the discounts are estimated on the main stream."""
    assert float(V) ** 3 < 2.0 ** 63
    tok = np.asarray(tok)
    assert tok.min() >= 0 and tok.max() < V
    c1 = np.bincount(tok, minlength=V).astype(np.float64)
    u2, n2, u3, n3 = _grams(tok, V)
    D2, D3 = _disc(n2), _disc(n3)
    total = float(len(tok))
    extras = []
    for et, w, label in extra:
        et = np.asarray(et)
        assert et.min() >= 0 and et.max() < V and int(w) == w and w >= 1
        c1 += float(w) * np.bincount(et, minlength=V)
        e2, m2, e3, m3 = _grams(et, V)
        u2, n2 = _merge(u2, n2, e2, m2 * int(w))
        u3, n3 = _merge(u3, n3, e3, m3 * int(w))
        total += float(w) * len(et)
        extras.append(dict(label=str(label), tokens=int(len(et)), weight=int(w)))
    c2k, s2, t2 = np.unique(u2 // V, return_index=True, return_counts=True)
    c3k, s3, t3 = np.unique(u3 // V, return_index=True, return_counts=True)
    arrays = dict(c1=c1, u2=u2.astype(np.int64), n2=n2.astype(np.int32), c2k=c2k.astype(np.int64),
                  c2ptr=np.append(s2, len(u2)).astype(np.int64), c2tot=np.add.reduceat(n2, s2).astype(np.int64),
                  c2n1=t2.astype(np.int32), u3=u3.astype(np.int64), n3=n3.astype(np.int32), c3k=c3k.astype(np.int64),
                  c3ptr=np.append(s3, len(u3)).astype(np.int64), c3tot=np.add.reduceat(n3, s3).astype(np.int64),
                  c3n1=t3.astype(np.int32))
    os.makedirs(out, exist_ok=True)
    files = {}
    for name in FILES:
        path = os.path.join(out, name + '.npy')
        np.save(path, arrays[name])
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 22), b''):
                h.update(b)
        files[name] = dict(bytes=os.path.getsize(path), sha256=h.hexdigest())
    meta = dict(format=FORMAT, vocab=int(V), tokens=total, main_tokens=int(len(tok)), order=3, delta=float(delta), D2=D2, D3=D3,
                eos=(None if eos is None else int(eos)), source=str(source), extra=extras, bigram_types=int(len(u2)),
                trigram_types=int(len(u3)), built_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=files)
    meta['id'] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    return meta


class Tables:
    def __init__(self, folder, mmap=True):
        self.folder = str(folder)
        self.meta = json.load(open(os.path.join(self.folder, 'meta.json')))
        if self.meta.get('format') != FORMAT:
            raise ValueError('unknown chain table format: %r' % (self.meta.get('format'),))
        for name in FILES:
            path = os.path.join(self.folder, name + '.npy')
            if os.path.getsize(path) != int(self.meta['files'][name]['bytes']):
                raise ValueError('chain table %s has the wrong size' % name)
            setattr(self, name, np.load(path, mmap_mode='r' if mmap else None))
        self.V = int(self.meta['vocab'])
        self.N = float(self.meta['tokens'])
        self.delta = float(self.meta['delta'])
        self.D2, self.D3 = float(self.meta['D2']), float(self.meta['D3'])
        self.p1 = (np.asarray(self.c1, dtype=np.float64) + self.delta) / (self.N + self.delta * self.V)   # 1.6 MB in RAM

    @staticmethod
    def _row(keys, q):
        i = int(np.searchsorted(keys, q))
        return i if i < len(keys) and int(keys[i]) == int(q) else -1

    def next_dist(self, prev2, prev1):
        """P(next | prev2, prev1) as float64 [V]; prev2 = None when only one token of context exists."""
        V = self.V
        p = self.p1.copy()
        r = self._row(self.c2k, int(prev1))
        if r >= 0:
            a, b, tot = int(self.c2ptr[r]), int(self.c2ptr[r + 1]), float(self.c2tot[r])
            p *= self.D2 * float(self.c2n1[r]) / tot
            p[np.asarray(self.u2[a:b]) % V] += (np.asarray(self.n2[a:b], dtype=np.float64) - self.D2) / tot
        if prev2 is not None:
            r = self._row(self.c3k, int(prev2) * V + int(prev1))
            if r >= 0:
                a, b, tot = int(self.c3ptr[r]), int(self.c3ptr[r + 1]), float(self.c3tot[r])
                p *= self.D3 * float(self.c3n1[r]) / tot
                p[np.asarray(self.u3[a:b]) % V] += (np.asarray(self.n3[a:b], dtype=np.float64) - self.D3) / tot
        return p

    def target_prob(self, ids):
        """P(ids[j+1] | ids[j-1], ids[j]) for one sequence (list of ids), float64 [len-1]; position 0 has one context token."""
        return np.array([self.next_dist(ids[j - 1] if j >= 1 else None, ids[j])[ids[j + 1]] for j in range(len(ids) - 1)])

    def selftest(self):
        for ctx in ((None, 0), (None, int(self.c2k[len(self.c2k) // 2])), (int(self.c3k[len(self.c3k) // 2]) // self.V,
                                                                             int(self.c3k[len(self.c3k) // 2]) % self.V)):
            p = self.next_dist(*ctx)
            if not (abs(float(p.sum()) - 1.0) < 1e-6 and float(p.min()) > 0):
                raise ValueError('chain self-test failed for context %r: sum %.9f min %.3g' % (ctx, p.sum(), p.min()))


def _resolve(folder):
    """folder is either a table folder (meta.json inside) or a root holding
    current.json = {"tables": "<subfolder>", "measured": "<one sentence, optional>"}.
    Returns (table folder, root or None, stamp of current.json or None, the measured sentence or None)."""
    cur = os.path.join(folder, 'current.json')
    if os.path.exists(cur):
        st = os.stat(cur)
        with open(cur) as f:
            d = json.load(f)
        sub = str(d['tables'])
        if os.sep in sub or sub.startswith('.'):
            raise ValueError('current.json must name a plain subfolder')
        m = d.get('measured')
        return os.path.join(folder, sub), folder, (st.st_mtime_ns, st.st_size), (m if isinstance(m, str) and m.strip() else None)
    return folder, None, None, None


def _load(V, folder):
    path, root, stamp, measured = _resolve(folder)
    t = Tables(path)
    if t.V != int(V):
        raise ValueError('chain vocabulary %d != model vocabulary %d' % (t.V, int(V)))
    t.selftest()
    return t, root, stamp, measured


def install(V, folder=None):
    """Load the tables (never raises). Without CHAINMIX512_DIR, or on any error, chain_mix requests are refused with the reason."""
    folder = folder if folder is not None else os.getenv('CHAINMIX512_DIR', '').strip()
    STATE.update(root=None, stamp=None, V=int(V), reload_error=None, measured=None)
    if not folder:
        STATE.update(tables=None, error='CHAINMIX512_DIR is not set')
        return False
    try:
        t, root, stamp, measured = _load(V, folder)
        STATE.update(tables=t, error=None, root=root, stamp=stamp, measured=measured)
        return True
    except Exception as e:  # noqa: BLE001
        STATE.update(tables=None, error=repr(e)[:300])
        return False


def _maybe_reload():
    """New tables without a worker restart: when <root>/current.json changes, the tables it names are loaded and checked;
    only then do they replace the ones in use. A bad update leaves the current tables serving and records the reason."""
    root = STATE.get('root')
    if not root:
        return
    try:
        st = os.stat(os.path.join(root, 'current.json'))
        stamp = (st.st_mtime_ns, st.st_size)
        if stamp == STATE.get('stamp'):
            return
        STATE['stamp'] = stamp                       # one attempt per change
        t, _, _, measured = _load(STATE['V'], root)
        STATE.update(tables=t, error=None, reload_error=None, measured=measured)
    except Exception as e:  # noqa: BLE001
        STATE['reload_error'] = repr(e)[:300]


def info():
    _maybe_reload()                                  # /health shows the tables and the sentence current.json names now
    t = STATE['tables']
    measured = (STATE.get('measured') or t.meta.get('measured')) if t is not None else None
    d = dict(supported=t is not None, version=VERSION, modes=['ar', 'score'], max=G_MAX,
             field='chain_mix: number in [0, %.2f]; the next-token distribution becomes (1 - chain_mix) * model + chain_mix * chain; '
                   '0 or absent = the model alone' % G_MAX,
             measured=(str(measured)[:500] if measured else NOT_MEASURED))
    if t is None:
        d['reason'] = STATE['error']
    else:
        m = t.meta
        d['tables'] = {k: m.get(k) for k in ('id', 'vocab', 'tokens', 'order', 'source', 'extra', 'bigram_types', 'trigram_types', 'built_at_utc')}
        if m.get('estimator'):
            d['tables']['estimator'] = m['estimator']
    d['reloadable'] = bool(STATE.get('root'))        # tables can be replaced through <root>/current.json without a restart
    if STATE.get('reload_error'):
        d['reload_error'] = STATE['reload_error']
    return d


def parse(req, score=False):
    """None (no chain_mix field, 0, or a mode other than AR) or this request's plan. ValueError -> 400."""
    v = req.get('chain_mix')
    if v is None:
        return None
    if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= G_MAX:
        raise ValueError('chain_mix must be a number in [0, %.2f]' % G_MAX)
    if v == 0 or (not score and str(req.get('mode') or 'ar').lower() != 'ar'):
        return None
    _maybe_reload()
    if STATE['tables'] is None:
        raise ValueError('chain_mix: the chain tables are not available (%s)' % STATE['error'])
    return dict(g=float(v), tokens=0, t=STATE['tables'])   # one table set for the whole request, whatever is reloaded meanwhile


def mix_logits(cm, logits, hist):
    """log((1 - g) softmax(logits) + g P_chain(. | last two tokens of hist)) for one position, in the dtype of logits."""
    import torch
    if not hist:
        return logits
    t = cm.get('t') or STATE['tables']
    pc = torch.from_numpy(t.next_dist(hist[-2] if len(hist) >= 2 else None, hist[-1]))
    pm = torch.softmax(logits.detach().double().cpu(), dim=-1)
    cm['tokens'] += 1
    return ((1.0 - cm['g']) * pm + cm['g'] * pc).log().to(device=logits.device, dtype=logits.dtype)


def mix_rows(cm, lg, x, s):
    """/score: lg [T, V] are the logits for input positions s .. s+T-1 of the id list x; returns the mixture's log-probabilities."""
    import torch
    t = cm.get('t') or STATE['tables']
    out = torch.empty_like(lg)
    for j in range(lg.shape[0]):
        pos = s + j
        pc = torch.from_numpy(t.next_dist(x[pos - 1] if pos >= 1 else None, x[pos]))
        pm = torch.softmax(lg[j].double(), dim=-1)
        out[j] = ((1.0 - cm['g']) * pm + cm['g'] * pc).log().to(lg.dtype)
    cm['tokens'] += int(lg.shape[0])
    return out


def summary(cm):
    t = cm.get('t') or STATE['tables']
    return dict(g=cm['g'], tokens=cm['tokens'], table=(t.meta.get('id') if t is not None else None), version=VERSION)


def _main(argv):
    if len(argv) >= 4 and argv[1] == 'build':
        import argparse
        ap = argparse.ArgumentParser(prog='chainmix512.py build')
        ap.add_argument('tokens')
        ap.add_argument('out')
        ap.add_argument('--count', type=int, default=-1)
        ap.add_argument('--offset', type=int, default=128, help='bytes to skip (128 = a .npy header)')
        ap.add_argument('--vocab', type=int, default=200064)
        ap.add_argument('--eos', type=int, default=199999)
        ap.add_argument('--source', default='')
        ap.add_argument('--extra', action='append', default=[], help='FILE:WEIGHT:LABEL raw int32 token file (no header) counted WEIGHT times')
        a = ap.parse_args(argv[2:])
        t0 = time.time()
        tok = np.fromfile(a.tokens, dtype=np.int32, offset=a.offset, count=a.count)
        extra = []
        for spec in a.extra:
            f, w, label = spec.split(':', 2)
            extra.append((np.fromfile(f, dtype=np.int32), int(w), label))
        meta = build(tok, a.vocab, a.out, source=a.source or os.path.basename(a.tokens), eos=a.eos, extra=extra)
        t = Tables(a.out)
        t.selftest()
        print(json.dumps({k: meta[k] for k in ('id', 'tokens', 'main_tokens', 'bigram_types', 'trigram_types', 'D2', 'D3', 'extra')}),
              'bytes', sum(v['bytes'] for v in meta['files'].values()), 'seconds %.0f' % (time.time() - t0))
        print('CHAIN-BUILD-DONE')
        return 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(_main(sys.argv))
