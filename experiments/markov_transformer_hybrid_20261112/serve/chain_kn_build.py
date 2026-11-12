#!/usr/bin/env python3
"""chain_kn_build.py TOKENS OUT [--count N] [--extra FILE:WEIGHT:LABEL] -- chain tables with Kneser-Ney lower orders,
written in the SAME file format as chainmix512.build(), so the worker that is already running loads them as they are
(no new code in the worker, no restart: current.json names the folder).

What changes against the plain tables: the trigram level is the same (how often y followed a, b). The level it backs
off to no longer asks "how often did y follow b" but "after how many different tokens a was the pair b, y seen", and
the lowest level asks "after how many different tokens was y seen". A token that is frequent only inside a few fixed
phrases ("Francisco" after "San") stops being offered after every other context. One discount per level, as before.
On the simulator's stream (12.6M counted tokens, 8,192-token vocabulary, held-out windows): trigram 4.683 -> 4.574 nats.

In the file format: n2 holds the continuation count of each pair instead of its frequency, c1 the continuation count of
each token scaled so that it sums to the number of counted tokens (the format's unigram formula then normalises), and
the pair that no trigram ends in (the first two tokens of a stream) is absent. D2 is estimated on the continuation
counts.

Extra text counted `weight` times: its trigram counts are multiplied, and a trigram type that occurs in it counts
`weight` times in the continuation counts. The discounts are estimated on the main stream.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chainmix512 as cmx  # noqa: E402

ESTIMATOR = 'interpolated Kneser-Ney: continuation counts at the bigram and unigram level, one discount per level'


def _starts(sorted_ctx):
    first = np.ones(len(sorted_ctx), dtype=bool)
    first[1:] = sorted_ctx[1:] != sorted_ctx[:-1]
    return np.flatnonzero(first)


def _cont(u3, tw, V):
    """for each pair (b, y): the summed weights of the distinct trigram types (a, b, y); weight 1 each when tw is None"""
    suf = u3 % (V * V)
    if tw is None:
        return np.unique(suf, return_counts=True)
    o = np.argsort(suf, kind='stable')
    ss = suf[o]
    del suf
    idx = _starts(ss)
    return ss[idx], np.add.reduceat(tw[o], idx)


def build_kn(tok, V, out, source='', delta=0.5, eos=None, extra=()):
    assert float(V) ** 3 < 2.0 ** 63
    tok = np.asarray(tok)
    assert tok.min() >= 0 and tok.max() < V
    u2, n2, u3, n3 = cmx._grams(tok, V)
    del u2, n2
    D3 = cmx._disc(n3)
    kk, kn = _cont(u3, None, V)
    D2 = cmx._disc(kn)
    total = float(len(tok))
    tw, extras = None, []
    for et, w, label in extra:
        et = np.asarray(et)
        assert et.min() >= 0 and et.max() < V and int(w) == w and w >= 1
        e2, m2, e3, m3 = cmx._grams(et, V)
        del e2, m2
        u3n, n3 = cmx._merge(u3, n3, e3, m3 * int(w))
        twn = np.ones(len(u3n), dtype=np.int64)
        if tw is not None:
            twn[np.searchsorted(u3n, u3)] = tw
        pos = np.searchsorted(u3n, e3)
        twn[pos] = np.maximum(twn[pos], int(w))
        u3, tw = u3n, twn
        total += float(w) * len(et)
        extras.append(dict(label=str(label), tokens=int(len(et)), weight=int(w)))
    if tw is not None:
        kk, kn = _cont(u3, tw, V)
    del tw
    k1 = np.bincount(kk % V, minlength=V).astype(np.float64)        # after how many different tokens each token was seen
    c1 = k1 * (total / k1.sum())
    ctx2 = kk // V
    s2 = _starts(ctx2)
    c2k = ctx2[s2]
    del ctx2
    ctx3 = u3 // V
    s3 = _starts(ctx3)
    c3k = ctx3[s3]
    del ctx3
    arrays = dict(c1=c1, u2=kk.astype(np.int64), n2=kn.astype(np.int32), c2k=c2k.astype(np.int64),
                  c2ptr=np.append(s2, len(kk)).astype(np.int64), c2tot=np.add.reduceat(kn, s2).astype(np.int64),
                  c2n1=np.diff(np.append(s2, len(kk))).astype(np.int32), u3=u3.astype(np.int64), n3=n3.astype(np.int32),
                  c3k=c3k.astype(np.int64), c3ptr=np.append(s3, len(u3)).astype(np.int64),
                  c3tot=np.add.reduceat(n3, s3).astype(np.int64), c3n1=np.diff(np.append(s3, len(u3))).astype(np.int32))
    os.makedirs(out, exist_ok=True)
    files = {}
    for name in cmx.FILES:
        path = os.path.join(out, name + '.npy')
        np.save(path, arrays[name])
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 22), b''):
                h.update(b)
        files[name] = dict(bytes=os.path.getsize(path), sha256=h.hexdigest())
    meta = dict(format=cmx.FORMAT, vocab=int(V), tokens=total, main_tokens=int(len(tok)), order=3, delta=float(delta), D2=D2, D3=D3,
                eos=(None if eos is None else int(eos)), source=str(source), extra=extras, bigram_types=int(len(kk)),
                trigram_types=int(len(u3)), estimator=ESTIMATOR, built_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=files)
    meta['id'] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    return meta


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog='chain_kn_build.py')
    ap.add_argument('tokens')
    ap.add_argument('out')
    ap.add_argument('--count', type=int, default=-1)
    ap.add_argument('--offset', type=int, default=128, help='bytes to skip (128 = a .npy header)')
    ap.add_argument('--vocab', type=int, default=200064)
    ap.add_argument('--eos', type=int, default=199999)
    ap.add_argument('--source', default='')
    ap.add_argument('--extra', action='append', default=[], help='FILE:WEIGHT:LABEL raw int32 token file (no header) counted WEIGHT times')
    a = ap.parse_args(argv[1:])
    t0 = time.time()
    tok = np.fromfile(a.tokens, dtype=np.int32, offset=a.offset, count=a.count)
    extra = []
    for spec in a.extra:
        f, w, label = spec.split(':', 2)
        extra.append((np.fromfile(f, dtype=np.int32), int(w), label))
    meta = build_kn(tok, a.vocab, a.out, source=a.source or os.path.basename(a.tokens), eos=a.eos, extra=extra)
    t = cmx.Tables(a.out)                                   # the worker's own loader and self-test
    t.selftest()
    print(json.dumps({k: meta[k] for k in ('id', 'tokens', 'main_tokens', 'bigram_types', 'trigram_types', 'D2', 'D3', 'extra', 'estimator')}),
          'bytes', sum(v['bytes'] for v in meta['files'].values()), 'seconds %.0f' % (time.time() - t0))
    print('CHAIN-KN-BUILD-DONE')
    return 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv))
