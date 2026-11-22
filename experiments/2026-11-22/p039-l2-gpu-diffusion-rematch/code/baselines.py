#!/usr/bin/env python3
"""P039-L CPU baselines on exactly the same cases as the neural arms (numpy only, no GPU).
- all-space and train-unigram-mode completions (the trivial baselines that tied P039's fastest arm)
- interpolated Kneser-Ney byte n-grams, orders 1..12, trained on P045 train.bin only: teacher-forced suffix NLL
  (bits/byte) and greedy argmax completion of the 16 bytes. Contexts are 56-bit polynomial hashes (a context of
  up to 11 bytes; a hash collision merges two contexts and only weakens the baseline). The maximum order is chosen
  on the dev cases (lowest dev suffix bits) before the test cases are scored.
Usage: python3 baselines.py [results_dir] [private_dir]"""
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p039l import load_data, windows, make_cases, macro, gen_metrics, jdump, utc, PROMPT, SUFFIX, ROOT

U64 = np.uint64
NMAX = 12
PRIME = U64(0x9E3779B97F4A7C15)
SH8 = U64(8)


def ctx_hash(cols):
    """cols: list of uint64 arrays (oldest context byte first). Returns the 56-bit context id (hash >> 8)."""
    h = np.zeros(len(cols[0]) if cols else 0, U64)
    for c in cols:
        h = h * PRIME + c + U64(1)
    h = h ^ (h >> U64(33)); h = h * U64(0xFF51AFD7ED558CCD); h = h ^ (h >> U64(33))   # splitmix64 finaliser: every
    h = h * U64(0xC4CEB9FE1A85EC53); h = h ^ (h >> U64(33))                            # byte reaches the kept top bits
    return h >> SH8


class KN:
    """Interpolated Kneser-Ney: raw counts at the top order, continuation counts N1+(. h w) below it, uniform at the base."""
    def __init__(self, train):
        t0 = time.time(); b = train.astype(U64); N = len(b); I = np.arange(NMAX - 1, N)   # target positions, all orders aligned
        w = b[I]; keys = {}
        for n in range(1, NMAX + 1):
            m = n - 1
            ctx = ctx_hash([b[I - m + j] for j in range(m)]) if m else np.zeros(len(I), U64)
            keys[n] = (ctx << SH8) | w
        self.raw = {}; self.cont = {}
        for n in range(1, NMAX + 1):
            u, first, c = np.unique(keys[n], return_index=True, return_counts=True)
            self.raw[n] = self.stats(u, c)
            if n > 1:                                   # (n)-grams -> continuation counts of their (n-1)-suffix
                su, sc = np.unique(keys[n - 1][first], return_counts=True)
                self.cont[n - 1] = self.stats(su, sc)
        self.build_s = time.time() - t0

    @staticmethod
    def stats(keys, counts):
        ctx = keys >> SH8
        uc, start, ntypes = np.unique(ctx, return_index=True, return_counts=True)
        tot = np.add.reduceat(counts, start)
        n1 = int((counts == 1).sum()); n2 = int((counts == 2).sum())
        D = n1 / (n1 + 2 * n2) if n1 + 2 * n2 > 0 else 0.5
        return {'keys': keys, 'counts': counts.astype(np.float64), 'ctx': uc, 'tot': tot.astype(np.float64),
                'types': ntypes.astype(np.float64), 'D': D}

    @staticmethod
    def look(st, ctx, w):
        key = (ctx << SH8) | w
        i = np.minimum(np.searchsorted(st['keys'], key), len(st['keys']) - 1)
        num = np.where(st['keys'][i] == key, st['counts'][i], 0.0)
        j = np.minimum(np.searchsorted(st['ctx'], ctx), len(st['ctx']) - 1)
        hit = st['ctx'][j] == ctx
        return num, np.where(hit, st['tot'][j], 0.0), np.where(hit, st['types'][j], 0.0)

    def prob(self, ctxs, w, N):
        """P_KN(w | previous N-1 bytes); ctxs[k] = context id array for order k (its k-1 previous bytes)."""
        p = np.full(len(w), 1.0 / 256)
        for k in range(1, N + 1):
            st = self.raw[k] if k == N else self.cont[k]
            num, den, types = self.look(st, ctxs[k], w)
            safe = np.maximum(den, 1.0)
            q = np.maximum(num - st['D'], 0.0) / safe + st['D'] * types / safe * p
            p = np.where(den > 0, q, p)
        return p


def ctxs_at(seq, p, N):
    """Context ids for orders 1..N at target position p of each row of seq (int64, >= NMAX-1 bytes before p)."""
    s = seq.astype(U64); out = {1: np.zeros(len(seq), U64)}
    for k in range(2, N + 1):
        m = k - 1
        out[k] = ctx_hash([s[:, p - m + j] for j in range(m)])
    return out


def suffix_bits(kn, X, N):
    tot = np.zeros(len(X))
    for j in range(SUFFIX):
        p = PROMPT + j
        tot += -np.log2(kn.prob(ctxs_at(X, p, N), X[:, p].astype(U64), N))
    return tot / SUFFIX


def greedy(kn, X, N, chunk=256):
    seq = X.copy(); out = np.zeros((len(X), SUFFIX), np.int64); W = np.arange(256, dtype=U64)
    for j in range(SUFFIX):
        p = PROMPT + j; cx = ctxs_at(seq, p, N)
        for i in range(0, len(X), chunk):
            n = min(chunk, len(X) - i)
            cc = {k: np.repeat(v[i:i + n], 256) for k, v in cx.items()}
            out[i:i + n, j] = kn.prob(cc, np.tile(W, n), N).reshape(n, 256).argmax(1)
        seq[:, p] = out[:, j]
    return out


def main():
    res_dir = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'results'); priv = ROOT / (sys.argv[2] if len(sys.argv) > 2 else 'private')
    res_dir.mkdir(exist_ok=True); priv.mkdir(exist_ok=True)
    raw = load_data(ROOT / 'data')
    train = np.frombuffer(raw['train.bin'], np.uint8)
    dX, dmeta = make_cases(windows(raw['dev_windows.json']))
    X, meta = make_cases(windows(raw['test_windows.json']))
    Y = X[:, PROMPT:]
    kn = KN(train)
    W = np.arange(256, dtype=U64); norm = []
    for N in range(1, NMAX + 1):                        # every order's distribution must sum to 1
        cx = ctxs_at(dX[:4], PROMPT, N); cc = {k: np.repeat(v, 256) for k, v in cx.items()}
        norm.append(float(np.abs(kn.prob(cc, np.tile(W, 4), N).reshape(4, 256).sum(1) - 1).max()))
    assert max(norm) < 1e-9, norm
    res = {'utc': utc(), 'build_s': kn.build_s, 'max_order': NMAX, 'normalisation_max_err': norm, 'hash_bits': 56,
           'discounts_top': {n: kn.raw[n]['D'] for n in kn.raw}, 'dev_bits': {}, 'test_bits': {}, 'gen': {}}
    for N in range(1, NMAX + 1):
        res['dev_bits'][N] = macro(suffix_bits(kn, dX, N).tolist(), dmeta)[0]
    best = min(res['dev_bits'], key=res['dev_bits'].get); res['best_order_on_dev'] = best
    ref = ROOT / 'results' / 'baselines_kn8_v1.json'              # exact 64-bit packing implementation, orders <= 8
    if ref.exists():
        exact = json.loads(ref.read_text())['dev_bits']
        res['hash_vs_exact_dev_bits_max_abs'] = max(abs(res['dev_bits'][n] - exact[str(n)]) for n in range(1, 9))
        assert res['hash_vs_exact_dev_bits_max_abs'] < 0.01, res['hash_vs_exact_dev_bits_max_abs']
    arrays = {}
    for N in range(1, NMAX + 1):
        v = suffix_bits(kn, X, N)
        res['test_bits'][N] = dict(zip(('macro', 'domain'), macro(v.tolist(), meta)))
        if N == best:
            arrays['nll_kn'] = v
    res['gen']['all_space'] = gen_metrics(np.full_like(Y, 32), Y, meta)[0]
    mode = int(np.bincount(train, minlength=256).argmax()); res['unigram_mode_byte'] = mode
    res['gen']['unigram_mode'] = gen_metrics(np.full_like(Y, mode), Y, meta)[0]
    t0 = time.time(); P = greedy(kn, X, best); res['greedy_s'] = time.time() - t0
    res['gen'][f'kn{best}_greedy'] = gen_metrics(P, Y, meta)[0]; arrays['pred_kn_greedy'] = P.astype(np.uint8)
    np.savez_compressed(priv / 'cases_baselines.npz', **arrays)
    jdump(res_dir / 'baselines.json', res)
    print(json.dumps({k: res[k] for k in ('build_s', 'best_order_on_dev', 'dev_bits', 'greedy_s')}), flush=True)
    g = res['gen'][f'kn{best}_greedy']
    print('test bits best', res['test_bits'][best]['macro'], 'greedy byte', g['byte_acc']['macro'], 'nonspace', g['nonspace_acc']['macro'], flush=True)


if __name__ == '__main__':
    main()
