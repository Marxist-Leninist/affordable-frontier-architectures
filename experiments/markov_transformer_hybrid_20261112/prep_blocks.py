#!/usr/bin/env python3
"""prep_blocks.py --name X -- document-aligned blocks of the train stream and the per-block count tables that
leave-one-block-out (cross-fitting) needs. Writes data/<name>/markov_blocks.npz and blocks_meta.json.

A block is a run of whole documents (stories / articles) of at least --min_tokens tokens. When the chain is evaluated
for a training position, every count contributed by that position's block is removed, so the chain has never seen the
document it is scoring, which is the situation it is in on test text.
"""
import argparse
import json
import os
import time

os.environ.setdefault('RAYON_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from markov import MarkovPrior, build_block_tables, save_block_cache  # noqa: E402


def wikitext_article_starts(out, train):
    """token index of every article start, by re-encoding with the exact chunking prep_data used"""
    from tokenizers import Tokenizer

    from prep_data import WT_DIR, arrow_lines
    lines = arrow_lines(os.path.join(WT_DIR, 'wikitext-train-00000-of-00002.arrow'), 20_000_000)
    docs, marks, cur, n = [], [], [], 0                 # marks: (chunk index, char offset in chunk) of a title line
    for s in lines:
        t = s.strip()
        if t.startswith('= ') and t.endswith(' =') and not t.startswith('= ='):
            marks.append((len(docs), n))
        cur.append(s)
        n += len(s)
        if n >= 20000:
            docs.append(''.join(cur))
            cur, n = [], 0
    if cur:
        docs.append(''.join(cur))
    tok = Tokenizer.from_file(os.path.join(out, 'tokenizer.json'))
    base, tok_starts, ids = [], [], []
    total = 0
    for i in range(0, len(docs), 2000):
        for e in tok.encode_batch(docs[i:i + 2000]):
            base.append(total)
            tok_starts.append(np.asarray([o[0] for o in e.offsets], dtype=np.int64))
            ids.append(np.asarray(e.ids, dtype=np.int64))
            total += len(e.ids)
    assert np.array_equal(np.concatenate(ids), train.astype(np.int64)), 're-encoding does not reproduce train.npy'
    starts = sorted(set([0] + [int(base[c] + np.searchsorted(tok_starts[c], off, 'left')) for c, off in marks]))
    return [s for s in starts if s < len(train)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, choices=['tinystories', 'wikitext'])
    ap.add_argument('--min_tokens', type=int, default=16384)
    ap.add_argument('--T', type=int, default=128)
    ap.add_argument('--out', default='data')
    a = ap.parse_args()
    t0 = time.time()
    out = os.path.join(a.out, a.name)
    meta = json.load(open(os.path.join(out, 'meta.json')))
    V, order = meta['V'], meta['order']
    train = np.load(os.path.join(out, 'train.npy'))
    N = len(train)
    if a.name == 'tinystories':
        starts = [0] + (np.nonzero(train == meta['eot_id'])[0] + 1).tolist()      # a story ends with the EOT token
        starts = [s for s in starts if s < N]
    else:
        starts = wikitext_article_starts(out, train)
    starts = np.asarray(sorted(set(starts)), dtype=np.int64)
    lens = np.diff(np.append(starts, N))
    blk_of_doc, b, acc = np.zeros(len(starts), dtype=np.int64), 0, 0
    for i, n in enumerate(lens):
        blk_of_doc[i] = b
        acc += n
        if acc >= a.min_tokens:
            b, acc = b + 1, 0
    blk = np.repeat(blk_of_doc, lens)
    assert len(blk) == N
    nb = int(blk.max()) + 1
    sizes = np.bincount(blk)
    print(f'{len(starts)} documents -> {nb} blocks, tokens per block min {sizes.min()} median {int(np.median(sizes))} '
          f'max {sizes.max()}  {time.time() - t0:.1f}s', flush=True)
    tables = np.load(os.path.join(out, 'markov.npz'))
    bt = build_block_tables(train, V, order, blk, tables)
    save_block_cache(bt, os.path.join(out, 'blocks_cache'), order)
    print('block tables built', {k: int(len(bt[f'bfull{k}'])) for k in range(2, order + 1)}, f'{time.time() - t0:.1f}s', flush=True)

    # does the chain, scored block-out on TRAIN text, now look like the chain on NEW text?
    ids = torch.from_numpy(train.astype(np.int64))
    g = torch.Generator().manual_seed(0)
    st = torch.randint(0, N - a.T - 1, (1000,), generator=g)
    res = {}
    for o in range(2, order + 1):
        pr = MarkovPrior(tables, o, btables=os.path.join(out, 'blocks_cache'))      # the files the runs will use
        tot = {'lobo': 0.0, 'loo': 0.0, 'full': 0.0}
        for s in range(0, len(st), 50):
            idx = st[s:s + 50, None] + torch.arange(a.T)[None, :]
            x, y = ids[idx], ids[idx + 1]
            for m in tot:
                tot[m] += float(-pr.target_logprob(x, y, mode=m, tpos=idx + 1).sum())
        res[str(o)] = {'train_' + m: v / (len(st) * a.T) for m, v in tot.items()}
        res[str(o)].update({'dev': meta['chain_alone_loss'][str(o)]['dev'], 'test': meta['chain_alone_loss'][str(o)]['test']})
        print('chain alone, order', o, {k: round(v, 4) for k, v in res[str(o)].items()}, flush=True)
    json.dump({'documents': int(len(starts)), 'blocks': nb, 'min_tokens': a.min_tokens,
               'block_tokens': {'min': int(sizes.min()), 'median': int(np.median(sizes)), 'max': int(sizes.max())},
               'entries': {str(k): int(len(bt[f'bfull{k}'])) for k in range(2, order + 1)}, 'chain_alone_loss': res,
               'seconds': round(time.time() - t0, 1)}, open(os.path.join(out, 'blocks_meta.json'), 'w'), indent=1)
    print('DONE', f'{time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
