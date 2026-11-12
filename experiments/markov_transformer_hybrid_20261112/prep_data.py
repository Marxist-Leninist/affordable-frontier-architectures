#!/usr/bin/env python3
"""prep_data.py -- token streams and Markov count tables for the Markov-chain + transformer experiment.

  python3 prep_data.py --name tinystories     TinyStories V2 (GPT-4): first 20 MB of train; dev/test from the valid file
  python3 prep_data.py --name wikitext        WikiText-103 raw: first 20 MB of train; the official validation / test

A byte-level BPE tokenizer is trained on the TRAIN text only. Outputs under data/<name>/:
  train.npy dev.npy test.npy (uint16), tokenizer.json, markov.npz (counts from train only), meta.json
"""
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault('RAYON_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from markov import MarkovPrior, build_tables  # noqa: E402

TS_TRAIN = '/root/.cache/bintf/roneneldan_TinyStories_TinyStoriesV2-GPT4-train.txt_20MB'
TS_VALID = '/root/.cache/bintf/roneneldan_TinyStories_TinyStoriesV2-GPT4-valid.txt_20MB'
WT_DIR = ('/root/.cache/huggingface/datasets/Salesforce___wikitext/wikitext-103-raw-v1/0.0.0/'
          'b08601e04326c79dfdd32d625aee71d232d685c3')
EOT = '<|endoftext|>'


def stories(path):
    txt = open(path, 'rb').read().decode('utf-8', 'replace')
    parts = txt.split(EOT)
    parts = parts[:-1]                       # the file is cut at a byte count: drop the last, partial story
    return [p.strip() for p in parts if p.strip()]


def arrow_lines(path, max_bytes=None):
    import pyarrow as pa
    out, n = [], 0
    with pa.memory_map(path) as src:
        reader = pa.ipc.open_stream(src)
        for batch in reader:
            for s in batch.column(batch.schema.get_field_index('text')).to_pylist():
                out.append(s)
                n += len(s)
                if max_bytes and n >= max_bytes:
                    return out
    return out


def chunk_lines(lines, size=20000):
    docs, cur, n = [], [], 0
    for s in lines:
        cur.append(s)
        n += len(s)
        if n >= size:
            docs.append(''.join(cur))
            cur, n = [], 0
    if cur:
        docs.append(''.join(cur))
    return docs


def encode(tok, docs, sep_id):
    out = []
    for i in range(0, len(docs), 2000):
        for e in tok.encode_batch(docs[i:i + 2000]):
            out.append(np.asarray(e.ids, dtype=np.int64))
            if sep_id is not None:
                out.append(np.asarray([sep_id], dtype=np.int64))
    return np.concatenate(out)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def chain_loss(prior, ids, T, max_windows, loo=False, seed=0):
    """mean loss of the chain alone over windows of T tokens (non-overlapping from the start; random for loo/train)"""
    ids = torch.from_numpy(ids.astype(np.int64))
    nw = min(max_windows, (len(ids) - 1) // T)
    if loo:
        g = torch.Generator().manual_seed(seed)
        starts = torch.randint(0, len(ids) - T - 1, (nw,), generator=g)
    else:
        starts = torch.arange(nw) * T
    tot = 0.0
    for s in range(0, nw, 64):
        idx = starts[s:s + 64, None] + torch.arange(T)[None, :]
        tot += float(-prior.target_logprob(ids[idx], ids[idx + 1], loo=loo).sum())
    return tot / (nw * T), nw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, choices=['tinystories', 'wikitext'])
    ap.add_argument('--vocab', type=int, default=2048)
    ap.add_argument('--order', type=int, default=4)
    ap.add_argument('--T', type=int, default=128)
    ap.add_argument('--out', default='data')
    a = ap.parse_args()
    t0 = time.time()
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    meta = {'name': a.name, 'vocab': a.vocab, 'order': a.order}

    if a.name == 'tinystories':
        tr = stories(TS_TRAIN)
        va = stories(TS_VALID)
        seen = set(hashlib.md5(s.encode()).digest() for s in tr)
        va2 = [s for s in va if hashlib.md5(s.encode()).digest() not in seen]
        meta['valid_stories_also_in_train_removed'] = len(va) - len(va2)
        dev, test, n = [], [], 0
        for s in va2:
            if n < 1_000_000:
                dev.append(s)
            elif n < 3_000_000:
                test.append(s)
            else:
                break
            n += len(s)
        docs = {'train': tr, 'dev': dev, 'test': test}
        sep = True
        meta['sources'] = {'train': TS_TRAIN, 'dev_test': TS_VALID}
    else:
        docs = {'train': chunk_lines(arrow_lines(os.path.join(WT_DIR, 'wikitext-train-00000-of-00002.arrow'), 20_000_000)),
                'dev': chunk_lines(arrow_lines(os.path.join(WT_DIR, 'wikitext-validation.arrow'))),
                'test': chunk_lines(arrow_lines(os.path.join(WT_DIR, 'wikitext-test.arrow')))}
        sep = False
        meta['sources'] = {'dir': WT_DIR}
    meta['chars'] = {k: int(sum(len(s) for s in v)) for k, v in docs.items()}
    meta['docs'] = {k: len(v) for k, v in docs.items()}
    print('text loaded', meta['chars'], meta['docs'], f'{time.time() - t0:.1f}s', flush=True)

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=[EOT], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs['train'], trainer)          # TRAIN text only
    tok.save(os.path.join(out, 'tokenizer.json'))
    V = tok.get_vocab_size()
    eot = tok.token_to_id(EOT)
    assert V <= a.vocab and V <= 65535
    print('tokenizer trained, vocab', V, f'{time.time() - t0:.1f}s', flush=True)

    arr = {}
    for k, v in docs.items():
        ids = encode(tok, v, eot if sep else None)
        assert ids.max() < V
        arr[k] = ids.astype(np.uint16)
        np.save(os.path.join(out, k + '.npy'), arr[k])
    meta['V'] = int(V)
    meta['eot_id'] = int(eot)
    meta['tokens'] = {k: int(len(v)) for k, v in arr.items()}
    meta['chars_per_token'] = {k: round(meta['chars'][k] / len(arr[k]), 3) for k in arr}
    meta['sample_decode'] = tok.decode(arr['dev'][:60].tolist())
    print('encoded', meta['tokens'], meta['chars_per_token'], f'{time.time() - t0:.1f}s', flush=True)

    tables = build_tables(arr['train'], V, a.order)          # counts from TRAIN only
    np.savez(os.path.join(out, 'markov.npz'), **tables)
    meta['discounts'] = {str(k + 2): float(d) for k, d in enumerate(tables['D'])}
    meta['table_entries'] = {'2': int((tables['C2'] > 0).sum()),
                             **{str(k): int(len(tables[f'ufull{k}'])) for k in range(3, a.order + 1)}}
    print('markov tables built', meta['table_entries'], meta['discounts'], f'{time.time() - t0:.1f}s', flush=True)

    # the chain on its own: the reference line every transformer result is read against
    chain = {}
    for order in range(2, a.order + 1):
        pr = MarkovPrior(tables, order)
        row = {}
        for split in ('dev', 'test'):
            row[split], row[split + '_windows'] = chain_loss(pr, arr[split], a.T, 10 ** 9)
        row['train_loo'], _ = chain_loss(pr, arr['train'], a.T, 1000, loo=True)
        row['train_no_loo'], _ = chain_loss(pr, arr['train'][:1000 * a.T + 1], a.T, 1000)
        chain[str(order)] = row
        print('chain alone, order', order, {k: round(v, 4) for k, v in row.items()}, flush=True)
    meta['chain_alone_loss'] = chain
    meta['unigram_entropy'] = float(-(pr.P1.double() * pr.P1.double().log()).sum())
    meta['sha256'] = {f: sha(os.path.join(out, f)) for f in ('train.npy', 'dev.npy', 'test.npy', 'markov.npz', 'tokenizer.json')}
    meta['seconds'] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('DONE', out, f'{time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
