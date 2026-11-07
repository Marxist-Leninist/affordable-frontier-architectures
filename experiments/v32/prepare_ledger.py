"""Freeze pre-score provenance and exact-overlap exclusions; no loss inspection."""
from run import *

def main():
    assert not (P / 'protocol.json').exists()
    source = json.loads((P / 'training_data_manifest.json').read_text())
    train = (TRAIN / 'train.txt').read_bytes()
    val = (TRAIN / 'validation.txt').read_bytes()
    for name, raw in [('train', train), ('validation', val)]:
        assert sha_bytes(raw) == source['splits'][name]['text_sha256']
    prior, receipts = {}, []
    for directory in ('frontier_research_v*', 'anti_latency_v*'):
        for file in sorted(P.parent.glob(directory + '/rows.json')):
            if file.parent == P:
                continue
            receipts.append({'directory': file.parent.name, 'response_sha256': sha(file)})
            for row in json.loads(file.read_text())['rows']:
                raw = row['row']['text'].encode()
                prior[sha_bytes(raw)] = raw
    validation, excluded_val = [], []
    for start in range(0, len(val)-4096, 4096):
        raw = val[start:start+4097]
        if raw[-128:] in train:
            excluded_val.append(start)
            continue
        validation.append(dict(doc=0, start=start, window_sha256=sha_bytes(raw), target_sha256=sha_bytes(raw[-128:])))
        if len(validation) == 32:
            break
    assert len(validation) == 32
    rows = json.loads((P / 'rows.json').read_text())['rows']
    texts = {r['row_idx']: r['row']['text'].encode() for r in rows}
    items, excluded, target_seen, full_seen = [], [], set(), set()
    for doc, full in texts.items():
        full_hash = sha_bytes(full)
        reason = 'duplicate_full_document' if full_hash in prior or full_hash in full_seen else None
        full_seen.add(full_hash)
        for start in range(0, min(len(full), 8193)-4096, 4096):
            raw = full[start:start+4097]
            target = raw[-128:]
            why = reason
            if why is None and target in train:
                why = 'target_in_training'
            if why is None and target in val:
                why = 'target_in_validation'
            if why is None and any(target in old for old in prior.values()):
                why = 'target_in_prior_evaluation'
            if why is None and (sha_bytes(target) in target_seen or any(target in other for other_doc,other in texts.items() if other_doc != doc)):
                why = 'duplicate_current_target'
            if why is not None:
                excluded.append(dict(doc=doc,start=start,reason=why))
                continue
            target_seen.add(sha_bytes(target))
            items.append(dict(doc=doc,start=start,window_sha256=sha_bytes(raw),target_sha256=sha_bytes(target)))
    assert len(items) >= 20
    write('items.json', dict(validation=validation,evaluation=items))
    write('provenance.json', dict(training_hash=sha_bytes(train),validation_hash=sha_bytes(val),validation_windows=32,validation_targets=4096,validation_excluded_starts=excluded_val,prior_unique_documents=len(prior),prior_receipts=receipts,evaluation_windows=len(items),evaluation_documents=len({x['doc'] for x in items}),evaluation_targets=128*len(items),evaluation_exclusions=excluded,exact_target_overlap_after_exclusions=False,limitations='Exact128-byte spans and whole documents only. Shorter/near-duplicate/semantic overlap unmeasured; deterministic minimum-length cohort.'))
    print('LEDGER',len(items), 'windows',len({x['doc'] for x in items}),'documents; excluded',excluded)
if __name__ == '__main__':
    main()
