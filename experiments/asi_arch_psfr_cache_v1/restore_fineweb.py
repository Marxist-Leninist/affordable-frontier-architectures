"""Restore the v44 package's pinned held-out FineWeb windows without pyarrow or run.py.

Mirrors prepare_inputs.py for the FineWeb half only: download the frozen datasets-server response,
check its x-revision header and SHA-256 against the package ledger, rebuild every evaluation window
and check each window hash and the whole-array hash. Writes into this work directory, never the package.
"""
import hashlib, json, sys, urllib.request
from pathlib import Path
import numpy as np

PKG = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()

req = json.loads((PKG / 'fineweb_request.json').read_text())
ledger = json.loads((PKG / 'ledger.json').read_text())
rows_file = OUT / 'fineweb_rows.json'
if not rows_file.exists():
    with urllib.request.urlopen(req['url'], timeout=90) as r:
        assert r.headers.get('x-revision') == req['headers']['x-revision'], r.headers.get('x-revision')
        body = r.read(4000001)
    assert len(body) <= 4000000
    rows_file.write_bytes(body)
body = rows_file.read_bytes()
assert sha(body) == ledger['fineweb_response_sha256'], 'FineWeb response hash mismatch'
rows = {r['row_idx']: r['row']['text'].encode() for r in json.loads(body)['rows']}
windows = []
for e in ledger['fineweb']:
    b = rows[e['doc']][e['start']:e['start'] + 513]
    assert sha(b) == e['window_sha256'] and sha(b[-128:]) == e['target_sha256']
    windows.append(list(b))
arr = np.array(windows, dtype=np.uint8)
assert sha(arr.tobytes()) == ledger['array_hashes']['fineweb'], 'array hash mismatch'
np.save(OUT / 'fineweb_eval.npy', arr)
long_docs = [list(t) for k, t in sorted(rows.items()) if len(t) >= 1100]
np.save(OUT / 'fineweb_long_docs.npy', np.array([d[:1100] for d in long_docs], dtype=np.uint8))
print(json.dumps({'fineweb_windows': list(arr.shape), 'long_docs_ge_1100_bytes': len(long_docs),
                  'response_sha256': ledger['fineweb_response_sha256'][:16], 'verified': True}))
