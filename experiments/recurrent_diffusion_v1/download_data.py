"""Recreate exact public WikiText2 input files; no authenticated services."""
from pathlib import Path
import hashlib,json,urllib.request
import pyarrow.parquet as pq
P=Path(__file__).parent
m=json.loads((P/'source_data_manifest.json').read_text());(P/'data').mkdir(exist_ok=True)
for split,meta in m['splits'].items():
 out=P/'data'/f'{split}.txt'
 if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest()==meta['text_sha256']:continue
 raw=urllib.request.urlopen(meta['url'],timeout=120).read()
 assert hashlib.sha256(raw).hexdigest()==meta['parquet_sha256']
 parquet=P/'data'/f'{split}.parquet';parquet.write_bytes(raw)
 data='\n'.join(pq.read_table(parquet)['text'].to_pylist()).encode('utf-8')
 assert hashlib.sha256(data).hexdigest()==meta['text_sha256'];out.write_bytes(data)
 print(split,len(data),flush=True)
