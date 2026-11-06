"""Restore exact official snapshots; fail closed on changed revision or bytes."""
from pathlib import Path
import json,hashlib,urllib.request,sys
P=Path(__file__).resolve().parent;REV='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
def fetch(path,offset,length,expected):
 if not path.exists():
  url=f'https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW%2Ffineweb-edu&config=sample-10BT&split=train&offset={offset}&length={length}'
  with urllib.request.urlopen(url,timeout=60) as r:
   assert r.headers.get('x-revision')==REV,'Source revision changed';raw=r.read(2000001);assert len(raw)<=2000000
  assert hashlib.sha256(raw).hexdigest()==expected,'Snapshot response changed; do not refreeze';path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(raw)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Cached snapshot mismatch'
def main():
 spec=json.loads((P/'protocol.json').read_text());fetch(P/'rows.json',1700,100,spec['recipient_response_sha256'])
 for r in json.loads((P/'prior_download_manifest.json').read_text()):fetch(P.parent/r['directory']/'rows.json',r['offset'],r['length'],r['sha256'])
 print('Fresh and17 prior snapshots hash-verified; maximum missing download budget36MB')
if __name__=='__main__':main()
