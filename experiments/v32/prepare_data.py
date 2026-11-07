"""Bounded HF viewer snapshot with fail-closed revision and content checks."""
import json,hashlib,urllib.request
from pathlib import Path
P=Path(__file__).parent
URL='https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW%2Ffineweb-edu&config=sample-10BT&split=train&offset=1800&length=100'
REV='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
def sha(x):return hashlib.sha256(x).hexdigest()
def main():
 expected=json.loads((P/'data_manifest.json').read_text()) if (P/'data_manifest.json').exists() else None
 path=P/'rows.json'
 if not path.exists():
  with urllib.request.urlopen(URL,timeout=60) as response:
   if response.headers.get('x-revision')!=REV:raise ValueError('Viewer revision changed; do not silently benchmark new rows')
   data=response.read(2_000_001)
   if len(data)>2_000_000:raise ValueError('Bounded download exceeded')
  path.write_bytes(data)
 data=path.read_bytes();j=json.loads(data)
 assert not j['partial'];assert len(j['rows'])==100
 docs=[]
 for i,row in enumerate(j['rows']):
  assert row['row_idx']==i+1800 and not row['truncated_cells'];raw=row['row']['text'].encode('utf-8');capped=raw[:8193]
  docs.append(dict(row_index=i+1800,full_text_sha256=sha(raw),capped_text_sha256=sha(capped),original_bytes=len(raw),used_bytes=len(capped)))
 manifest=dict(dataset='HuggingFaceFW/fineweb-edu',config='sample-10BT',split='train',viewer_url=URL,observed_x_revision=REV,response_sha256=sha(data),response_bytes=len(data),row_count=100,cap_bytes_per_document=8193,selection='Viewer rows1800-1899, no content or score selection; first8193 UTF8 bytes per document, no normalization or concatenation.',license='Dataset card ODC-BY; underlying web text may retain third-party rights. Corpus not redistributed.',documents=docs)
 if expected is not None:
  assert manifest==expected,'Snapshot changed; do not update reference hashes implicitly'
 else:(P/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Verified',len(docs),'documents;',sum(x['used_bytes'] for x in docs),'capped bytes')
if __name__=='__main__':main()
