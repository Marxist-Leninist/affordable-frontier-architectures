"""Pinned official Hugging Face WikiText-2 raw splits. Preserve all UTF-8 bytes."""
from pathlib import Path
import json,hashlib,urllib.request
import pyarrow.parquet as pq
REV='b08601e04326c79dfdd32d625aee71d232d685c3'
EXPECTED={'train':'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7','validation':'204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c','test':'5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'}
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    rows={}
    for split in ('train','validation','test'):
        name=f'{split}-00000-of-00001.parquet'
        url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REV}/wikitext-2-raw-v1/{name}'
        path=Path(name)
        if not path.exists():
            with urllib.request.urlopen(url,timeout=120) as response: path.write_bytes(response.read())
        if sha(path.read_bytes())!=EXPECTED[split]: raise ValueError('Parquet SHA256 mismatch: '+split)
        table=pq.read_table(path); texts=table.column('text').to_pylist()
        # Explicit join retains original rows (including empty ones), with one separator LF.
        data='\n'.join(texts).encode('utf-8'); Path(split+'.txt').write_bytes(data)
        rows[split]={'url':url,'parquet_sha256':sha(path.read_bytes()),'rows':len(texts),'utf8_bytes':len(data),'text_sha256':sha(data)}
    Path('data_manifest.json').write_text(json.dumps({'dataset':'Salesforce/wikitext','config':'wikitext-2-raw-v1','revision':REV,'serialization':'UTF-8 bytes of LF-joined original text rows; no normalization or filtering','splits':rows},indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
