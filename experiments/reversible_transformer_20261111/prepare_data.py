"""Optional portable download. Does not run during the queued experiment.

Downloads licensed WikiText source data only; never publishes/redistributes it.
Requires pyarrow from its official/reputable package source.
"""
import argparse,hashlib,json,urllib.request
from pathlib import Path

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    import pyarrow.parquet as pq
    p=argparse.ArgumentParser();p.add_argument('--destination',required=True);p.add_argument('--fineweb',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;dest=Path(a.destination);dest.mkdir(parents=True,exist_ok=True)
    wm=json.loads((root/'wikitext_manifest.json').read_text())
    for split,info in wm['splits'].items():
        parquet=dest/f'{split}.parquet'
        if not parquet.exists():
            with urllib.request.urlopen(info['url'],timeout=120) as r:b=r.read()
            assert sha(b)==info['parquet_sha256'];parquet.write_bytes(b)
        assert sha(parquet.read_bytes())==info['parquet_sha256']
        texts=pq.read_table(parquet).column('text').to_pylist();b='\n'.join(texts).encode()
        assert len(texts)==info['rows'] and sha(b)==info['text_sha256'];(dest/f'{split}.txt').write_bytes(b)
    if a.fineweb:
        fm=json.loads((root/'fineweb_manifest.json').read_text())
        with urllib.request.urlopen(fm['viewer_url'],timeout=120) as r:b=r.read()
        assert sha(b)==fm['response_sha256'],'FineWeb viewer bytes differ from frozen source; stop, do not substitute another set.'
        (dest/'fineweb_rows.json').write_bytes(b)
    print('Verified data prepared. Set REV_WIKITEXT_DIR and REV_FINEWEB_ROWS to these paths before running.')
if __name__=='__main__':main()
