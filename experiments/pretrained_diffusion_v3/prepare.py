"""Acquire pinned safe model files and public text; never execute hub code."""
from pathlib import Path
import hashlib,json,urllib.request
P=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=json.loads((P/'model_manifest.json').read_text());(P/'original').mkdir(exist_ok=True)
    for f,v in m['files'].items():
        p=P/'original'/f
        if not p.exists():p.write_bytes(urllib.request.urlopen(f'https://huggingface.co/{m["repo"]}/resolve/{m["revision"]}/{f}',timeout=120).read())
        assert sha(p)==v['sha256'],f
    d=json.loads((P/'source_data_manifest.json').read_text());(P/'data').mkdir(exist_ok=True)
    from tokenizers import Tokenizer
    import numpy as np
    tok=Tokenizer.from_file(str(P/'original'/'tokenizer.json'))
    counts={}
    for split,v in d['splits'].items():
        p=P/'data'/f'{split}.txt'
        if not p.exists():
            import pyarrow.parquet as pq
            raw=urllib.request.urlopen(v['url'],timeout=120).read();assert hashlib.sha256(raw).hexdigest()==v['parquet_sha256']
            pf=P/'data'/f'{split}.parquet';pf.write_bytes(raw)
            p.write_text('\n'.join(pq.read_table(pf)['text'].to_pylist()),encoding='utf-8')
        assert sha(p)==v['text_sha256'],split
        # Tokenize each original line separately, preserving its newline. Bounded
        # memory, no special tokens and no joins across official split boundaries.
        # This is intentionally not claimed identical to whole-file BPE tokenization.
        ids=np.concatenate([np.array(tok.encode(line,add_special_tokens=False).ids,dtype=np.int64)
            for line in p.read_text().splitlines(keepends=True)])
        np.save(P/'data'/f'{split}.npy',ids)
        counts[split]={'tokens':len(ids),'sha256':sha(P/'data'/f'{split}.npy')}
    (P/'token_manifest.json').write_text(json.dumps(counts,indent=2)+'\n');print(counts)
if __name__=='__main__':main()
