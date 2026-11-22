"""Use a fresh HF network stream and freeze only the bounded selected examples locally."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from datasets import load_dataset
from pilot import L,N,SEED
p=argparse.ArgumentParser()
p.add_argument('prior',type=Path);p.add_argument('out',type=Path)
a=p.parse_args()
assert not a.out.exists()
a.out.mkdir(mode=0o700)
def sha(b):return hashlib.sha256(b).hexdigest()
m=json.loads(Path(__file__).with_name('wikitext.json').read_text())
assert m['revision']=='b08601e04326c79dfdd32d625aee71d232d685c3'
d=json.loads((a.prior/'dev_frozen.json').read_text())
h=json.loads((a.prior/'test.json').read_text())
for off,(split,n,canonical) in enumerate((('train',N,d['data']['train']),('validation',12,d['data']['dev']),('test',24,h['fixture']))):
 s=m['splits'][split];url=s['url'];assert m['revision'] in url
 t=time.perf_counter()
 ds=load_dataset('parquet',data_files={split:[url]},split=split,streaming=True,token=False)
 rng=np.random.default_rng(SEED+off)
 rows=[];xs=[]
 for k,v in enumerate(ds):
  assert k<2048, 'frozen bounded stream row limit'
  text=v['text'];b=text.encode('utf-8')
  if len(b)<L+1 or text.startswith(' ='):continue
  st=int(rng.integers(0,len(b)-L))
  x=np.frombuffer(b[st:st+L+1],dtype='uint8').astype('int64')
  rows.append([k,st,sha(x.tobytes())]);xs.append(x)
  if len(xs)==n:break
 assert len(xs)==n
 arr=np.stack(xs)
 meta=dict(parquet_sha256=s['parquet_sha256'],sha=sha(arr.tobytes()),positions=rows,n=n)
 assert meta==canonical, (split,meta['sha'],canonical['sha'])
 np.save(a.out/f'{split}.npy',arr,allow_pickle=False)
 (a.out/f'{split}.json').write_text(json.dumps(meta,indent=2)+'\n')
 print('LIVE_HF_STREAM',split,'rows_examined',k+1,'selected',n,'fixture_sha',meta['sha'],
       'wall_seconds',round(time.perf_counter()-t,3),flush=True)
(a.out/'source.json').write_text(json.dumps(dict(
    materialization='datasets.load_dataset(parquet, streaming=True) over pinned Hugging Face HTTPS shard URLs, frozen selected windows on disk once and verified against source-parity row fixtures',
    revision=m['revision'],urls={k:v['url'] for k,v in m['splits'].items()},datasets_version=__import__('datasets').__version__),indent=2)+'\n')
