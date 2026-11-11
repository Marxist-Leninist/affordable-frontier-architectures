"""Pinned corpora and full-file article-bounded held-out NAT evaluation."""
import hashlib,json,re,os
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'frontier_research_v8'
LENGTH,PREFIX,OUTPUT=128,96,32
PAD=257
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(split):
    b=(SOURCE/(split+'.txt')).read_bytes()
    m=json.loads((SOURCE/'data_manifest.json').read_text())
    assert sha(b)==m['splits'][split]['text_sha256']
    return b

def articles(b):
    # Single-equals headings are article titles. All original bytes are retained.
    starts=[m.start() for m in re.finditer(rb'^ = [^=\n].*?[^=\n] = \r?$',b,re.M)]
    if not starts or starts[0]!=0:starts=[0]+starts
    starts.append(len(b))
    return [b[a:z] for a,z in zip(starts,starts[1:]) if z>a]

class TrainingCorpus:
    def __init__(self):
        self.docs=[np.frombuffer(b,dtype=np.uint8) for b in articles(raw('train')) if len(b)>=LENGTH]
        self.counts=np.array([len(b)-LENGTH+1 for b in self.docs],dtype=np.int64)
        self.ends=np.cumsum(self.counts)
    def batch(self,rng,batch):
        picks=rng.integers(0,self.ends[-1],batch)
        ids=np.searchsorted(self.ends,picks,side='right')
        begin=np.where(ids==0,0,self.ends[np.maximum(0,ids-1)])
        rows=np.stack([self.docs[d][p-s:p-s+LENGTH] for d,p,s in zip(ids,picks,begin)]).astype(np.int64)
        return torch.from_numpy(rows[:,:PREFIX].copy()),torch.from_numpy(rows[:,PREFIX:].copy())

def eval_windows(docs,train_blob,limit=None):
    rows=[];groups=[];starts=[];seen=set();rejected=0;target_seen=0
    for did,doc in enumerate(docs):
        for st in range(0,len(doc)-LENGTH+1,LENGTH):
            w=doc[st:st+LENGTH]
            if w in seen or w in train_blob:rejected+=1;continue
            seen.add(w);rows.append(list(w));groups.append(did);starts.append(st)
            target_seen+=int(w[PREFIX:] in train_blob)
    a=np.asarray(rows,dtype=np.int64);g=np.asarray(groups,dtype=np.int64)
    if limit and len(a)>limit:
        chosen=np.linspace(0,len(a)-1,limit,dtype=np.int64)
        a,g=a[chosen],g[chosen]
    return torch.from_numpy(a[:,:PREFIX].copy()),torch.from_numpy(a[:,PREFIX:].copy()),g,{'eligible_windows_before_limit':len(rows),'evaluated_windows':len(a),'documents':len(set(g.tolist())),'excluded_exact128_train_or_duplicate':rejected,'target32_matches_training_all_windows':target_seen,'selection_sha256':sha(a.tobytes()),'selection':'all nonoverlapping article-contained windows, or evenly spaced over all windows when limit applies'}

def text_sets(split,limit=None):
    # Cache licensed local evaluation bytes only; never redistribute this cache.
    # Key includes both source identities and this preprocessing implementation.
    source,train=raw(split),raw('train')
    key=sha(source+train+Path(__file__).read_bytes())[:20]
    directory=ROOT/'data_cache';directory.mkdir(exist_ok=True)
    cache=directory/f'{split}_{key}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:
            p,y,g=z['p'],z['y'],z['groups'];meta=json.loads(str(z['meta']))
        assert sha(np.concatenate((p,y),1).tobytes())==meta['selection_sha256'],'cache integrity mismatch'
    else:
        pt,yt,g,meta=eval_windows(articles(source),train)
        p,y=pt.numpy(),yt.numpy()
        temporary=cache.with_suffix(f'.{os.getpid()}.tmp')
        with temporary.open('wb') as f:np.savez_compressed(f,p=p,y=y,groups=g,meta=json.dumps(meta))
        temporary.replace(cache)
    if limit and len(p)>limit:
        idx=np.linspace(0,len(p)-1,limit,dtype=np.int64);p,y,g=p[idx],y[idx],g[idx]
        meta={**meta,'evaluated_windows':len(p),'documents':len(set(g.tolist())),'selection_sha256':sha(np.concatenate((p,y),1).tobytes())}
    return torch.from_numpy(p.copy()),torch.from_numpy(y.copy()),g.copy(),meta

def fineweb():
    base=ROOT.parent/'frontier_research_v31'
    m=json.loads((base/'data_manifest.json').read_text());b=(base/'rows.json').read_bytes()
    assert sha(b)==m['response_sha256']
    parsed=json.loads(b);docs=[]
    for record,expected in zip(parsed['rows'],m['documents']):
        text=record['row']['text'].encode('utf-8')
        assert sha(text)==expected['full_text_sha256']
        docs.append(text[:4097])
    p,y,g,meta=eval_windows(docs,raw('train'))
    meta.update({'source_dataset':m['dataset'],'revision':m['observed_x_revision'],'source_rows':'1500-1599','source_response_sha256':m['response_sha256'],'license':m['license'],'new_to_this_nat_comparison':True})
    return p,y,g,meta

def toy(rng,n,task=None):
    digits=rng.integers(0,10,(n,OUTPUT),dtype=np.int64)
    tasks=rng.integers(0,4,n) if task is None else np.full(n,task)
    p=np.full((n,PREFIX),PAD,dtype=np.int64);y=np.empty((n,OUTPUT),dtype=np.int64)
    for i,t in enumerate(tasks):
        name=[b'copy:',b'reverse:',b'sort:',b'prefixsum_mod10:'][t]
        p[i,:len(name)]=np.frombuffer(name,dtype=np.uint8)
        p[i,len(name):len(name)+OUTPUT]=digits[i]+48
        target=digits[i] if t==0 else digits[i,::-1] if t==1 else np.sort(digits[i]) if t==2 else np.cumsum(digits[i])%10
        y[i]=target+48
    return torch.from_numpy(p),torch.from_numpy(y),tasks

if __name__=='__main__':
    out={}
    for name in ['validation','test']:
        p,y,g,m=text_sets(name);out[name]=m;print(name,m,flush=True)
    p,y,g,m=fineweb();out['fineweb']=m;print('fineweb',m,flush=True)
    (ROOT/'data_manifest_long.json').write_text(json.dumps(out,indent=2)+'\n')
