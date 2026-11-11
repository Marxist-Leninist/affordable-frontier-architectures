"""Pinned read-only HF corpora. No downloaded corpus is redistributed."""
import bisect,hashlib,json,os,random,re
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent

def sha(b):return hashlib.sha256(b).hexdigest()
def articles(b):
    starts=[m.start() for m in re.finditer(rb'^ = [^=\n].*?[^=\n] = \r?$',b,re.M)]
    if not starts or starts[0]!=0:starts=[0]+starts
    starts.append(len(b));return [b[a:z] for a,z in zip(starts,starts[1:]) if z>a]

def load():
    docs={};wm=json.loads((P/'wikitext_manifest.json').read_text())
    for split in ('train','validation','test'):
        b=(Path(os.environ.get('REV_WIKITEXT_DIR',str(P.parent/'frontier_research_v8')))/f'{split}.txt').read_bytes()
        assert sha(b)==wm['splits'][split]['text_sha256'];docs[split]=articles(b)
    fm=json.loads((P/'fineweb_manifest.json').read_text());b=Path(os.environ.get('REV_FINEWEB_ROWS',str(P.parent/'frontier_research_v31'/'rows.json'))).read_bytes()
    assert sha(b)==fm['response_sha256'];records=json.loads(b)['rows'];assert len(records)==fm['row_count']==len(fm['documents']);docs['fineweb']=[]
    for r,m in zip(records,fm['documents']):
        text=r['row']['text'].encode();assert sha(text)==m['full_text_sha256'];docs['fineweb'].append(text[:4097])
    return docs

class Corpus:
    def __init__(self,docs,t=32):
        self.t=t;self.docs=[np.frombuffer(b,dtype=np.uint8).copy() for b in docs if len(b)>t]
        self.ends=np.cumsum([len(b)-t for b in self.docs]);assert len(self.ends)
    def batch(self,g,batch=8):
        picks=torch.randint(int(self.ends[-1]),(batch,),generator=g).tolist();rows=[]
        for p in picks:
            d=bisect.bisect_right(self.ends,p);st=p-(int(self.ends[d-1]) if d else 0);rows.append(self.docs[d][st:st+self.t+1])
        a=torch.from_numpy(np.stack(rows)).long();return a[:,:-1],a[:,1:]

def selections(docs,t=32):
    out={};meta={};train=b''.join(docs['train'])
    for split,n,seed in [('validation',128,7201),('test',256,7202),('fineweb',256,7203)]:
        candidates=[(i,j) for i,b in enumerate(docs[split]) for j in range(0,len(b)-t,t+1)]
        random.Random(seed).shuffle(candidates); chosen=candidates[:n];rows=[docs[split][i][j:j+t+1] for i,j in chosen]
        assert len(rows)==n;out[split]=chosen
        meta[split]={'windows':n,'window_length':t+1,'scored_bytes':n*(t//2),'documents':len(set(i for i,j in chosen)),'identical_full_windows_in_train':sum(r in train for r in rows),'duplicate_selected_windows':len(rows)-len(set(rows)),'selection_sha256':sha(json.dumps(chosen).encode()),'selected_bytes_sha256':sha(b''.join(rows))}
    return out,meta

def eval_arrays(docs,selections,split,t=32):
    a=torch.tensor([list(docs[split][i][j:j+t+1]) for i,j in selections[split]],dtype=torch.long)
    return a[:,:-1],a[:,1:]
