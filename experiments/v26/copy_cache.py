"""Causal exact-match retrieval counts. Does not inspect unobserved bytes."""
from collections import defaultdict,Counter
import sys
class CopyCache:
    def __init__(self,orders=(4,8,16)):
        self.orders=tuple(orders);self.text=bytearray();self.tables={k:defaultdict(list) for k in orders}
    def feed(self,byte):
        j=len(self.text)
        for k in self.orders:
            if j>=k:self.tables[k][bytes(self.text[j-k:j])].append((j,int(byte)))
        self.text.append(int(byte))
    def query(self,limit=None):
        t=len(self.text);cutoff=0 if limit is None else max(0,t-limit)
        for k in reversed(self.orders):
            if t<k:continue
            entries=self.tables[k].get(bytes(self.text[t-k:t]),())
            legal=[(j,b) for j,b in entries if j-k>=cutoff]
            if legal:
                counts=Counter(b for j,b in legal)
                return counts,len(legal),k,sum(j-k<max(0,t-511) for j,b in legal)
        return Counter(),0,0,0
    def distribution(self,limit=None):
        import numpy as np
        counts,n,k,far=self.query(limit)
        q=np.full(256,1/256,dtype=np.float64)
        for b,v in counts.items():q[b]+=v
        q/=n+1
        return q,n,k,far
    def size_bytes(self):
        seen=set()
        def sz(o):
            if id(o) in seen:return 0
            seen.add(id(o));n=sys.getsizeof(o)
            if isinstance(o,dict):n+=sum(sz(k)+sz(v) for k,v in o.items())
            elif isinstance(o,(tuple,list)):n+=sum(sz(x) for x in o)
            return n
        return sz(self.text)+sz(self.tables)

def target_probs(raw,tail=128):
    import numpy as np
    c=CopyCache();out={p:dict(prob=[],active=[],order=[],far_fraction=[]) for p in ('full','suffix511')}
    for t,b in enumerate(raw[:-1]):
        c.feed(b)
        if t>=len(raw)-1-tail:
            for path,lim in [('full',None),('suffix511',511)]:
                counts,n,k,far=c.query(lim);q=(counts[raw[t+1]]+1/256)/(n+1)
                r=out[path];r['prob'].append(q);r['active'].append(n>0);r['order'].append(k);r['far_fraction'].append(far/n if n else 0)
    return {p:{k:np.array(v) for k,v in d.items()} for p,d in out.items()}
