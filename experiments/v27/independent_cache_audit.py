"""Independent substring-search oracle and eviction-storage audit (not production source)."""
from pathlib import Path
import json, hashlib
import numpy as np
from bounded_cache import BoundedCache, reachable_bytes
from compact_cache import CompactCache
P=Path(__file__).parent

def oracle(history,limit=None):
    suffix=history if limit is None else history[-limit:]
    q=np.full(256,1/256,dtype=np.float64)
    for order in (16,8,4):
        if len(history)<order: continue
        key=history[-order:];counts=np.zeros(256,dtype=np.int64)
        # Search only starts with an already observed follower.
        pos=suffix.find(key)
        while pos>=0 and pos+order<len(suffix):
            counts[suffix[pos+order]]+=1
            pos=suffix.find(key,pos+1)
        n=int(counts.sum())
        if n:return (q+counts)/(n+1),n,order
    return q,0,0

def run():
    rng=np.random.default_rng(270031)
    streams=[b'a'*2200,b'abcdXabcdY'*240,bytes(range(256))*10,bytes(rng.integers(0,5,2400).tolist()),bytes(rng.integers(0,256,2400).tolist())]
    comparisons=0;largest_error=0.;storage=[]
    for raw in streams:
        caches={s:BoundedCache(s) for s in (17,32,512,1024,2048)};caches[None]=CompactCache()
        for t,byte in enumerate(raw,1):
            for s,c in caches.items():
                c.feed(byte)
                if t<40 or t%17==0 or t in (511,512,513,1023,1024,1025,2047,2048,2049,len(raw)):
                    got,n,k=c.distribution();want,nn,kk=oracle(raw[:t],s)
                    np.testing.assert_array_equal(got,want);assert (n,k)==(nn,kk)
                    largest_error=max(largest_error,abs(float(got.sum())-1));comparisons+=1
                if s:
                    for k,events in c.events.items():
                        assert len(events)==max(0,min(t,s)-k)
                        assert all(start>=max(0,t-s) for start,key,b in events)
                        assert sum(sum(v.values()) for v in c.tables[k].values())==len(events)
                        assert all(n>0 for counts in c.tables[k].values() for n in counts.values())
    # Long high-cardinality stream checks retained allocations, not merely filtered query scope.
    for s in (512,1024,2048):
        c=BoundedCache(s);snap=[]
        for t,b in enumerate(rng.integers(0,256,32768).tolist(),1):
            c.feed(b)
            if t in (4096,8192,16384,32768):snap.append(dict(seen=t,reachable_bytes=reachable_bytes(c),contexts=sum(map(len,c.tables.values())),events=sum(map(len,c.events.values()))))
        assert max(x['reachable_bytes'] for x in snap)<1.3*min(x['reachable_bytes'] for x in snap)
        storage.append(dict(scope=s,snapshots=snap))
    out=dict(full_distribution_oracle_comparisons=comparisons,maximum_normalization_error=largest_error,storage_growth=storage,source_sha256=hashlib.sha256((P/'bounded_cache.py').read_bytes()).hexdigest(),status='pass')
    (P/'independent_cache_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
