"""Exact cached inference at fixed segment boundaries; no sliding-window substitution."""
import baseline as base
import numpy as np
import json, time, hashlib, platform
from pathlib import Path

class Incremental:
    def __init__(self, model, capacity):
        self.model=model; self.capacity=capacity; self.n=0
        d=model.d
        self.state=np.zeros(d)
        self.keys=np.empty((capacity,d)) if model.kind=='attention' else None
        self.values=np.empty((capacity,d)) if model.kind=='attention' else None
    def reset(self):
        self.n=0; self.state.fill(0)
    def step(self, token):
        if self.n >= self.capacity: raise ValueError('Explicit reset required at segment boundary')
        m=self.model; p=m.p; d=m.d; z=p['E'][token]
        if m.kind=='attention':
            z=z+.1*np.sin(self.n*10000**(-np.arange(d)/d))
            self.keys[self.n]=z@p['K']; self.values[self.n]=z@p['W']
            q=z@p['Q']; a=base.softmax((q@self.keys[:self.n+1].T)/np.sqrt(d))
            h=z+(a@self.values[:self.n+1])@p['O']
            out=h+np.tanh(h@p['F'])@p['G']
        else:
            self.state=np.tanh(z+self.state@p['R']); out=self.state
        self.n+=1
        return base.softmax(out@p['U'])
    def bytes(self):
        return self.state.nbytes + (0 if self.keys is None else self.keys.nbytes+self.values.nbytes)

def evaluate(m):
    va=base.encode(base.VALID); segments=va[:8192].reshape(-1,32); targets=va[1:8193].reshape(-1,32)
    reference=m.loss(segments,None,False); cached=[]
    decoder=Incremental(m,32)
    for segment in segments:
        decoder.reset(); cached.append([decoder.step(t) for t in segment])
    cached=np.array(cached)
    delta=float(np.max(np.abs(cached-reference)))
    np.testing.assert_allclose(cached,reference,rtol=1e-12,atol=1e-12)
    def bpc(prob): return float(-np.log2(np.take_along_axis(prob,targets[:,:,None],axis=-1)).mean())
    return {'max_probability_error':delta,'reference_bpc':bpc(reference),'cached_bpc':bpc(cached),'targets':8192}

def timing(m, length):
    # Fixed held-out token replay: both implementations see exactly the same prefixes.
    tokens=base.encode(base.VALID)[:length]; decoder=Incremental(m,length)
    samples={'cached':[],'recompute':[]}; totals={'cached':[],'recompute':[]}
    for trial in range(6):
        for mode in (('cached','recompute') if trial%2==0 else ('recompute','cached')):
            decoder.reset(); row=[]
            for i,t in enumerate(tokens):
                tick=time.perf_counter_ns()
                if mode=='cached': prob=decoder.step(t)
                else: prob=m.loss(tokens[None,:i+1],None,False)[0,-1]
                row.append(time.perf_counter_ns()-tick)
            if trial:
                samples[mode].extend(row); totals[mode].append(sum(row)/1e6)
    return {'length':length,'resident_state_bytes':decoder.bytes(),
        'timings':{mode:{'p50_ms':float(np.median(v))/1e6,'p95_ms':float(np.quantile(v,.95))/1e6,'sequence_median_ms':float(np.median(totals[mode]))} for mode,v in samples.items()},
        'sequence_speedup':float(np.median(totals['recompute'])/np.median(totals['cached']))}

def main():
    rows=[]
    for seed in (1,2,3):
        for kind,d in [('attention',16),('recurrent',34)]:
            m=base.train(kind,d,seed)
            row={'kind':kind,'seed':seed,'parameters':m.count(),'weights_sha256':hashlib.sha256(b''.join(p.tobytes() for p in m.p.values())).hexdigest(),'quality':evaluate(m),'timings':[timing(m,n) for n in (32,128,512)]}
            rows.append(row)
    result={'classification':'Actual trained toy neural LM exact incremental inference; synthetic text; not frontier evidence','numpy':np.__version__,'platform':platform.platform(),'training_tokens_per_model':102400,'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('benchmark.py','baseline.py')},'rows':rows}
    Path('results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
