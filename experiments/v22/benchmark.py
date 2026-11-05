from run import *
@torch.inference_mode()
def probe(m,kind,raw,hybrid,w):
 x=torch.tensor([list(raw)]);begin=1536 if kind=='transformer_reset512' else 0;cache=None;t=time.perf_counter()
 z,state=start(m,kind,x[:,begin:2048]);p=z[0,-1].double().softmax(-1).numpy()
 if hybrid:
  cache=CompactCache()
  for b in raw[:2048]:cache.feed(b)
  q,n,_=cache.distribution();p=(1-w)*p+w*q if n else p
 err=abs(p.sum()-1);pre=time.perf_counter()-t;t=time.perf_counter()
 for i in range(2048,2080):
  z,state=m(x[:,i:i+1],state,cache=True);p=z[0,-1].double().softmax(-1).numpy()
  if hybrid:
   cache.feed(raw[i]);q,n,_=cache.distribution();p=(1-w)*p+w*q if n else p
  err=max(err,abs(p.sum()-1))
 update=time.perf_counter()-t
 return dict(prefill_ms=pre*1000,updates_ms=update*1000,total_ms=(pre+update)*1000,probability_sum_error=float(err),state_tensor_bytes=state_bytes(state),copy_reachable_bytes=reachable_bytes(cache) if cache else 0)
def main():
 protocol=json.loads((P/'protocol.json').read_text());assert protocol['source_sha256']['benchmark.py']==sha(P/'benchmark.py');cpu=time.process_time();_,docs=data();rng=random.Random(2026110522);rows=[];frozen=json.loads((P/'frozen_selection.json').read_text())
 for kind in KINDS:
  for seed in SEEDS:
   ms={q:load(kind,seed,q) for q in (False,True)}
   for doc,raw in docs:
    for rep in range(4):
     variants=[(q,h) for q in (False,True) for h in (False,True)];rng.shuffle(variants)
     for quant,hybrid in variants:
      r=probe(ms[quant],kind,raw,hybrid,frozen['transformer_full2048' if kind=='transformer' else kind]['weight'])
      if rep>=1:rows.append(dict(kind=kind,seed=seed,doc=doc,rep=rep,precision='int8' if quant else 'fp32',hybrid=hybrid,**r))
 summary=[]
 for kind in KINDS:
  for precision in ('fp32','int8'):
   for hybrid in (False,True):
    rr=[r for r in rows if (r['kind'],r['precision'],r['hybrid'])==(kind,precision,hybrid)]
    summary.append(dict(kind=kind,precision=precision,hybrid=hybrid,samples=len(rr),**{k:dict(p50=float(np.median([r[k] for r in rr])),p95=float(np.quantile([r[k] for r in rr],.95))) for k in ('total_ms','prefill_ms','updates_ms','state_tensor_bytes','copy_reachable_bytes')}))
 write('timings.json',dict(rows=rows,summary=summary,cpu_seconds=time.process_time()-cpu));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
