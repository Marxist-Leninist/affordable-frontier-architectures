from run import *
from models import count,state_bytes
from compact_cache import reachable_bytes
@torch.inference_mode()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(120,125));cpu=time.process_time();verify();docs=[r for r in json.loads((P/'rows.json').read_text())['rows'] if len(r['row']['text'].encode())>=2080][:8];assert len(docs)==8
 jobs=[(s,a,v) for s in (1,2,3) for a in ARMS for v in ('neural','hybrid')];np.random.default_rng(260026).shuffle(jobs);rows=[]
 for seed,arm,variant in jobs:
  m=load(arm,seed)
  for r in docs:
   raw=r['row']['text'].encode()[:2080];x=torch.tensor(list(raw[:2048]))[None,:];updates=torch.tensor(list(raw[2048:]))[None,:]
   def step(z,c):
    p=z.double().softmax(-1).numpy().reshape(256)
    if c is not None:
     q,n,k=c.distribution();w=ALPHAS[arm] if n else 0;p=(1-w)*p+w*q
    assert p.shape==(256,) and np.isfinite(p).all() and abs(p.sum()-1)<1e-10
    return p
   def go():
    c0=time.process_time();t0=time.perf_counter();c=CompactCache() if variant=='hybrid' else None
    if c is not None:
     for b in raw[:2048]:c.feed(b)
    z,state=local_prefill(m,x,1,prune=True) if arm=='local' else prefill(m,x);p=step(z,c);pc=time.process_time()-c0;pw=time.perf_counter()-t0;ps=state_bytes(state);pi=reachable_bytes(c) if c is not None else 0
    c0=time.process_time();t0=time.perf_counter()
    for j in range(32):
     if c is not None:c.feed(int(raw[2048+j]))
     z,state=m(updates[:,j:j+1],state,cache=True);p=step(z,c)
    dc=time.process_time()-c0;dw=time.perf_counter()-t0
    return dict(prefill_cpu_seconds=pc,prefill_wall_seconds=pw,decode_cpu_seconds=dc,decode_wall_seconds=dw,prefill_state_bytes=ps,final_state_bytes=state_bytes(state),prefill_copy_index_bytes=pi,final_copy_index_bytes=reachable_bytes(c) if c is not None else 0)
   go();row=go();row.update(seed=seed,arm=arm,variant=variant,doc=r['row_idx'],parameters=count(m),fp32_weight_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()));rows.append(row)
 write('timing.json',dict(rows=rows,cpu_seconds=time.process_time()-cpu,order=jobs,limits='24samples per variant; one warmup one measured; full256way double softmax and full index build/query/mixing included; original2048 only; batch1 observed updates; shared host; reachable Python bytes not peak RSS'))
 groups={}
 for a in ARMS:
  for v in ('neural','hybrid'):
   rr=[r for r in rows if r['arm']==a and r['variant']==v];g={k:float(np.median([r[k] for r in rr])) for k in ('prefill_state_bytes','final_state_bytes','prefill_copy_index_bytes','final_copy_index_bytes','parameters','fp32_weight_bytes')};g.update(prefill_bytes_per_cpu_second=2048*len(rr)/sum(r['prefill_cpu_seconds'] for r in rr),observed_bytes_per_decode_cpu_second=32*len(rr)/sum(r['decode_cpu_seconds'] for r in rr),total_wall_ms_p50_p95=np.quantile([1000*(r['prefill_wall_seconds']+r['decode_wall_seconds']) for r in rr],[.5,.95]).tolist(),samples=len(rr));groups[a+'_'+v]=g
 write('timing_summary.json',groups);print('TIMING CPU',time.process_time()-cpu)
if __name__=='__main__':main()
