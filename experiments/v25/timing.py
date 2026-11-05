from run import *
from models import count,state_bytes
@torch.inference_mode()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(100,105));cpu=time.process_time();verify();docs=[r for r in json.loads((P/'rows.json').read_text())['rows'] if len(r['row']['text'].encode())>=2080][:8];assert len(docs)==8
 jobs=[(s,a,c) for s in (1,2,3) for a in ARMS for c in (512,2048)];np.random.default_rng(250025).shuffle(jobs);rows=[]
 for seed,arm,ctx in jobs:
  m=load(arm,seed)
  for r in docs:
   raw=torch.tensor(list(r['row']['text'].encode()[:2080]))[None,:];x=raw[:,2048-ctx:2048];updates=raw[:,2048:2080]
   def pf():return local_prefill(m,x,1,prune=True) if arm=='local' else prefill(m,x)
   z,state=pf();prob=z.softmax(-1)
   for j in range(32):z,state=m(updates[:,j:j+1],state,cache=True);prob=z.softmax(-1)
   c0=time.process_time();t0=time.perf_counter();z,state=pf();prob=z.softmax(-1);pc=time.process_time()-c0;pw=time.perf_counter()-t0;ps=state_bytes(state)
   c0=time.process_time();t0=time.perf_counter()
   for j in range(32):z,state=m(updates[:,j:j+1],state,cache=True);prob=z.softmax(-1);assert prob.shape==(1,1,256) and torch.isfinite(prob).all()
   dc=time.process_time()-c0;dw=time.perf_counter()-t0
   rows.append(dict(seed=seed,arm=arm,context=ctx,doc=r['row_idx'],prefill_cpu_seconds=pc,prefill_wall_seconds=pw,decode_cpu_seconds=dc,decode_wall_seconds=dw,parameters=count(m),fp32_weight_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()),prefill_state_bytes=ps,final_state_bytes=state_bytes(state)))
 write('timing.json',dict(rows=rows,cpu_seconds=time.process_time()-cpu,order=jobs,limits='8 convenience documents,3seeds,one repeat after complete warmup;batch1,shared host;observed-byte updates not generation/commercial tokens;all33 full softmax distributions included'))
 groups={}
 for a in ARMS:
  for ctx in (512,2048):
   rr=[r for r in rows if r['arm']==a and r['context']==ctx];groups[a+'_'+str(ctx)]={k:float(np.median([r[k] for r in rr])) for k in ('prefill_state_bytes','final_state_bytes','parameters','fp32_weight_bytes')};g=groups[a+'_'+str(ctx)]
   g.update(prefill_bytes_per_cpu_second=ctx*len(rr)/sum(r['prefill_cpu_seconds'] for r in rr),observed_bytes_per_decode_cpu_second=32*len(rr)/sum(r['decode_cpu_seconds'] for r in rr),total_wall_ms_p50_p95=np.quantile([1000*(r['prefill_wall_seconds']+r['decode_wall_seconds']) for r in rr],[.5,.95]).tolist(),samples=len(rr))
 write('timing_summary.json',groups);print('TIMING CPU',time.process_time()-cpu)
if __name__=='__main__':main()
