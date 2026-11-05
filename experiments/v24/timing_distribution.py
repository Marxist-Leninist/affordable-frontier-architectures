"""Descriptive matched observed-byte timing; not sampled production generation."""
from run import *
from models import count,state_bytes
@torch.inference_mode()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(80,85));cpu=time.process_time();docs=[r for r in json.loads((P/'rows.json').read_text())['rows'] if len(r['row']['text'].encode())>=2080][:8];assert len(docs)==8
 jobs=[(s,a,c) for s in (1,2,3) for a in ('context2048','reset512','mixed') for c in (512,2048)];np.random.default_rng(240024).shuffle(jobs);rows=[]
 for seed,arm,ctx in jobs:
  f=(P if arm=='mixed' else C)/'checkpoints'/f'{arm}_seed{seed}_step250.pt';m=load('transformer',seed);m.load_state_dict(torch.load(f,weights_only=True));m.eval()
  for idx,r in enumerate(docs):
   raw=torch.tensor(list(r['row']['text'].encode()[:2080]))[None,:];x=raw[:,2048-ctx:2048];updates=raw[:,2048:2080]
   # One full identical warmup per measured document, including all decode steps.
   z,state=prefill(m,x);prob=z.softmax(-1)
   for j in range(32):z,state=m(updates[:,j:j+1],state,cache=True);prob=z.softmax(-1)
   c0=time.process_time();t0=time.perf_counter();z,state=prefill(m,x);prob=z.softmax(-1);pc=time.process_time()-c0;pw=time.perf_counter()-t0;ps=state_bytes(state)
   c0=time.process_time();t0=time.perf_counter()
   for j in range(32):z,state=m(updates[:,j:j+1],state,cache=True);prob=z.softmax(-1);assert prob.shape==(1,1,256)
   dc=time.process_time()-c0;dw=time.perf_counter()-t0
   rows.append(dict(seed=seed,arm=arm,context=ctx,doc=r['row_idx'],prefill_cpu_seconds=pc,prefill_wall_seconds=pw,decode_cpu_seconds=dc,decode_wall_seconds=dw,prefill_bytes=ctx,decode_observed_bytes=32,parameters=count(m),fp32_weight_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()),prefill_state_bytes=ps,final_state_bytes=state_bytes(state)))
 out=dict(rows=rows,cpu_seconds=time.process_time()-cpu,order=jobs,units='UTF8 byte inputs / observed next-byte full-distribution updates; not commercial billable tokens or sampled generation',limits='8 deterministic documents, one repeat after per-document warmup, shared CPU host, batch1, no concurrency, no statistical speed attribution to weights')
 write('timing.json',out);print('Timing done',out['cpu_seconds'])
if __name__=='__main__':main()
