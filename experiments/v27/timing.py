from run import *
@torch.inference_mode()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(200,205));cpu=time.process_time();verify();docs=[r for r in json.loads((P/'rows.json').read_text())['rows'] if len(r['row']['text'].encode())>=4128][:8];assert len(docs)==8
 jobs=[(s,a,v) for s in (1,2,3) for a in ARMS for v in ('neural',*SCOPES)];np.random.default_rng(270027).shuffle(jobs);rows=[]
 for seed,a,v in jobs:
  m=load(a,seed)
  for r in docs:
   raw=r['row']['text'].encode()[:4128];x=torch.tensor(list(raw[:4096]))[None,:];updates=torch.tensor(list(raw[4096:]))[None,:]
   def step(z,c):
    p=z.double().softmax(-1).numpy().reshape(256)
    if c is not None:
     q,n,k=c.distribution();w=ALPHAS[a] if n else 0;p=(1-w)*p+w*q
    assert abs(p.sum()-1)<1e-10;return p
   def go():
    t=time.perf_counter();c0=time.process_time();c=None if v=='neural' else make_cache(v)
    if c is not None:
     for b in raw[0 if v=='full' else 4096-v:4096]:c.feed(b)
    z,state=local_prefill(m,x,1,prune=True) if a=='local' else prefill(m,x);step(z,c);pw=time.perf_counter()-t;pc=time.process_time()-c0;pi=reachable_bytes(c) if c else 0;ps=state_bytes(state)
    t=time.perf_counter();c0=time.process_time()
    for j in range(32):
     if c is not None:c.feed(int(raw[4096+j]))
     z,state=m(updates[:,j:j+1],state,cache=True);step(z,c)
    return dict(prefill_wall_seconds=pw,prefill_cpu_seconds=pc,decode_wall_seconds=time.perf_counter()-t,decode_cpu_seconds=time.process_time()-c0,prefill_copy_index_bytes=pi,final_copy_index_bytes=reachable_bytes(c) if c else 0,prefill_state_bytes=ps,final_state_bytes=state_bytes(state))
   go()
   for rep in range(2):
    row=go();row.update(seed=seed,arm=a,scope=v,doc=r['row_idx'],rep=rep,parameters=count(m),weight_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()));rows.append(row)
 write('timing.json',dict(rows=rows,cpu_seconds=time.process_time()-cpu,jobs=jobs));print('TIMING CPU',time.process_time()-cpu,flush=True)
if __name__=='__main__':main()
