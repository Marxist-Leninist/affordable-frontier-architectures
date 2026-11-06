from train import *
import platform

def main():
 train,held,receipt=datasets();out={'dataset':receipt,'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'results':{}}
 rng=random.Random(603)
 for context in [128,512]:
  models={};opts={}
  for kind in ['attention','signed']:
   m=create(kind,2,True);state=torch.load(P/f'restart_{kind}_2_retain.pt',weights_only=True)['model']
   if kind=='signed':
    state={k:(-v if k.endswith('mix.out.weight') else v) for k,v in state.items()}
   m.load_state_dict(state);models[kind]=m;opts[kind]=torch.optim.AdamW(m.parameters(),lr=.0001)
  x=torch.stack([train[i*(context+1):(i+1)*(context+1)] for i in range(4)]);raw={}
  for task in ['prefill','decode','training']:
   raw[task]={k:[] for k in models}
   def bench(kind):
    m=models[kind]
    if task=='training':
     m.train();o=opts[kind];t=time.perf_counter();o.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x[:,:-1]).flatten(0,1),x[:,1:].flatten());loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step();return time.perf_counter()-t
    m.eval()
    with torch.inference_mode():
     if task=='prefill':
      t=time.perf_counter();m(x[:,:-1]);return time.perf_counter()-t
     _,state=m(x[:,:-1],cache=True);t=time.perf_counter();m(x[:,-1:],state,cache=True);return time.perf_counter()-t
   for k in models:
    for _ in range(5):bench(k)
   for rep in range(41):
    keys=list(models);rng.shuffle(keys)
    for k in keys:raw[task][k].append(bench(k))
  out['results'][context]={'raw_seconds':raw,'median_seconds':{task:{k:statistics.median(v) for k,v in r.items()} for task,r in raw.items()}}
 (P/'profile_v6.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
