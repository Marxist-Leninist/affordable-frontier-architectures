from mixed_control import *
import platform

def main():
 train,held,_=datasets();models={};opts={};raw={};rng=random.Random(7007)
 for kind in ['attention','signed']:
  initial=torch.load(P/f'initial_{kind}_2.pt',weights_only=True)
  for mode in ['short','long','mixed']:
   key=kind+'_'+mode;m=previous.model(kind).train();m.load_state_dict(initial['model']);o=torch.optim.AdamW(m.parameters(),lr=.001);o.load_state_dict(copy.deepcopy(initial['optimizer']))
   for g in o.param_groups:g['lr']=.001
   models[key]=m;opts[key]=o;raw[key]=[]
 starts=torch.randint(len(train)-513,(36,4),generator=torch.Generator().manual_seed(701))
 for rep in range(36):
  x=torch.stack([train[s:s+513] for s in starts[rep]]);keys=list(models);rng.shuffle(keys)
  for key in keys:
   t=time.perf_counter();update(models[key],opts[key],x,key.split('_')[1],rep);elapsed=time.perf_counter()-t
   if rep>=5:raw[key].append(elapsed)
 out={'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'training_seconds':raw,'training_median_seconds':{k:statistics.median(v) for k,v in raw.items()},'inference':{}}
 models={}
 for kind in ['attention','signed']:
  m=create(kind,2,True);state=torch.load(P/f'{kind}_2_mixed.pt',weights_only=True)['model'];state={k:(-v if kind=='signed' and k.endswith('mix.out.weight') else v) for k,v in state.items()};m.load_state_dict(state);m.eval();models[kind]=m
 with torch.inference_mode():
  for n in [128,512]:
   x=torch.stack([train[i*(n+1):(i+1)*(n+1)] for i in range(4)]);raw={task:{k:[] for k in models} for task in ['prefill','decode']}
   for task in raw:
    for rep in range(36):
     keys=list(models);rng.shuffle(keys)
     for k in keys:
      m=models[k]
      if task=='decode':_,cache=m(x[:,:-1],cache=True)
      t=time.perf_counter()
      if task=='prefill':m(x[:,:-1])
      else:m(x[:,-1:],cache,cache=True)
      elapsed=time.perf_counter()-t
      if rep>=5:raw[task][k].append(elapsed)
   out['inference'][n]={'raw_seconds':raw,'median_seconds':{task:{k:statistics.median(v) for k,v in r.items()} for task,r in raw.items()}}
 (P/'profile.json').write_text(json.dumps(out,indent=2));print('Profile complete',out['training_median_seconds'],flush=True)
if __name__=='__main__':main()
