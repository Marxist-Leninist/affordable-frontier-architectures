from evaluate_fresh import *
def main():
 data,receipt=prepare();models={k:load(k,2,'mixed',True)[0] for k in ['attention','signed']};rng=random.Random(8008);out={}
 with torch.inference_mode():
  for n in [128,512]:
   x=data[:4,-n-1:];raw={task:{k:[] for k in models} for task in ['prefill','decode']}
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
   out[n]={'raw_seconds':raw,'median_seconds':{task:{k:statistics.median(v) for k,v in r.items()} for task,r in raw.items()}}
 (P/'profile.json').write_text(json.dumps(out,indent=2));print('Profile complete')
if __name__=='__main__':main()
