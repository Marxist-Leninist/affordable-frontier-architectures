from prefix_test import *
def tail_logits(m,x):
 h=m.emb(x)
 for b in m.blocks:h,_=b(h)
 return m.head(m.norm(h[:,-128:]))
def main():
 data,_=prepare();models={k+'_'+a:load(k,2,a,True)[0] for k in ['attention','signed'] for a in ['long','mixed']};raw={};rng=random.Random(9009)
 with torch.inference_mode():
  for task in ['tail128_prefill','next_token_decode']:
   for key in models:
    for condition in data:raw[task+':'+key+':'+condition]=[]
   for rep in range(36):
    jobs=[(k,c) for k in models for c in data];rng.shuffle(jobs)
    for key,condition in jobs:
     x=data[condition][:4];m=models[key]
     if task=='next_token_decode':_,cache=m(x[:,:-1],cache=True)
     t=time.perf_counter()
     if task=='tail128_prefill':tail_logits(m,x[:,:-1])
     else:m(x[:,-1:],cache,cache=True)
     elapsed=time.perf_counter()-t
     if rep>=5:raw[task+':'+key+':'+condition].append(elapsed)
 (P/'profile.json').write_text(json.dumps({'raw_seconds':raw,'median_seconds':{k:statistics.median(v) for k,v in raw.items()},'threads':torch.get_num_threads(),'torch':torch.__version__,'batch':4,'warmups':5,'repetitions':31},indent=2));print('Profile complete')
if __name__=='__main__':main()
