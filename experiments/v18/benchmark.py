"""Paired randomized CPU timing; execute only after audit CPU work ends."""
import time,random,statistics
from run import torch,load,models,write
from pruning import local_prefill
from local_attention import allocated_tensor_bytes
@torch.inference_mode()
def probe(m,x,prune):
 n=x.shape[1]-32;t=time.perf_counter();z,state=local_prefill(m,x[:,:n],prune=prune);torch.softmax(z[:,-1],-1);t1=time.perf_counter()
 for j in range(32):z,state=m(x[:,n+j:n+j+1],state,True);torch.softmax(z[:,-1],-1)
 t2=time.perf_counter();return dict(prefill_ms=(t1-t)*1000,end_ms=(t2-t)*1000,decode_ms=(t2-t1)*1000,state_bytes=allocated_tensor_bytes(state))
def main():
 docs=load();ids=[i for i,d in enumerate(docs) if len(d)>=2080][:8];rng=random.Random(18112026);rows=[]
 for seed in [1,2,3]:
  m=models(seed)['local']
  for length in [128,512,2048]:
   for i in ids:
    x=torch.tensor(list(docs[i][:length+32]))[None,:]
    for rep in range(6):
     order=[False,True];rng.shuffle(order)
     for prune in order:
      result=probe(m,x,prune)
      if rep>=2:rows.append(dict(seed=seed,length=length,row_index=i+300,rep=rep,pruned=prune,**result))
 summaries=[]
 for length in [128,512,2048]:
  samples=[x for x in rows if x['length']==length];pairs={}
  for r in samples:pairs.setdefault((r['seed'],r['row_index'],r['rep']),{})[r['pruned']]=r
  ratios=[r[False]['end_ms']/r[True]['end_ms'] for r in pairs.values()];summaries.append(dict(length=length,pairs=len(pairs),median_paired_end_speedup=statistics.median(ratios),faster=sum(r>1 for r in ratios),median_paired_prefill_speedup=statistics.median(r[False]['prefill_ms']/r[True]['prefill_ms'] for r in pairs.values()),full_end_p50=statistics.median(r[False]['end_ms'] for r in pairs.values()),pruned_end_p50=statistics.median(r[True]['end_ms'] for r in pairs.values())))
 write('timings.json',dict(protocol='float32 batch1 one CPU thread; same8 documents each length;3seeds;6 reps incl2warmups;random seed18112026;paired v17 chunk128 baseline vs same with prefix pruning;33 softmax distributions incl32 observed updates;not sampled generation or service',cohort=[i+300 for i in ids],samples=rows,summary=summaries));print(summaries)
if __name__=='__main__':main()
