"""Separate post-audit timing; identical document cohorts for all models and lengths."""
import time,json,statistics
import numpy as np
from run import P,V,load,LM,Ngram,localize,torch,write,tensor_bytes,allocated_tensor_bytes
@torch.inference_mode()
def probe(model,docs,length):
 pre=[];dec=[];end=[];logical=[];storage=[]
 for raw in docs:
  x=torch.tensor(list(raw[:length+32]),dtype=torch.long)[None,:]
  for rep in range(5):
   t=time.perf_counter()
   if isinstance(model,Ngram):c=raw[:length][-4:];np.array([model.prob(c,y) for y in range(256)])
   else:z,s=model(x[:,:length],cache=True);torch.softmax(z[:,-1],-1)
   t1=time.perf_counter()
   for j in range(32):
    if isinstance(model,Ngram):c=(c+raw[length+j:length+j+1])[-4:];np.array([model.prob(c,y) for y in range(256)])
    else:z,s=model(x[:,length+j:length+j+1],s,True);torch.softmax(z[:,-1],-1)
   t2=time.perf_counter()
   if rep>=2:
    pre.append(1000*(t1-t));dec.append(1000*(t2-t1)/32);end.append(1000*(t2-t))
    logical.append(4 if isinstance(model,Ngram) else tensor_bytes(s));storage.append(None if isinstance(model,Ngram) else allocated_tensor_bytes(s))
 return dict(prefix_bytes=length,contexts=len(docs),retained_probes=len(end),distributions=33,prefill_p50_ms=float(np.median(pre)),sequence_mean_decode_p50_ms=float(np.median(dec)),end_to_end_p50_ms=float(np.median(end)),end_to_end_p95_ms=float(np.quantile(end,.95)),final_state_tensor_bytes=max(logical) if not isinstance(model,Ngram) else None,allocated_state_storage_bytes=max(storage) if not isinstance(model,Ngram) else None,statistical_logical_suffix_bytes=4 if isinstance(model,Ngram) else None)
def main():
 docs,train,_=load();ids=[i for i,d in enumerate(docs) if len(d)>=544][:8];chosen=[docs[i] for i in ids];rows=[]
 stat=Ngram()
 for start in np.load(V/'train_starts.npy').flatten():stat.add(train[start:start+129])
 for kind in ['transformer','gru']:
  for seed in [1,2,3]:
   base=LM(kind).eval();base.load_state_dict(torch.load(V/'checkpoints'/f'{kind}_seed{seed}_step4000.pt',weights_only=True));variants=[(kind,base)]
   if kind=='transformer':variants.append(('local_transformer',localize(base).eval()))
   for name,model in variants:rows.append(dict(model=name,seed=seed,timings=[probe(model,chosen,n) for n in [128,512]]))
 rows.append(dict(model='ngram',seed=None,timings=[probe(stat,chosen,n) for n in [128,512]]))
 write('timings.json',dict(cohort_row_indices=[i+100 for i in ids],scope='CPU float32 batch1;33 distributions,32 teacher-forced updates;all audits before this process. p95 over24 local sequence probes,not service p95;fixed order and uncontrolled host scheduling. No dollar allocation.',rows=rows))
if __name__=='__main__':main()
