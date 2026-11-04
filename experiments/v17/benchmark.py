"""Randomized-order timing after audits. All neural prefill heads compute final position only."""
import json,time,statistics
import numpy as np
from run import P,V,LM,torch,load,write
from local_attention import localize,allocated_tensor_bytes
from prefill import prefill
from statistical_control import build
from ngram import Ngram
@torch.inference_mode()
def probe(model,x,chunk=None):
 if isinstance(model,Ngram):raw=bytes(x[0].tolist())
 t=time.perf_counter()
 if isinstance(model,Ngram):n=len(raw)-32;c=raw[:n][-4:];np.array([model.prob(c,y) for y in range(256)])
 else:n=x.shape[1]-32;z,state=prefill(model,x[:,:n],chunk,1);torch.softmax(z[:,-1],-1)
 t1=time.perf_counter()
 for j in range(32):
  if isinstance(model,Ngram):c=(c+raw[n+j:n+j+1])[-4:];np.array([model.prob(c,y) for y in range(256)])
  else:z,state=model(x[:,n+j:n+j+1],state,True);torch.softmax(z[:,-1],-1)
 t2=time.perf_counter()
 return dict(prefill_ms=1000*(t1-t),decode_mean_ms=1000*(t2-t1)/32,end_ms=1000*(t2-t),allocated_state_bytes=None if isinstance(model,Ngram) else allocated_tensor_bytes(state))
def main():
 docs,_=load();ids=[i for i,d in enumerate(docs) if len(d)>=2080][:8];assert len(ids)==8;stat=build();rows=[];rng=np.random.default_rng(17112026)
 protocol=dict(cohort_rows=[i+200 for i in ids],prefix_bytes=[128,512,2048],chunk_size_fixed=128,output_head='final1 position for all neural monolithic/chunked prefill',repetitions=6,warmups=2,retained=32,ordering='seed17112026 permutation of variants per document/repetition; paired local measurements on same document/repetition;statistical onlyseed1',distributions=33,scope='Single CPU thread float32 batch1;32 observed updates;no sampling/network/queue/service cost;timing after audits')
 write('timing_protocol.json',protocol)
 for seed in [1,2,3]:
  base=LM('transformer').eval();base.load_state_dict(torch.load(V/'checkpoints'/f'transformer_seed{seed}_step4000.pt',weights_only=True));local=localize(base).eval();gru=LM('gru').eval();gru.load_state_dict(torch.load(V/'checkpoints'/f'gru_seed{seed}_step4000.pt',weights_only=True));variants=[('full_transformer',base,None),('local_monolithic',local,None),('local_chunked',local,128),('gru',gru,None)]
  if seed==1:variants.append(('ngram',stat,None))
  for length in [128,512,2048]:
   for i in ids:
    x=torch.tensor(list(docs[i][:length+32]),dtype=torch.long)[None,:]
    for rep in range(6):
     for order in rng.permutation(len(variants)):
      name,model,chunk=variants[order];r=probe(model,x,chunk)
      if rep>=2:rows.append(dict(model=name,seed=seed,prefix_bytes=length,row_index=i+200,repetition=rep,**r))
 write('timings.json',dict(protocol=protocol,samples=rows))
if __name__=='__main__':main()
