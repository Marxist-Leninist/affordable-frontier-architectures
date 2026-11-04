"""Non-neural suffix n-gram with recursive Dirichlet interpolation.
Same 1,024,000 target positions and reset boundaries as each neural run.
No test-based tuning. Stores no dataset in published output.
"""
import os,json,time,math,hashlib,pickle
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
P=Path(__file__).parent;D=Path(os.environ.get('WIKITEXT_DATA',str(P.parent/'frontier_research_v8')))
class Ngram:
 def __init__(self,order=4,alpha=10.):
  self.order=order;self.alpha=alpha;self.counts=[defaultdict(Counter) for _ in range(order+1)];self.totals=[Counter() for _ in range(order+1)]
 def add(self,x):
  for i in range(1,len(x)):
   y=x[i]
   for n in range(min(i,self.order)+1):
    c=x[i-n:i] if n else b'';self.counts[n][c][y]+=1;self.totals[n][c]+=1
 def prob(self,context,y,order=None):
  maxn=min(len(context),self.order if order is None else order);p=(self.counts[0][b''].get(y,0)+1)/(self.totals[0][b'']+256)
  for n in range(1,maxn+1):
   c=context[-n:];counter=self.counts[n].get(c);total=self.totals[n].get(c,0)
   if total:p=((counter.get(y,0) if counter else 0)+self.alpha*p)/(total+self.alpha)
  return p
 def eval(self,raw,order):
  total=0.;n=((len(raw)-1)//128)*128
  for start in range(0,n,128):
   for j in range(128):
    pos=start+j;total-=math.log2(self.prob(raw[max(start,pos-order+1):pos+1],raw[pos+1],order))
  return {'bpb':total/n,'targets':n,'unscored_trailing_targets':(len(raw)-1)%128}
def main():
 manifest=json.loads((P/'data_manifest.json').read_text());raw={}
 for k,v in manifest['splits'].items():
  raw[k]=(D/(k+'.txt')).read_bytes();assert hashlib.sha256(raw[k]).hexdigest()==v['text_sha256']
 starts=np.load(P/'train_starts.npy');m=Ngram();t=time.perf_counter()
 for s in starts.flatten():m.add(raw['train'][s:s+129])
 fit=time.perf_counter()-t;validation={}
 for n in [2,3,4]:
  t=time.perf_counter();validation[n]=m.eval(raw['validation'],n);validation[n]['evaluation_seconds']=time.perf_counter()-t
  print('VALIDATION',n,validation[n],flush=True)
 selected=min(validation,key=lambda n:validation[n]['bpb']);(P/'ngram_selection.json').write_text(json.dumps({'selected_order':selected,'validation':validation},indent=2))
 t=time.perf_counter();test=m.eval(raw['test'],selected);test['evaluation_seconds']=time.perf_counter()-t
 t=time.perf_counter();lat=[]
 for j in range(500):
  c=raw['test'][10000+j:10000+j+selected];start=time.perf_counter();probs=[m.prob(c,b,selected) for b in range(256)];lat.append((time.perf_counter()-start)*1000)
 assert abs(sum(probs)-1)<1e-10
 blob=pickle.dumps(m,protocol=5);(P/'checkpoints'/'ngram.pkl').write_bytes(blob)
 result={'family':'non-neural interpolated byte ngram','orders_tuned':[2,3,4],'alpha_fixed':10,'training_targets':len(starts)*starts.shape[1]*128,'training_seconds_shared_order4_table':fit,'selected_order':selected,'validation':validation,'test':test,'next_byte_distribution_p50_ms':float(np.median(lat)),'next_byte_distribution_p95_ms':float(np.quantile(lat,.95)),'timing_samples':500,'serialized_all_order_tables_bytes':len(blob),'table_context_counts':[len(x) for x in m.counts],'table_nonzero_entries':[sum(len(v) for v in x.values()) for x in m.counts],'training_windows_sha256':hashlib.sha256((P/'train_starts.npy').read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(blob).hexdigest(),'checkpoint_warning':'Local trusted pickle only; never load untrusted pickles','fairness':'Identical sampled byte targets, not parameter/FLOP matched. Three candidate orders share one max-order fit. Deterministic; repeated initialization seeds are not meaningful. CPU wall times can be affected by concurrent neural training.'}
 (P/'ngram_results.json').write_text(json.dumps(result,indent=2));print('DONE',result,flush=True)
if __name__=='__main__':main()
