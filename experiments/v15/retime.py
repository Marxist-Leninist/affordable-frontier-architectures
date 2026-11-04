"""Audit follow-up: isolated timing, same eight eligible documents at both lengths.
Retain original run timings; do not rerun quality or select models by timings.
"""
import json,time,numpy as np
from run import P,V,LM,Ngram,torch,load_inputs,timing,write

def main():
 mf,docs,vr,train=load_inputs();ids=[i for i,d in enumerate(docs) if len(d)>=544][:8];eligible=[docs[i] for i in ids];stat=Ngram()
 for s in np.load(V/'train_starts.npy').flatten():stat.add(train[s:s+129])
 stat.order=json.loads((V/'selection.json').read_text())['ngram_order'];rows=[]
 for kind in ['transformer','gru']:
  for seed in [1,2,3]:
   m=LM(kind).eval();m.load_state_dict(torch.load(V/'checkpoints'/f'{kind}_seed{seed}_step4000.pt',weights_only=True));rows.append(dict(kind=kind,seed=seed,timing=[timing(m,eligible,n) for n in [128,512]]))
 rows.append(dict(kind='ngram',seed=None,timing=[timing(stat,eligible,n) for n in [128,512]]))
 write('retiming.json',dict(reason='Original timings possibly overlapped tiny audit tests and length cohorts could differ. This rerun follows all CPU audits and uses identical document IDs across lengths. No whole-host isolation guarantee.',document_indices=ids,rows=rows))
if __name__=='__main__':main()
