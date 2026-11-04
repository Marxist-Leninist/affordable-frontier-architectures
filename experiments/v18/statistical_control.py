"""Frozen statistical control on the exact evaluation positions; no tuning."""
import os,json,time,math,statistics
from pathlib import Path
import numpy as np
from run import P,V,load,sha,write
from ngram import Ngram
D=Path(os.environ.get('WIKITEXT_DATA',P.parent/'frontier_research_v8'))
def build():
 vp=json.loads((V/'protocol.json').read_text());path=D/'train.txt';assert sha(path)==vp['dataset']['splits']['train']['text_sha256'];assert sha(V/'train_starts.npy')==vp['ledger_sha256'];raw=path.read_bytes();m=Ngram()
 for start in np.load(V/'train_starts.npy').flatten():m.add(raw[start:start+129])
 m.order=json.loads((V/'selection.json').read_text())['ngram_order'];assert m.order==4
 return m

def main():
 t=time.perf_counter();m=build();fit=time.perf_counter()-t;docs=load();rows=[]
 for length in [2048]:
  out=[]
  for i,d in enumerate(docs):
   for start in range(0,len(d)-length,length):
    raw=d[start:start+length+1];loss=0.
    for pos in range(length-128,length):loss-=math.log2(m.prob(raw[pos-3:pos+1],raw[pos+1]))
    out.append(dict(row_index=i+300,start=start,bpb=loss/128))
  rows.append(dict(input_bytes=length,mean_bpb=statistics.mean(x['bpb'] for x in out),windows=len(out),targets=len(out)*128,window_scores=out))
 write('statistical_results.json',dict(rebuild_seconds=fit,rows=rows,total_seconds=time.perf_counter()-t))
if __name__=='__main__':main()
