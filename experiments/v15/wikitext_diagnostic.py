"""Post-hoc domain control, triggered by unexpected FineWeb512 Transformer degradation.
No model changes, tuning, checkpoint selection, or performance measurement.
"""
import json,time
from run import P,V,D,sha,LM,torch,neural_scores,aggregate,pair_summary,write

def main():
 t=time.perf_counter();vp=json.loads((V/'protocol.json').read_text());path=D/'test.txt';assert sha(path)==vp['dataset']['splits']['test']['text_sha256'];raw=path.read_bytes()
 items=[(i,i*512,raw[i*512:i*512+513]) for i in range(100)];short=[(i,s,r[-129:]) for i,s,r in items];rows=[]
 for kind in ['transformer','gru']:
  for seed in [1,2,3]:
   path=V/'checkpoints'/f'{kind}_seed{seed}_step4000.pt';m=LM(kind).eval();m.load_state_dict(torch.load(path,weights_only=True));a=neural_scores(m,short);b=neural_scores(m,items)
   rows.append(dict(kind=kind,seed=seed,short={k:v for k,v in aggregate(a).items() if k!='documents'},long={k:v for k,v in aggregate(b).items() if k!='documents'},long_minus_short_bpb=aggregate(b)['bpb']-aggregate(a)['bpb']))
 write('wikitext_diagnostic.json',dict(scope='Post-hoc domain control, first100 contiguous512-byte WikiText test windows,128 tail targets each. No retraining/tuning or fresh heldout claim. Same source corpus as training, distinct official test split; windows from flat serialization may cross article boundaries, unlike the FineWeb document reset protocol.',rows=rows,seconds=time.perf_counter()-t))
if __name__=='__main__':main()
