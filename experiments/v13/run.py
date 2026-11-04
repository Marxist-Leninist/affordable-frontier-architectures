"""Frozen-checkpoint statistical/neural complementarity gate, no test tuning."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import json,time,hashlib,math,platform
from pathlib import Path
import numpy as np
import torch
from models import LM,count,state_bytes
from ngram import Ngram
P=Path(__file__).parent; V12=Path(os.environ.get('V12_DIR',P.parent/'frontier_research_v12')); D=Path(os.environ.get('WIKITEXT_DATA',P.parent/'frontier_research_v8'))
torch.set_num_threads(1)
GRID=np.arange(11,dtype=float)/10

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(P/n).write_text(json.dumps(x,indent=2)+'\n')
def mix_bpb(neural,stat,weight):return float(-np.log2((1-weight)*neural+weight*stat).mean())
def stat_probs(m,raw):
 n=(len(raw)-1)//128*128
 return np.array([m.prob(raw[max(i//128*128,i-3):i+1],raw[i+1],4) for i in range(n)],dtype=np.float64)
@torch.inference_mode()
def neural_probs(m,raw,batch=32):
 a=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()).long();n=(len(raw)-1)//128;out=[]
 for first in range(0,n,batch):
  ix=torch.arange(first,min(n,first+batch))[:,None]*128+torch.arange(128)
  logp=m(a[ix]).log_softmax(-1).gather(-1,a[ix+1,None]).squeeze(-1)
  out.append(logp.double().exp().flatten().numpy())
 return np.concatenate(out)
def load(kind,seed):
 f=V12/'checkpoints'/f'{kind}_seed{seed}_lr0.01.pt';record=json.loads((V12/'checkpoint_manifest.json').read_text())[f.name];assert sha(f)==record['sha256']
 m=LM(kind);m.load_state_dict(torch.load(f,weights_only=True));m.eval();return m
@torch.inference_mode()
def timing(m,ng,raw,weight):
 # Same short trained context; repeated reset avoids claiming long-context quality.
 times={'neural':[],'ngram':[],'hybrid':[]};arr=torch.tensor(list(raw[10000:10200]))
 for rep in range(60):
  _,state=m(arr[:127][None,:],cache=True);x=arr[127:128][None,:];ctx=bytes(arr[124:128].tolist())
  t=time.perf_counter();logits,_=m(x,state,True);p=logits[0,0].softmax(-1).numpy();n=(time.perf_counter()-t)*1000
  t=time.perf_counter();q=np.array([ng.prob(ctx,y,4) for y in range(256)]);s=(time.perf_counter()-t)*1000
  t=time.perf_counter();r=(1-weight)*p+weight*q;mix=(time.perf_counter()-t)*1000
  assert abs(r.sum()-1)<1e-6
  if rep>=10:
   times['neural'].append(n);times['ngram'].append(s);times['hybrid'].append(n+s+mix)
 return {k:{'p50_ms':float(np.median(v)),'p95_ms':float(np.quantile(v,.95)),'samples':len(v)} for k,v in times.items()}
def main():
 begin=time.perf_counter();manifest=json.loads((P/'data_manifest.json').read_text());raw={}
 for split,info in manifest['splits'].items():
  p=D/(split+'.txt');assert sha(p)==info['text_sha256'];raw[split]=p.read_bytes()
 starts=np.load(V12/'train_starts.npy');write('protocol.json',{'experiment':'v13 frozen neural plus statistical mixture','weights_ngram':GRID.tolist(),'selection':'Full validation only, pilot seed101 for each architecture. Freeze weights before scoring three final test seeds.','training':'No new neural training. Rebuild same v12 ngram from identical ledger. Combined systems incur both original component training costs.','evaluation':'Identical complete 128-byte windows, reset each; no adaptive counts or test updates','v12_checkpoint_manifest_sha256':sha(V12/'checkpoint_manifest.json'),'ledger_sha256':sha(V12/'train_starts.npy'),'dataset':manifest,'source_sha256':{f:sha(P/f) for f in ['run.py','models.py','ngram.py']},'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()})
 ng=Ngram();t=time.perf_counter()
 for s in starts.flatten():ng.add(raw['train'][s:s+129])
 fit=time.perf_counter()-t;t=time.perf_counter();qval=stat_probs(ng,raw['validation']);valstatsec=time.perf_counter()-t
 selection={};timings={'ngram_rebuild_seconds':fit,'ngram_validation_target_scoring_seconds':valstatsec}
 for kind in ['transformer','gru']:
  t=time.perf_counter();p=neural_probs(load(kind,101),raw['validation']);rows=[{'weight':float(w),'bpb':mix_bpb(p,qval,w)} for w in GRID];selection[kind]={'grid':rows,'selected_weight':min(rows,key=lambda x:x['bpb'])['weight']};timings[kind+'_pilot_validation_seconds']=time.perf_counter()-t
 write('selection.json',selection);print('SELECTED',selection,flush=True)
 t=time.perf_counter();q=stat_probs(ng,raw['test']);timings['ngram_test_target_scoring_seconds']=time.perf_counter()-t;rows=[]
 for kind in ['transformer','gru']:
  for seed in [1,2,3]:
   m=load(kind,seed);t=time.perf_counter();p=neural_probs(m,raw['test']);sec=time.perf_counter()-t;w=selection[kind]['selected_weight'];base=mix_bpb(p,q,0);stat=mix_bpb(p,q,1);hybrid=mix_bpb(p,q,w)
   # descriptive same-corpus block gains, not independent cross-corpus confidence
   gain=(np.log2((1-w)*p+w*q)-np.log2(q)).reshape(-1,128).mean(1)
   row={'kind':kind,'seed':seed,'ngram_weight':w,'neural_test_bpb':base,'ngram_test_bpb':stat,'hybrid_test_bpb':hybrid,'gain_vs_ngram_bpb':stat-hybrid,'fraction_128byte_blocks_improved_over_ngram':float((gain>0).mean()),'test_targets':len(q),'neural_target_scoring_seconds':sec,'raw_neural_weights_bytes':count(m)*4,'ngram_serialized_bytes_v12':2820855,'combined_serialized_bytes':2820855+(V12/'checkpoints'/f'{kind}_seed{seed}_lr0.01.pt').stat().st_size,'timing':timing(m,ng,raw['test'],w)}
   rows.append(row);write('partial.json',rows);print('RESULT',row,flush=True)
 timings['total_script_seconds']=time.perf_counter()-begin
 write('results.json',{'selection':selection,'rows':rows,'timing':timings,'summary':{kind:{'mean_hybrid_bpb':float(np.mean([r['hybrid_test_bpb'] for r in rows if r['kind']==kind])),'sample_sd_hybrid_bpb':float(np.std([r['hybrid_test_bpb'] for r in rows if r['kind']==kind],ddof=1)),'mean_neural_bpb':float(np.mean([r['neural_test_bpb'] for r in rows if r['kind']==kind]))} for kind in ['transformer','gru']}})
if __name__=='__main__':main()
