"""Frozen exact-quality chunked prefill test; timing deliberately separate."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,math,statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import LM
from local_attention import localize,tensor_bytes,allocated_tensor_bytes
from prefill import prefill
P=Path(__file__).parent;V=Path(os.environ.get('V14_DIR',P.parent/'frontier_research_v14'));torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(f,x):(P/f).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load():
 mf=json.loads((P/'data_manifest.json').read_text());assert sha(P/'rows.json')==mf['response_sha256'];rows=json.loads((P/'rows.json').read_text())['rows'];assert [r['row_idx'] for r in rows]==list(range(200,300));docs=[r['row']['text'].encode()[:4097] for r in rows]
 for d,r in zip(docs,mf['documents']):assert hashlib.sha256(d).hexdigest()==r['capped_text_sha256']
 assert sha(P/'models.py')==sha(V/'models.py')
 return docs,mf
@torch.inference_mode()
def evaluate(seed,length,docs):
 items=[(i,s,d[s:s+length+1]) for i,d in enumerate(docs) for s in range(0,len(d)-length,length)];manifest=json.loads((V/'checkpoint_manifest.json').read_text());base=LM('transformer').eval();path=V/'checkpoints'/f'transformer_seed{seed}_step4000.pt';assert sha(path)==manifest[path.name];base.load_state_dict(torch.load(path,weights_only=True));local=localize(base).eval();gru=LM('gru').eval();path=V/'checkpoints'/f'gru_seed{seed}_step4000.pt';assert sha(path)==manifest[path.name];gru.load_state_dict(torch.load(path,weights_only=True));scores={k:[] for k in ['full_transformer','local_monolithic','local_chunked','gru']};logitmax=0.;state_max=0.;state_storage=[]
 for begin in range(0,len(items),4):
  b=items[begin:begin+4];raw=torch.tensor([list(r[2]) for r in b],dtype=torch.long);x=raw[:,:-1];y=raw[:,-128:];z={};states={}
  for name,m,chunk in [('full_transformer',base,None),('local_monolithic',local,None),('local_chunked',local,128),('gru',gru,None)]:
   z[name],states[name]=prefill(m,x,chunk,128);loss=F.cross_entropy(z[name].flatten(0,1),y.flatten(),reduction='none').reshape(len(b),128)
   scores[name].extend(dict(row_index=i+200,start=s,bpb=float(l.double().mean()/math.log(2))) for (i,s,_),l in zip(b,loss))
  logitmax=max(logitmax,float((z['local_monolithic']-z['local_chunked']).abs().max()))
  for s1,s2 in zip(states['local_monolithic'],states['local_chunked']):
   for a,c in zip(s1,s2):state_max=max(state_max,float((a-c).abs().max()))
  state_storage.append(dict(batch=len(b),monolithic=allocated_tensor_bytes(states['local_monolithic']),chunked=allocated_tensor_bytes(states['local_chunked'])))
 diff=[b['bpb']-a['bpb'] for a,b in zip(scores['local_monolithic'],scores['local_chunked'])]
 summary=dict(seed=seed,input_bytes=length,windows=len(items),targets=len(items)*128,documents=len(set(i for i,_,_ in items)),mean_bpb={k:statistics.mean(x['bpb'] for x in v) for k,v in scores.items()},max_abs_local_logits_error=logitmax,max_abs_local_state_error=state_max,max_abs_window_bpb_error=max(abs(x) for x in diff),mean_chunked_minus_monolithic_bpb=statistics.mean(diff),all_allocated_storage_equal=all(x['monolithic']==x['chunked'] for x in state_storage),strict_quality_parity=max(abs(x) for x in diff)<=2e-6 and logitmax<=1e-4)
 return summary,scores

def main():
 start=time.perf_counter();docs,mf=load();protocol=dict(experiment='v17 exact local-attention chunked prefill',dataset_manifest_sha256=sha(P/'data_manifest.json'),source_hashes={f:sha(P/f) for f in ['run.py','prefill.py','local_attention.py','models.py','prepare_data.py']},settings='Fresh rows200-299;fixed chunk128 from attention window;frozen v14 seeds1/2/3;no fitting/tuning;score identical last128 targets per512/2048 document window.',quality_gate='Every window abs bpb difference<=2e-6 and max local logits difference<=1e-4; no quality-reducing approximation. Preserve failed results.',baseline='Both monolithic and chunked compute only required final output-tail head. Full Transformer and GRU receive same output-head optimization.',v14_checkpoint_manifest_sha256=sha(V/'checkpoint_manifest.json'),torch=torch.__version__,memory='Persistent state measured; largest explicit mask shape derived separately, not measured whole-process peak. No linear-cost or throughput claim from shape alone.')
 write('protocol.json',protocol);results=[];details={}
 for seed in [1,2,3]:
  for length in [512,2048]:
   row,scores=evaluate(seed,length,docs);results.append(row);details[f'seed{seed}_length{length}']=scores;write('partial.json',results);print(row,flush=True)
 write('window_scores.json',details);write('results.json',dict(protocol_sha256=sha(P/'protocol.json'),rows=results,all_quality_gates_pass=all(r['strict_quality_parity'] for r in results),wall_seconds=time.perf_counter()-start));print('DONE',flush=True)
if __name__=='__main__':main()
