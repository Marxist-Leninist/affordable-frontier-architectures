"""Independent frozen-checkpoint and real-window verification for v16."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,math,time,statistics
from pathlib import Path
import torch,numpy as np
from models import LM
from local_attention import localize,tensor_bytes,allocated_tensor_bytes
P=Path(__file__).parent;V=Path(os.environ.get('V14_DIR',P.parent/'frontier_research_v14'))
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bpb(z,y):return float(-z.double().log_softmax(-1).gather(-1,y[...,None]).mean()/math.log(2))
def main():
 begin=time.perf_counter();r=json.loads((P/'results.json').read_text());protocol=json.loads((P/'protocol.json').read_text());details=json.loads((P/'window_scores.json').read_text());docs=[x['row']['text'].encode()[:4097] for x in json.loads((P/'rows.json').read_text())['rows']]
 assert sha(P/'protocol.json')==r['protocol_sha256'];assert sha(V/'checkpoint_manifest.json')==protocol['v14_manifest_sha256'];assert sha(V/'results.json')==protocol['v14_results_sha256']
 for f,h in protocol['source_hashes'].items():assert sha(P/f)==h
 records=[]
 for row in r['neural']:
  kind,seed=row['kind'],row['seed'];name=f'{kind}_{seed}';p=V/'checkpoints'/f'{kind}_seed{seed}_step4000.pt';assert sha(p)==row['checkpoint_sha256'];base=LM(kind).eval();base.load_state_dict(torch.load(p,weights_only=True));model=localize(base).eval() if kind=='transformer' else base
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
  for condition,field in [('reset','reset128'),('full','full512'),('primary','primary128')]+([('local','local512')] if kind=='transformer' else []):
   rows=details[name][condition];assert abs(statistics.mean(x['bpb'] for x in rows)-row[field]['bpb'])<1e-12;assert len(rows)*128==row[field]['targets'];assert len(rows)==row[field]['windows']
  if kind=='transformer':
   reset,local=details[name]['reset'],details[name]['local'];assert [(x['doc'],x['start']) for x in reset]==[(x['doc'],x['start']) for x in local];delta=statistics.mean(y['bpb']-x['bpb'] for x,y in zip(reset,local));assert abs(delta-row['local_vs_reset']['b_minus_a_bpb'])<1e-12;assert (abs(delta)<=.02)==row['local_vs_reset']['abs_aggregate_gap_le_002']
  loss_errors=[];cache_errors=[];cache_loss_errors=[];primary_errors=[];memory=[]
  for idx in np.linspace(0,len(details[name]['full'])-1,6,dtype=int):
   rec=details[name]['full'][idx];raw=docs[rec['doc']-100][rec['start']:rec['start']+513];x=torch.tensor(list(raw[:-1]))[None,:];y=torch.tensor(list(raw[1:]))[None,:]
   with torch.no_grad():
    z=model(x);field='local' if kind=='transformer' else 'full';loss_errors.append(abs(bpb(z[:,-128:],y[:,-128:])-details[name][field][idx]['bpb']))
    zbase=base(x);loss_errors.append(abs(bpb(zbase[:,-128:],y[:,-128:])-rec['bpb']))
    zshort=base(x[:,-128:]);loss_errors.append(abs(bpb(zshort,y[:,-128:])-details[name]['reset'][idx]['bpb']))
    parts=[];state=None
    for a,c in [(0,64),(64,128),(128,256),(256,257),(257,512)]:zz,state=model(x[:,a:c],state,True);parts.append(zz)
    cat=torch.cat(parts,1);cache_errors.append(float((z-cat).abs().max()));cache_loss_errors.append(abs(bpb(z[:,-128:],y[:,-128:])-bpb(cat[:,-128:],y[:,-128:])))
    memory.append(dict(tensor_bytes=tensor_bytes(state),allocated_bytes=allocated_tensor_bytes(state)))
    if kind=='transformer':
     for k,v,offset in state:
      assert offset.item()==512 and k.shape[-2]==v.shape[-2]==127
      for a in (k,v):assert a._base is None and a.untyped_storage().nbytes()==a.numel()*a.element_size()
     primary_errors.append(float((base(x[:,:128])-model(x[:,:128])).abs().max()))
  assert max(loss_errors)<2e-6;assert max(cache_errors)<1e-4;assert max(cache_loss_errors)<2e-6
  if primary_errors:assert max(primary_errors)<1e-4;assert all(m['tensor_bytes']==m['allocated_bytes']==130064 for m in memory)
  records.append(dict(kind=kind,seed=seed,sampled_long_windows=6,max_fresh_loss_error=max(loss_errors),max_cached_full_logit_error=max(cache_errors),max_cached_full_loss_error=max(cache_loss_errors),max_primary128_logit_error=max(primary_errors) if primary_errors else None,cache=memory[0]))
 stat=details['ngram']['full'];assert abs(statistics.mean(x['bpb'] for x in stat)-r['ngram']['full512']['bpb'])<1e-12
 result=dict(status='pass',records=records,scope='36 real512 windows with independent float64 loss; all stored means and main gate arithmetic; local checkpoint weights identical; actual cloned storage inspected. Not a fresh full-corpus rescore or statistical-table rebuild.',seconds=time.perf_counter()-begin)
 (P/'audit_final.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
