"""Independent v15 artifact audit and long-context cached/full consistency checks."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,math,time
import numpy as np
import torch
from models import LM,Attention
P=Path(__file__).parent;V=Path(os.environ.get('V14_DIR',P.parent/'frontier_research_v14'))
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manual_bpb(logits,y):return float((-torch.log_softmax(logits.double(),-1).gather(-1,y[...,None]).squeeze(-1)).mean()/math.log(2))
def main():
 start=time.perf_counter();r=json.loads((P/'results.json').read_text());p=json.loads((P/'protocol.json').read_text());w=json.loads((P/'window_scores.json').read_text());rows=json.loads((P/'rows.json').read_text())['rows'];docs=[x['row']['text'].encode()[:4097] for x in rows]
 assert r['protocol_sha256']==sha(P/'protocol.json')
 for f,h in p['source_hashes'].items():assert sha(P/f)==h
 assert sha(V/'results.json')==p['v14_results_sha256'];assert sha(V/'checkpoint_manifest.json')==p['v14_checkpoint_manifest_sha256']
 observations=[]
 for row in r['neural']:
  k,seed=row['kind'],row['seed'];name=f'{k}_{seed}';file=V/'checkpoints'/f'{k}_seed{seed}_step4000.pt';assert sha(file)==row['checkpoint_sha256']
  m=LM(k).eval();m.load_state_dict(torch.load(file,weights_only=True));errs=[];cacheerrs=[];causalerrs=[];shorterrs=[];cacheloss=[]
  for idx in np.linspace(0,len(w[name]['long'])-1,6,dtype=int):
   score=w[name]['long'][idx];doc,off=score['doc'],score['start'];raw=docs[doc][off:off+513];assert len(raw)==513
   x=torch.tensor(list(raw[:-1]))[None,:];y=torch.tensor(list(raw[1:]))[None,:]
   with torch.no_grad():
    full=m(x);parts=[];state=None
    for begin in [0,128,256,384]:
     z,state=m(x[:,begin:begin+128],state,True);parts.append(z)
    cached=torch.cat(parts,1);cacheerrs.append(float((cached-full).abs().max()))
    selfloss=manual_bpb(full[:,-128:],y[:,-128:]);errs.append(abs(selfloss-score['bpb']));cacheloss.append(abs(manual_bpb(cached[:,-128:],y[:,-128:])-selfloss))
    short=m(x[:,-128:]);shorterrs.append(abs(manual_bpb(short,y[:,-128:])-w[name]['short'][idx]['bpb']))
    changed=x.clone();changed[:,400:]=(changed[:,400:]+127)%256;changed_logits=m(changed);causalerrs.append(float((full[:,:400]-changed_logits[:,:400]).abs().max()))
   assert cacheerrs[-1]<1e-4,(name,cacheerrs[-1]);assert cacheloss[-1]<2e-6;assert errs[-1]<2e-6;assert shorterrs[-1]<2e-6;assert causalerrs[-1]<2e-6
  for label in ['primary','short','long']:
   values=w[name][label];assert len(values)==(1935 if label=='primary' else 456)
   key={'primary':'primary','short':'context_short','long':'context_long'}[label]
   assert abs(np.mean([a['bpb'] for a in values])-row[key]['bpb'])<1e-12
   assert sum(a['targets'] for a in values)==row[key]['targets']
  delta=np.array([b['bpb']-a['bpb'] for a,b in zip(w[name]['short'],w[name]['long'])]);assert abs(delta.mean()-row['paired']['long_minus_short_bpb'])<1e-12
  observations.append(dict(model=name,sampled_actual_long_windows=6,max_independent_long_loss_error=max(errs),max_independent_short_loss_error=max(shorterrs),max_cached_full_logit_error=max(cacheerrs),max_cached_full_loss_error=max(cacheloss),max_future_perturbation_logit_error=max(causalerrs)))
 # Separate attention reference: explicitly rotate and form causal softmax at length512.
 torch.manual_seed(77);a=Attention().eval();x=torch.randn(1,512,64)
 with torch.no_grad():
  actual,_=a(x);q,k,v=a.qkv(x).chunk(3,-1);q,k,v=[z.reshape(1,512,4,16).transpose(1,2) for z in (q,k,v)]
  angles=torch.arange(512,dtype=x.dtype)[:,None]/torch.pow(10000.,torch.arange(8,dtype=x.dtype)[None,:]/8)
  def rot(z):
   zz=z.reshape(1,4,512,8,2);first=zz[...,0]*angles.cos()-zz[...,1]*angles.sin();second=zz[...,0]*angles.sin()+zz[...,1]*angles.cos();return torch.stack([first,second],-1).reshape_as(z)
  q,k=rot(q),rot(k);scores=(q@k.transpose(-1,-2))/4;mask=torch.triu(torch.ones(512,512,dtype=torch.bool),1);scores=scores.masked_fill(mask,float('-inf'));reference=a.out((scores.softmax(-1)@v).transpose(1,2).reshape_as(x));attention_error=float((actual-reference).abs().max());assert attention_error<2e-6
 # Verify each family saw identical doc/target IDs and the pairing's literal byte targets.
 reference_ids=[(x['doc'],x['start'],x['targets']) for x in w['ngram']['primary']]
 for scores in w.values():assert [(x['doc'],x['start'],x['targets']) for x in scores['primary']]==reference_ids
 cohorts={str(n):[i for i,d in enumerate(docs) if len(d)>=n+32][:8] for n in [128,512]}
 result=dict(status='pass',neural_rescores=observations,independent_attention_512_max_error=attention_error,timing_document_indices=cohorts,timing_same_cohort=cohorts['128']==cohorts['512'],scope='36 actual paired windows freshly rescored; all stored means/target counts checked; no full fresh neural rescore or latency rerun',wall_seconds=time.perf_counter()-start)
 (P/'audit_final.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
