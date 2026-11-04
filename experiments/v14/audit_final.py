"""Independent final-artifact verification, sampled neural rescore, full statistical rescore."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,math,statistics,time
import numpy as np
import torch
from models import LM
from ngram import Ngram
P=Path(__file__).parent;D=Path(os.environ.get('WIKITEXT_DATA',P.parent/'frontier_research_v8'))
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 begin=time.perf_counter();r=json.loads((P/'results.json').read_text());protocol=json.loads((P/'protocol.json').read_text());selection=json.loads((P/'selection.json').read_text());summary=json.loads((P/'summary.json').read_text())
 assert r['protocol_sha256']==sha(P/'protocol.json');assert r['selection_sha256']==sha(P/'selection.json')
 for f,s in protocol['source_hashes'].items():assert sha(P/f)==s
 raw={}
 for split,row in protocol['dataset']['splits'].items():
  raw[split]=(D/(split+'.txt')).read_bytes();assert hashlib.sha256(raw[split]).hexdigest()==row['text_sha256']
 for kind in ['transformer','gru']:
  pilots=[x for x in r['tuning'] if x['kind']==kind]
  assert len(pilots)==3 and sorted(x['lr'] for x in pilots)==[.003,.01,.03]
  best=min(pilots,key=lambda x:x['curve'][-1]['validation']['bpb'])['lr'];assert best==selection['neural'][kind]==.01
  finals=[x for x in r['final'] if x['kind']==kind];assert sorted(x['seed'] for x in finals)==[1,2,3]
  vals=[next(y['test']['bpb'] for y in x['evaluated_checkpoints'] if y['step']==4000) for x in finals]
  assert abs(statistics.mean(vals)-summary['families'][kind]['mean_test_bpb'])<1e-12
  assert abs(statistics.stdev(vals)-summary['families'][kind]['seed_sample_sd_bpb'])<1e-12
  assert abs(sum(x['training_seconds'] for x in pilots+finals)-summary['families'][kind]['total_training_seconds_including_pilots'])<1e-8
 n=(len(raw['test'])-1)//128; ids=np.unique(np.r_[np.linspace(0,n-1,17,dtype=int),1,31,32]);observations=[]
 for row in r['final']:
  gates={str(g):next((x['step'] for x in row['curve'] if x['validation']['bpb']<=g),None) for g in protocol['gates_validation_bpb']}
  assert gates==row['validation_gate_steps']
  expected=sorted({4000}|{s for s in gates.values() if s is not None});assert expected==[cp['step'] for cp in row['evaluated_checkpoints']]
  for cp in row['evaluated_checkpoints']:
   path=P/'checkpoints'/cp['checkpoint_file'];assert sha(path)==cp['checkpoint_sha256'];assert path.stat().st_size==cp['checkpoint_bytes']
   blocks=np.load(P/f"{row['kind']}_seed{row['seed']}_step{cp['step']}_test_blocks.npy");assert blocks.shape==(n,)
   assert abs(blocks.mean()-cp['test']['bpb'])<1e-6
   assert cp['test']['targets']==n*128 and cp['test']['unscored_trailing_targets']==(len(raw['test'])-1)%128
   m=LM(row['kind']).eval();m.load_state_dict(torch.load(path,weights_only=True))
   # Independently constructed windows and float64 logsumexp gather, not run.evaluate.
   x=torch.tensor([list(raw['test'][int(i)*128:int(i)*128+128]) for i in ids],dtype=torch.long)
   y=torch.tensor([list(raw['test'][int(i)*128+1:int(i)*128+129]) for i in ids],dtype=torch.long)
   with torch.no_grad():
    logits=m(x).double();bpb=((torch.logsumexp(logits,-1)-logits.gather(-1,y[...,None]).squeeze(-1)).mean(1)/math.log(2)).numpy()
   error=float(np.max(np.abs(bpb-blocks[ids])));assert error<2e-6
   observations.append(dict(kind=row['kind'],seed=row['seed'],step=cp['step'],sampled_windows=len(ids),max_absolute_bpb_error=error,all_window_mean_error=float(abs(blocks.mean()-cp['test']['bpb']))))
 for gate in summary['quality_gates']:
  all_ok=True
  for pair in gate['pairs']:
   actual={}
   for row in r['final']:
    if row['seed']!=pair['seed']:continue
    step=row['validation_gate_steps'][str(gate['validation_threshold'])]
    if step is not None:
     cp=next(c for c in row['evaluated_checkpoints'] if c['step']==step);actual[row['kind']]=(step,cp['test']['bpb'])
   assert set(actual)==set(pair['selected'])
   for kind,(step,bpb) in actual.items():assert pair['selected'][kind]['step']==step and pair['selected'][kind]['test_bpb']==bpb
   assert pair['both_validation_gates_reached']==(len(actual)==2)
   if len(actual)==2:
    gap=actual['gru'][1]-actual['transformer'][1];assert gap==pair['test_gap_gru_minus_transformer_bpb'];ok=abs(gap)<=.02;assert ok==pair['within_descriptive_002_bpb_screen']
   else:ok=False
   all_ok &= ok
  assert all_ok==gate['all_seeds_reached_and_within_screen']==False
 starts=np.load(P/'train_starts.npy');ng=Ngram()
 for offset in starts.flatten():ng.add(raw['train'][offset:offset+129])
 order=min(selection['ngram_validation'],key=lambda k:selection['ngram_validation'][k]['bpb']);assert int(order)==r['ngram']['selected_order']==4
 # Full independent sequential target/context scorer; same probability implementation is reused.
 total=0.
 for block in range(n):
  base=block*128
  for j in range(128):
   context=raw['test'][base+max(0,j-3):base+j+1];target=raw['test'][base+j+1]
   total-=math.log2(ng.prob(context,target,4))
 score=total/(n*128);assert abs(score-r['ngram']['test']['bpb'])<1e-12
 out=dict(status='pass',checkpoint_count=len(observations),neural_sample_rescores=observations,ngram_full_test_bpb=score,ngram_full_test_targets=n*128,verified=['dataset/source/protocol/selection/checkpoint hashes','all pilot LR selections','all final gate/checkpoint choices','summary means/sample SD/training totals','all gate pair gaps and failed descriptive screens','neural sample scores via float64 logsumexp','full statistical test score'],scope='20 distributed windows per neural checkpoint, not full neural rescore; every saved block mean checked. Ngram fully rescored with independent loop, shared probability routine. No timing repeated.',wall_seconds=time.perf_counter()-begin)
 (P/'audit_final.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
