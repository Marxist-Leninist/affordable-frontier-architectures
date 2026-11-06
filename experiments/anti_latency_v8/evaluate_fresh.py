"""Fresh heldout document replication of frozen v7 models; no training or tuning."""
from run import *
import hashlib,numpy as np

def digest(x):return hashlib.sha256(x).hexdigest()
def prepare():
 spec=json.loads((P/'protocol.json').read_text());raw=(P/'rows.json').read_bytes();assert digest(raw)==spec['response_sha256'];source=json.loads(raw);assert not source['partial'];assert len(source['rows'])==100
 data_dir=Path(os.environ['ANTI_DATA_DIR']);manifest=json.loads((P/'source_data_manifest.json').read_text());train=(data_dir/'train.txt').read_bytes();assert digest(train)==manifest['splits']['train']['text_sha256']
 prior_paths=sorted(P.parent.glob('frontier_research_v*/rows.json'));prior_hashes=set();prior_text=[];source_receipts=[]
 for path in prior_paths:
  j=json.loads(path.read_text());source_receipts.append({'directory':path.parent.name,'sha256':digest(path.read_bytes())})
  for r in j['rows']:
   b=r['row']['text'].encode();h=digest(b)
   if h not in prior_hashes:prior_text.append(b);prior_hashes.add(h)
 seen=set();rows=[];excluded=[];windows=[]
 for i,r in enumerate(source['rows']):
  assert r['row_idx']==1600+i and not r['truncated_cells'];b=r['row']['text'].encode();h=digest(b);reason=None
  if len(b)<513:reason='shorter_than513'
  elif h in prior_hashes or h in seen:reason='exact_document_duplicate'
  else:
   target=b[385:513]
   if target in train:reason='scored_target_in_wikitext_train'
   elif any(target in prior for prior in prior_text):reason='scored_target_in_prior_fineweb'
  seen.add(h)
  if reason:excluded.append({'row':1600+i,'reason':reason});continue
  rows.append({'row':1600+i,'full_text_sha256':h,'window_sha256':digest(b[:513]),'target_sha256':digest(b[385:513]),'source_bytes':len(b)});windows.append(list(b[:513]))
 assert len(rows)>=40
 receipt={'dataset':spec['dataset'],'config':spec['config'],'split':spec['split'],'revision':spec['revision'],'response_sha256':digest(raw),'selected_documents':len(rows),'targets_per_document':128,'targets_per_model_context':len(rows)*128,'documents':rows,'excluded':excluded,'prior_source_receipts':source_receipts,'prior_unique_documents':len(prior_hashes),'training_sha256':digest(train),'tokenizer':'UTF8 bytes, no normalization','license':'ODC-BY dataset; underlying web text may retain third-party rights','near_duplicate_check':'not performed'}
 path=P/'data_manifest.json'
 if path.exists():assert json.loads(path.read_text())==receipt,'Dataset or prior-cohort provenance mismatch; do not refreeze'
 else:path.write_text(json.dumps(receipt,indent=2))
 return torch.tensor(windows),receipt

def load(kind,seed,arm,opt=False):
 root=Path(os.environ.get('ANTI_V7_CHECKPOINT_DIR',str(P.parent/'anti_latency_v7')));path=root/f'{kind}_{seed}_{arm}.pt';expected=json.loads((P/'expected_checkpoint_hashes.json').read_text());assert digest(path.read_bytes())==expected[path.name],'Checkpoint differs from published v7';saved=torch.load(path,weights_only=True);m=previous.model(kind);m.load_state_dict(saved['model'])
 if opt:
  for block in m.blocks:
   old=block.mix;new=Optimized(kind);new.load_state_dict(old.state_dict())
   if kind=='signed':
    with torch.no_grad():new.out.weight.neg_()
   block.mix=new
 return m.eval(),digest(path.read_bytes())

def main():
 t=time.perf_counter();data,receipt=prepare();out=[]
 with torch.inference_mode():
  for kind in ['attention','signed']:
   for seed in [1,2,3]:
    for arm in ['short','long','mixed']:
     m,h=load(kind,seed,arm)
     for n in [128,256,512]:
      per=[]
      for x in data.split(8):
       z=m(x[:,-n-1:-1])[:,-128:];y=x[:,-128:];loss=-z.double().log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)/math.log(2);per.extend(loss.mean(1).tolist())
      out.append({'kind':kind,'seed':seed,'arm':arm,'context':n,'bpb':statistics.mean(per),'document_bpb':per,'checkpoint_sha256':h})
     (P/'results.json').write_text(json.dumps({'rows':out,'complete':len(out)==54,'elapsed_wall_seconds':time.perf_counter()-t},indent=2));print(kind,seed,arm,[(r['context'],round(r['bpb'],6)) for r in out[-3:]],flush=True)
if __name__=='__main__':main()
