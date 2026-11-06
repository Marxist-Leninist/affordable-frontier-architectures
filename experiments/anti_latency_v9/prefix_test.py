from run import *
import hashlib,numpy as np
from compact_cache import CompactCache

def sha(x):return hashlib.sha256(x).hexdigest()
def prepare():
 spec=json.loads((P/'protocol.json').read_text());raw=(P/'rows.json').read_bytes();assert sha(raw)==spec['recipient_response_sha256'];rec=json.loads(raw);donorpath=Path(os.environ.get('ANTI_DONOR_ROWS',str(P.parent/'anti_latency_v8/rows.json')));dr=donorpath.read_bytes();assert sha(dr)==spec['donor_response_sha256'];donors=json.loads(dr)['rows'];assert not rec['partial'] and len(rec['rows'])==len(donors)==100
 trainraw=(Path(os.environ['ANTI_DATA_DIR'])/'train.txt').read_bytes();source=json.loads((P/'source_data_manifest.json').read_text());assert sha(trainraw)==source['splits']['train']['text_sha256'];seen=set();clean=[]
 for line in trainraw.splitlines(keepends=True):
  key=line.strip()
  if key and key in seen:continue
  if key:seen.add(key)
  clean.append(line)
 train=b''.join(clean)[:2000000];prior_paths=sorted(P.parent.glob('frontier_research_v*/rows.json'))+[donorpath];priorhash=set();prior=[];receipts=[]
 for path in prior_paths:
  content=path.read_bytes();receipts.append({'directory':path.parent.name,'sha256':sha(content)})
  for r in json.loads(content)['rows']:
   b=r['row']['text'].encode();h=sha(b)
   if h not in priorhash:priorhash.add(h);prior.append(b)
 original=[];replacement=[];items=[];excluded=[];seen=set();targetseen=set()
 for i,r in enumerate(rec['rows']):
  assert r['row_idx']==1700+i and donors[i]['row_idx']==1600+i and not r['truncated_cells'] and not donors[i]['truncated_cells'];b=r['row']['text'].encode();d=donors[i]['row']['text'].encode();h=sha(b);reason=None;target=b[385:513]
  if len(b)<513 or len(d)<256:reason='insufficient_length'
  elif h in priorhash or h in seen:reason='exact_document_duplicate'
  elif target in train or target in trainraw or any(target in x for x in prior):reason='exact_scored_target_in_prior_or_train'
  elif sha(target) in targetseen:reason='duplicate_current_target'
  seen.add(h)
  if reason:excluded.append({'row':1700+i,'reason':reason});continue
  targetseen.add(sha(target));original.append(list(b[:513]));replacement.append(list(d[:256]+b[256:513]));items.append({'recipient_row':1700+i,'donor_row':1600+i,'recipient_sha256':h,'donor_sha256':sha(d),'original_window_sha256':sha(b[:513]),'replacement_window_sha256':sha(d[:256]+b[256:513]),'target_sha256':sha(target)})
 assert len(items)>=40
 a=torch.tensor(original);b=torch.tensor(replacement);assert torch.equal(a[:,256:],b[:,256:]);assert torch.equal(a[:,-128:],b[:,-128:])
 receipt={'documents':len(items),'targets_per_model_condition':len(items)*128,'selected_pairs':items,'excluded':excluded,'training_raw_sha256':sha(trainraw),'actual_training_sha256':sha(train),'prior_unique_documents':len(priorhash),'prior_source_receipts':receipts,'recipient_response_sha256':sha(raw),'donor_response_sha256':sha(dr),'near_duplicate_check':False};path=P/'data_manifest.json'
 if path.exists():assert json.loads(path.read_text())==receipt,'Prior/source data changed; do not refreeze'
 else:path.write_text(json.dumps(receipt,indent=2))
 return {'original512':a,'donor512':b,'suffix256':a[:,256:]},receipt

def load(kind,seed,arm,opt=False):
 root=Path(os.environ.get('ANTI_V7_CHECKPOINT_DIR',str(P.parent/'anti_latency_v7')));path=root/f'{kind}_{seed}_{arm}.pt';h=sha(path.read_bytes());assert h==json.loads((P/'expected_checkpoint_hashes.json').read_text())[path.name];m=previous.model(kind);m.load_state_dict(torch.load(path,weights_only=True)['model'])
 if opt:
  for block in m.blocks:
   old=block.mix;new=Optimized(kind);new.load_state_dict(old.state_dict())
   if kind=='signed':
    with torch.no_grad():new.out.weight.neg_()
   block.mix=new
 return m.eval(),h

def main():
 start=time.perf_counter();data,receipt=prepare();rows=[];klrows=[]
 with torch.inference_mode():
  for kind in ['attention','signed']:
   for seed in [1,2,3]:
    for arm in ['long','mixed']:
     m,h=load(kind,seed,arm);logps={}
     for condition,x in data.items():
      allp=[]
      for batch in x.split(8):allp.append(m(batch[:,:-1])[:,-128:].double().log_softmax(-1))
      logp=torch.cat(allp);logps[condition]=logp;loss=-logp.gather(-1,x[:,-128:,None]).squeeze(-1)/math.log(2);per=loss.mean(1).tolist();rows.append(dict(kind=kind,seed=seed,arm=arm,condition=condition,bpb=statistics.mean(per),document_bpb=per,checkpoint_sha256=h))
     p,q=logps['original512'],logps['donor512'];kl=(p.exp()*(p-q)).sum(-1)/math.log(2);assert kl.min()>-1e-12;klrows.append(dict(kind=kind,seed=seed,arm=arm,mean_kl_bits=kl.mean().item(),document_kl_bits=kl.mean(1).tolist(),max_token_kl_bits=kl.max().item()))
     (P/'results.json').write_text(json.dumps(dict(rows=rows,kl=klrows,complete=len(rows)==36,elapsed_wall_seconds=time.perf_counter()-start),indent=2));print(kind,seed,arm,[round(r['bpb'],6) for r in rows[-3:]],flush=True)
 copy=[]
 for condition,x in data.items():
  per=[]
  for a in x.tolist():
   cache=CompactCache();loss=[]
   for byte in a[:-128]:cache.feed(byte)
   for target in a[-128:]:
    prob,n,k=cache.target_probability(target);loss.append(-math.log2(prob));cache.feed(target)
   per.append(statistics.mean(loss))
  copy.append(dict(condition=condition,bpb=statistics.mean(per),document_bpb=per))
 (P/'copy_results.json').write_text(json.dumps({'rows':copy,'orders':[4,8,16],'uniform_pseudocount_total':1,'neural_mixture':False},indent=2));print('COPY',[(r['condition'],r['bpb']) for r in copy],flush=True)
if __name__=='__main__':main()
