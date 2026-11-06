"""Bounded training focus: exact implementation profiling and matched 512-context continuation."""
from run import *
import hashlib,numpy as np

def create(kind,seed,opt):
 m=make(kind,'original',seed)
 if opt:
  for block in m.blocks:
   old=block.mix;new=Optimized(kind);new.load_state_dict(old.state_dict())
   if kind=='signed':
    with torch.no_grad():new.out.weight.neg_()
   block.mix=new
 return m

def profile():
 result={};rng=random.Random(71)
 for n in [128,512]:
  torch.manual_seed(1);x=torch.randint(256,(4,n+1));ms={f'{k}_{o}':create(k,1,o).train() for k in ['attention','signed'] for o in [False,True]};opts={k:torch.optim.AdamW(m.parameters(),lr=.001) for k,m in ms.items()};raw={k:[] for k in ms}
  def step(k):
   t=time.perf_counter();opts[k].zero_grad(set_to_none=True);loss=F.cross_entropy(ms[k](x[:,:-1]).flatten(0,1),x[:,1:].flatten());loss.backward();torch.nn.utils.clip_grad_norm_(ms[k].parameters(),1);opts[k].step();return time.perf_counter()-t
  for k in ms:
   for _ in range(5):step(k)
  for rep in range(41):
   keys=list(ms);rng.shuffle(keys)
   for k in keys:raw[k].append(step(k))
  result[n]={'seconds':raw,'median_seconds':{k:statistics.median(v) for k,v in raw.items()},'note':'full training step incl loss, backward, clipping, AdamW; random tokens only for timing, not quality'}
 return result

def datasets():
 src=Path(os.environ.get('ANTI_DATA_DIR',str(P/'data')));os.environ['ANTI_DATA_DIR']=str(src);train,_=previous.prepare();streams={};seen_train=set()
 for split in ['train','test']:
  raw=(src/f'{split}.txt').read_bytes();seen=set();clean=[]
  for line in raw.splitlines(keepends=True):
   k=line.strip()
   if k and (k in seen or (split!='train' and k in seen_train)):continue
   if k:seen.add(k)
   clean.append(line)
  if split=='train':seen_train=seen
  streams[split]=b''.join(clean)
 raw=streams['test'][196608:196608+65537];chunks=[list(raw[i:i+513]) for i in range(0,len(raw)-512,512) if raw[i:i+513] not in streams['train']];wiki=torch.tensor(chunks)
 mf=json.loads((P/'fineweb/data_manifest.json').read_text());path=Path(os.environ.get('ANTI_FINEWEB_ROWS',str(P/'fineweb/rows.json')));assert hashlib.sha256(path.read_bytes()).hexdigest()==mf['response_sha256'];chunks=[]
 for r,d in zip(json.loads(path.read_text())['rows'],mf['documents']):
  raw=r['row']['text'].encode()[:4097];assert hashlib.sha256(raw).hexdigest()==d['capped_text_sha256']
  for start in range(0,len(raw)-512,512):
   block=raw[start:start+513]
   if block not in streams['train']:chunks.append(list(block))
 fine=torch.tensor(chunks[:64]);receipt={'wiki_offset':196608,'wiki_windows':len(wiki),'fineweb_windows':len(fine),'fineweb_selection':'First64 eligible windows in frozen v31 source; no tuning/selection on losses','window_bytes':513,'targets_per_window':128,'train_bytes':len(train),'train_sha256':hashlib.sha256(bytes(train.tolist())).hexdigest(),'datasets':{k:hashlib.sha256(bytes(v.flatten().tolist())).hexdigest() for k,v in [('wiki',wiki),('fineweb',fine)]}}
 return train,{'wiki':wiki,'fineweb':fine},receipt

@torch.inference_mode()
def evaluate(m,held):
 m.eval();out={}
 for name,data in held.items():
  for n in [128,512]:
   losses=[]
   for x in data.split(8):
    z=m(x[:,-n-1:-1])[:,-128:];y=x[:,-128:];loss=-z.double().log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)/math.log(2);losses.extend(loss.mean(1).tolist())
   out[f'{name}_{n}']={'bpb':statistics.mean(losses),'window_bpb':losses}
 return out

def main():
 t=time.perf_counter();profile_results=json.loads((P/'training_profile.json').read_text()) if (P/'training_profile.json').exists() else profile();(P/'training_profile.json').write_text(json.dumps(profile_results,indent=2));print('PROFILE', {n:r['median_seconds'] for n,r in profile_results.items()},flush=True)
 train,held,receipt=datasets();(P/'training_data.json').write_text(json.dumps(receipt,indent=2));rows=[]
 for seed in [1,2,3]:
  starts=torch.randint(len(train)-513,(200,4),generator=torch.Generator().manual_seed(3000+seed))
  for kind in ['attention','signed']:
   m=create(kind,seed,True);before=evaluate(m,held);m.train();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);records=[];elapsed=0
   for step in range(200):
    # Batch assembly excluded from optimizer-step compute, included in run elapsed.
    x=torch.stack([train[s:s+513] for s in starts[step]]);lr=.001*(.1+.9*(1+math.cos(math.pi*step/199))/2)
    for group in opt.param_groups:group['lr']=lr
    t0=time.perf_counter();opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x[:,:-1]).flatten(0,1),x[:,1:].flatten());loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();dt=time.perf_counter()-t0;elapsed+=dt
    assert torch.isfinite(loss) and torch.isfinite(gn)
    records.append({'step':step+1,'nll':loss.item(),'gradient_norm_before_clip':gn.item(),'seconds':dt,'lr':lr})
   after=evaluate(m,held);path=P/f'continued_{kind}_{seed}.pt';torch.save(m.state_dict(),path)
   row={'kind':kind,'seed':seed,'steps':200,'batch':4,'context':512,'tokens':409600,'optimizer_step_seconds':elapsed,'training_tokens_per_second':409600/elapsed,'before':before,'after':after,'history':records,'checkpoint':path.name,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};rows.append(row)
   (P/'training_results.json').write_text(json.dumps({'rows':rows,'elapsed_wall_seconds':time.perf_counter()-t,'complete':len(rows)==6},indent=2));print(kind,seed,elapsed,{k:v['bpb'] for k,v in after.items()},flush=True)
if __name__=='__main__':main()
