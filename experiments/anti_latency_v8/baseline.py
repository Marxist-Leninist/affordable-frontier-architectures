import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import torch, numpy as np, math,json,time,hashlib,random,statistics
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from base_models import LM,Attention,count
P=Path(__file__).resolve().parent
torch.set_num_threads(2)
class Mixer(Attention):
 def __init__(self,kind):
  super().__init__(); self.kind=kind
  if kind=='learned': self.g=nn.Parameter(torch.full((1,4,1,1),.25))
 def forward(self,x,state=None,cache=False):
  assert state is None and not cache
  b,t,d=x.shape;q,k,v=self.qkv(x).chunk(3,-1)
  q,k,v=[z.reshape(b,t,4,16).transpose(1,2) for z in (q,k,v)]
  q,k=self.rope(q,0),self.rope(k,0)
  a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
  if self.kind=='attention': z=a
  else:
   u=v.cumsum(-2)/torch.arange(1,t+1,dtype=v.dtype,device=v.device)[None,None,:,None]
   z=u-a if self.kind=='signed' else (1-2*self.g)*a+self.g*u
  return self.out(z.transpose(1,2).reshape(b,t,d)),None

def model(kind):
 m=LM('transformer')
 for b in m.blocks:b.mix=Mixer(kind)
 return m

def prepare():
 src=Path(os.environ.get('ANTI_DATA_DIR',str(P/'data')));man=json.loads((P/'source_data_manifest.json').read_text());streams={};receipt={}
 for split in ['train','validation','test']:
  raw=(src/f'{split}.txt').read_bytes();assert hashlib.sha256(raw).hexdigest()==man['splits'][split]['text_sha256']
  seen=set();clean=[]
  for line in raw.splitlines(keepends=True):
   k=line.strip()
   if k and (k in seen or (split!='train' and k in train_seen)):continue
   if k:seen.add(k)
   clean.append(line)
  if split=='train':train_seen=seen
  streams[split]=b''.join(clean)
 train=torch.tensor(list(streams['train'][:2000000]),dtype=torch.long);held={}
 for split,start in [('validation',32769),('test',98307)]:
  raw=streams[split][start:start+32769];chunks=[list(raw[i:i+129]) for i in range(0,len(raw)-128,128) if raw[i:i+129] not in streams['train']]
  held[split]=torch.tensor(chunks);receipt[split]={'offset':start,'sha256':hashlib.sha256(raw).hexdigest(),'tokens':len(chunks)*128}
 (P/'data.json').write_text(json.dumps(receipt,indent=2));return train,held
@torch.inference_mode()
def evaluate(m,data):
 m.eval();loss=0;n=0
 for batch in data.split(8):
  z=m(batch[:,:-1]);loss+=float(F.cross_entropy(z.flatten(0,1),batch[:,1:].flatten(),reduction='sum'));n+=batch.shape[0]*128
 return loss/n/math.log(2)
def main():
 train,held=prepare();results=[];start=time.perf_counter()
 for seed in [1,2,3]:
  starts=torch.randint(len(train)-129,(800,8),generator=torch.Generator().manual_seed(seed+1000))
  for lr in [.001,.003,.006]:
   for kind in ['attention','signed','learned']:
    torch.manual_seed(seed);m=model(kind);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.01);t=time.perf_counter()
    for step in range(800):
     ids=torch.stack([train[s:s+129] for s in starts[step]]);m.train();opt.zero_grad(set_to_none=True)
     rate=lr*min(1,(step+1)/40)*(.1+.9*(1+math.cos(math.pi*max(0,step-40)/760))/2)
     for gr in opt.param_groups:gr['lr']=rate
     loss=F.cross_entropy(m(ids[:,:-1]).flatten(0,1),ids[:,1:].flatten());loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
    r=dict(seed=seed,lr=lr,kind=kind,parameters=count(m),seconds=time.perf_counter()-t,validation=evaluate(m,held['validation']),gates=[b.mix.g.detach().flatten().tolist() for b in m.blocks] if kind=='learned' else None)
    path=P/f'{kind}_{seed}_{lr}.pt';torch.save(m.state_dict(),path);results.append(r);print(json.dumps(r),flush=True);(P/'tuning.json').write_text(json.dumps(results,indent=2))
 # choose a SINGLE lr per architecture on mean validation across seeds, no test looked at above
 chosen={kind:min([.001,.003,.006],key=lambda lr:statistics.mean(r['validation'] for r in results if r['kind']==kind and r['lr']==lr)) for kind in ['attention','signed','learned']}
 final=[];models={}
 for kind,lr in chosen.items():
  for seed in [1,2,3]:
   m=model(kind);m.load_state_dict(torch.load(P/f'{kind}_{seed}_{lr}.pt',weights_only=True));m.eval();final.append(dict(kind=kind,seed=seed,lr=lr,test=evaluate(m,held['test'])))
   if seed==1:models[kind]=m
 timing={};rng=random.Random(77)
 with torch.inference_mode():
  for context in [128,512]:
   x=torch.randint(256,(8,context));ts={k:[] for k in models}
   for m in models.values():
    for _ in range(10):m(x)
   for rep in range(50):
    order=list(models);rng.shuffle(order)
    for k in order:
     t=time.perf_counter();models[k](x);ts[k].append(time.perf_counter()-t)
   timing[context]=ts
 out=dict(chosen=chosen,test=final,timing=timing,wall_seconds=time.perf_counter()-start)
 (P/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps({'chosen':chosen,'test':final}),flush=True)
if __name__=='__main__':main()
