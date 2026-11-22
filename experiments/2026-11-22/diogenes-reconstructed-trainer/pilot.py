"""CPU-only literal-original Diogenes checkpoint continuation, no original trainer/remote code."""
import argparse,json,hashlib,math,statistics,time
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from safetensors.torch import save_file
import pyarrow.parquet as pq
from model import load,functional_step_eval
REV='f39659c9805f098b581483e7bf46a529e905e303'
ORIG_SHA='5c274c53dbfbbc2f57ca34a8e8634c7e8ea56db9b48b0799a792c352b05f6190'
CONFIG_SHA='1420632883cec8b376fe9b183db44a8021c09e3614279371167b8ffc0e4f7e4c'
L=32; SEED=21197; N=16; LR=(3e-5,1e-4)
def sha(b):return hashlib.sha256(b).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
def tape(path,split,nn):
 m=json.loads(Path(__file__).with_name('wikitext.json').read_text())
 assert m['revision']=='b08601e04326c79dfdd32d625aee71d232d685c3'
 frozen=path/f'{split}.npy'
 if frozen.exists():
  meta=json.loads((path/f'{split}.json').read_text());arr=np.load(frozen,allow_pickle=False)
  assert len(arr)==nn and arr.dtype==np.int64 and arr.shape==(nn,L+1)
  assert np.all((0<=arr)&(arr<256)) and sha(arr.tobytes())==meta['sha'] and meta['n']==nn
  assert meta['parquet_sha256']==m['splits'][split]['parquet_sha256']
  return torch.from_numpy(arr),meta
 p=path/f'{split}.parquet';actual=file_sha(p);assert actual==m['splits'][split]['parquet_sha256']
 r=np.random.default_rng(SEED+{'train':0,'validation':1,'test':2}[split]);rows=[];xx=[]
 # Stream the Arrow batches from validated official HF Parquet, not a network fetch/execution.
 for batch in pq.ParquetFile(p).iter_batches(batch_size=64,columns=['text']):
  for s in batch.column(0).to_pylist():
   rows.append(None)
   b=s.encode('utf-8')
   if len(b)<L+1 or s.startswith(' ='):continue
   st=int(r.integers(0,len(b)-L))
   x=np.frombuffer(b[st:st+L+1],dtype='uint8').astype('int64')
   rows[-1]=[len(rows)-1,st,sha(x.tobytes())];xx.append(x)
   if len(xx)==nn:break
  if len(xx)==nn:break
 assert len(xx)==nn
 res=np.stack(xx)
 return torch.from_numpy(res),dict(parquet_sha256=actual,sha=sha(res.tobytes()),positions=[z for z in rows if z is not None],n=nn)
def write(path,v):Path(path).write_text(json.dumps(v,indent=2)+'\n')
@torch.no_grad()
def scores(m,x):
 m.eval();l=[];t=time.perf_counter()
 for a in x.split(4):
   y=m(a[:,:-1]);l.extend(cross_entropy(y.transpose(1,2),a[:,1:],reduction='none').mean(1).tolist())
 return float(statistics.mean(l)/math.log(2)),l,time.perf_counter()-t
def train(a):
 out=a.out;out.mkdir(exist_ok=False,mode=0o700)
 config=a.src/'config.json';orig=a.src/'model.safetensors'
 assert file_sha(config)==CONFIG_SHA and orig.stat().st_size==81192824 and file_sha(orig)==ORIG_SHA
 train,tm=tape(a.data,'train',N);dev,dm=tape(a.data,'validation',12)
 original,c=load(config,orig);original.eval()
 with torch.no_grad():
    x=dev[:2,:12]
    z=original(x);ind=functional_step_eval(original.state_dict(),x,c)
    report0={'initial_reference_max_abs':float((ind-z).abs().max()),'argmax_agreement':bool(torch.equal(ind.argmax(-1),z.argmax(-1))),
             'causal_prefix_max_abs':float((original(x[:,:7])-z[:,:7]).abs().max())}
    assert report0['initial_reference_max_abs']<5e-5 and report0['argmax_agreement'] and report0['causal_prefix_max_abs']<5e-5,report0
 print('ORIGINAL_PARITY',report0,flush=True)
 base,_l,be=scores(original,dev); del original
 runs=[]
 for v in LR:
    net,_=load(config,orig);opt=torch.optim.AdamW(net.parameters(),lr=v,weight_decay=.01)
    train_s=0.;trace=[];changed=[]
    for i,x in enumerate(train.split(1),1):
      net.train()
      before=time.perf_counter()
      opt.zero_grad(set_to_none=True)
      logits=net(x[:,:-1]);loss=cross_entropy(logits.transpose(1,2),x[:,1:])
      assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
      if i==1:assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
      opt.step();train_s+=time.perf_counter()-before
      if i in (8,N):
        val,nll,tm0=scores(net,dev);trace.append(dict(step=i,bpb=val,training_s=train_s,eval_s=tm0,grad_before_clip=float(grad)))
      if i%4==0:print(f'lr={v} step={i}/{N} nats={float(loss.detach()):.6f} elapsed={train_s:.3f}',flush=True)
    net0,_=load(config,orig)
    for k,v1 in net.state_dict().items():
      changed.append(k) if not torch.equal(net0.state_dict()[k],v1) else None
    assert changed and 'read.weight' in changed and 'eq.weight' in changed and 'merge.weight' in changed
    del net0
    # File names represent one exact measured run; no selection on test.
    target=out/f'continued-{v:.0e}.safetensors'
    save_file({k:x.detach().contiguous().cpu() for k,x in net.state_dict().items()},target,
              metadata={'format':'pt','base_sha':ORIG_SHA,'base_revision':REV,'objective':'causal_next_byte','updates':str(N)})
    del opt,net
    runs.append(dict(lr=v,name=target.name,sha=file_sha(target),bytes=target.stat().st_size,
         train_s=train_s,dev_bpb=trace[-1]['bpb'],dev_trace=trace,updated_tensors=len(changed)))
 winner=min(runs,key=lambda x:x['dev_bpb'])
 dev_frozen={'model':{'name':'Counfield/Diogenes-0.1-20M','revision':REV,'sha':ORIG_SHA,'license':'MIT'},
    'data':{'train':tm,'dev':dm,'revision':'b08601e04326c79dfdd32d625aee71d232d685c3','materialization':json.loads((a.data/'source.json').read_text())['materialization'] if (a.data/'source.json').exists() else 'stream records from verified immutable HF Parquet cache'},
    'baseline_dev_bpb':base,'baseline_dev_time_s':be,'original_reference_parity':report0,'runs':runs,
    'decided_before_test':winner['name'],'steps':N,'target_bytes':N*L,'torch':torch.__version__}
 write(out/'dev_frozen.json',dev_frozen)
 held(a,False)
def held(a,verify):
 out=a.out;f=out/'dev_frozen.json';d=json.loads(f.read_text())
 assert d['model']['sha']==ORIG_SHA
 t,meta=tape(a.data,'test',24)
 if verify:
   old=json.loads((out/'test.json').read_text()); assert meta==old['fixture'] and old['dev_seal']==file_sha(f)
 else:
   old={}
 rows=[]
 paths=[(a.src/'model.safetensors','original',ORIG_SHA)]+[(out/r['name'],r['name'],r['sha']) for r in d['runs']]
 for p,name,exp in paths:
    assert file_sha(p)==exp
    m,c=load(a.src/'config.json',p)
    acc,nll,st=scores(m,t)
    x=t[:2,:8]
    with torch.no_grad(): ref=m(x);parity=float((ref-functional_step_eval(m.state_dict(),x,c)).abs().max())
    assert parity<5e-5
    row=dict(model=name,bpb=acc,window_nats=nll,eval_s=st,model_sha=exp,reference_replay_max_abs=parity)
    if verify:
      expected=old['rows'][len(rows)]
      assert expected['bpb']==acc and expected['window_nats']==nll and expected['model_sha']==exp
    print(('REPLAY' if verify else 'TEST'),name,acc,st,parity,flush=True);rows.append(row)
 if verify:
   write(out/'fresh_replay.json',{'exact':True,'test_windows_checked':3*len(t),'dev_seal':file_sha(f),'changed_checkpoints':[r['model_sha'] for r in rows[1:]]})
 else:
   base=rows[0]
   for r in rows:r['index']=100*2**(.9*(base['bpb']-r['bpb']))*(base['eval_s']/r['eval_s'])**.1
   write(out/'test.json',dict(data_scope='fresh for continuation only; original pretraining overlap unknown',fixture=meta,rows=rows,
       rank=[r['model'] for r in sorted(rows,key=lambda o:-o['index'])],
       index_formula='100*2^[.9*(original BPB – arm BPB)]*(original CPU eval time / arm CPU eval time)^.1; CPU time is small and noisy',
       dev_seal=file_sha(f)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('src',type=Path);p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('--replay',action='store_true');a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(SEED)
 if a.replay:held(a,True)
 else:train(a)
