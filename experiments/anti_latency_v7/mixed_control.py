"""Matched-target retained-Adam short/long/mixed context continuation, three seeds."""
from train import *
import copy

def split128(x):
 return torch.cat([x[:,i:i+129] for i in [0,128,256,384]])
def loss_for(m,x,mode,step):
 if mode=='short':parts=[split128(x)]
 elif mode=='long':parts=[x]
 else:
  # Alternate which spans see long context; every update is 50/50 target-weighted.
  a,b=(x[:2],x[2:]) if step%2==0 else (x[2:],x[:2]);parts=[a,split128(b)]
 losses=[F.cross_entropy(m(z[:,:-1]).flatten(0,1),z[:,1:].flatten()) for z in parts]
 return sum(losses)/len(losses)
def update(m,opt,x,mode,step):
 opt.zero_grad(set_to_none=True);loss=loss_for(m,x,mode,step);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();assert torch.isfinite(loss) and torch.isfinite(gn);return loss.item()
def main():
 train,held,receipt=datasets();(P/'data_receipt.json').write_text(json.dumps(receipt,indent=2));rows=[];t0=time.perf_counter()
 for seed in [1,2,3]:
  for kind in ['attention','signed']:
   torch.manual_seed(seed);m=previous.model(kind).train();opt=torch.optim.AdamW(m.parameters(),lr=.006,weight_decay=.01);starts=torch.randint(len(train)-129,(800,8),generator=torch.Generator().manual_seed(seed+1000));t=time.perf_counter()
   for step in range(800):
    x=torch.stack([train[s:s+129] for s in starts[step]]);lr=.006*min(1,(step+1)/40)*(.1+.9*(1+math.cos(math.pi*max(0,step-40)/760))/2)
    for g in opt.param_groups:g['lr']=lr
    update(m,opt,x,'long',step)
   seconds=time.perf_counter()-t;expected=torch.load(SRC/f'{kind}_{seed}_0.006.pt',weights_only=True);err=max((p-expected[n]).abs().max().item() for n,p in m.state_dict().items());assert err==0
   initial={'model':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(opt.state_dict())};torch.save(initial,P/f'initial_{kind}_{seed}.pt');starts=torch.randint(len(train)-513,(200,4),generator=torch.Generator().manual_seed(3000+seed))
   modes=['short','long','mixed'];random.Random(seed+99).shuffle(modes)
   for mode in modes:
    a=previous.model(kind);a.load_state_dict(initial['model']);a.train();o=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.01);o.load_state_dict(copy.deepcopy(initial['optimizer']));times=[];history=[]
    for step in range(200):
     x=torch.stack([train[s:s+513] for s in starts[step]]);lr=.001*(.1+.9*(1+math.cos(math.pi*step/199))/2)
     for g in o.param_groups:g['lr']=lr
     t=time.perf_counter();loss=update(a,o,x,mode,step);times.append(time.perf_counter()-t);history.append(loss)
    after=evaluate(a,held);path=P/f'{kind}_{seed}_{mode}.pt';torch.save({'model':a.state_dict(),'optimizer':o.state_dict()},path)
    row=dict(seed=seed,kind=kind,mode=mode,after=after,reconstruction_seconds=seconds,reconstruction_error=err,steps=200,target_bytes=409600,history=history,step_seconds=times,checkpoint=path.name);rows.append(row)
    (P/'results.json').write_text(json.dumps(dict(rows=rows,complete=len(rows)==18,elapsed_wall_seconds=time.perf_counter()-t0),indent=2));print(seed,kind,mode,{k:round(v['bpb'],6) for k,v in after.items()},flush=True)
if __name__=='__main__':main()
