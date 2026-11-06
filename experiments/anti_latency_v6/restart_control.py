"""Seeds2/3 reconstruction for matched restart vs retained optimizer control."""
from train import *
import copy

def optimize(m,opt,x):
 opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x[:,:-1]).flatten(0,1),x[:,1:].flatten());loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();assert torch.isfinite(loss) and torch.isfinite(gn);return loss.item()
def main():
 train,held,_=datasets();results=[]
 for seed,kind in [(s,k) for s in [2,3] for k in ['attention','signed']]:
  torch.manual_seed(seed);m=previous.model(kind).train();opt=torch.optim.AdamW(m.parameters(),lr=.006,weight_decay=.01)
  starts=torch.randint(len(train)-129,(800,8),generator=torch.Generator().manual_seed(seed+1000));t=time.perf_counter()
  for step in range(800):
   x=torch.stack([train[s:s+129] for s in starts[step]])
   rate=.006*min(1,(step+1)/40)*(.1+.9*(1+math.cos(math.pi*max(0,step-40)/760))/2)
   for group in opt.param_groups:group['lr']=rate
   optimize(m,opt,x)
  seconds=time.perf_counter()-t;expected=torch.load(SRC/f'{kind}_{seed}_0.006.pt',weights_only=True);error=max((p-expected[n]).abs().max().item() for n,p in m.state_dict().items());state=copy.deepcopy(m.state_dict());optimizer_state=copy.deepcopy(opt.state_dict());before=evaluate(m,held);print('reconstructed',seed,kind,seconds,error,flush=True)
  # Comparing reset/retained starts from exactly identical reconstructed parameters, even if historic replay differs.
  starts=torch.randint(len(train)-513,(200,4),generator=torch.Generator().manual_seed(3000+seed))
  for mode in ['reset','retain']:
   a=previous.model(kind);a.load_state_dict(state);a.train();o=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.01)
   if mode=='retain':o.load_state_dict(optimizer_state)
   ts=0;losses=[]
   for step in range(200):
    x=torch.stack([train[s:s+513] for s in starts[step]]);lr=.001*(.1+.9*(1+math.cos(math.pi*step/199))/2)
    for group in o.param_groups:group['lr']=lr
    t=time.perf_counter();loss=optimize(a,o,x);ts+=time.perf_counter()-t;losses.append(loss)
   after=evaluate(a,held);path=P/f'restart_{kind}_{seed}_{mode}.pt';torch.save({'model':a.state_dict(),'optimizer':o.state_dict(),'seed':seed,'kind':kind,'mode':mode},path)
   results.append({'kind':kind,'seed':seed,'mode':mode,'reconstruction_seconds':seconds,'historic_weight_maxerror':error,'before':before,'after':after,'steps':200,'context':512,'batch':4,'tokens':409600,'seconds':ts,'history':losses,'checkpoint':path.name,'checkpoint_convention':'original U-A; no projection sign reversal'})
   (P/'restart_results.json').write_text(json.dumps({'rows':results,'complete':len(results)==8},indent=2));print(kind,mode,{k:v['bpb'] for k,v in after.items()},flush=True)
if __name__=='__main__':main()
