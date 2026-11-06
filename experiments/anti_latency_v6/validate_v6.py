from train import *
checks=[]
for kind in ['attention','signed']:
 for seed in [2,3]:
  for mode in ['reset','retain']:
   saved=torch.load(P/f'restart_{kind}_{seed}_{mode}.pt',weights_only=True);a=previous.model(kind);a.load_state_dict(saved['model']);b=create(kind,seed,True);state={k:(-v if kind=='signed' and k.endswith('mix.out.weight') else v) for k,v in saved['model'].items()};b.load_state_dict(state);a=a.double().eval();b=b.double().eval();torch.manual_seed(39);x=torch.randint(256,(2,23))
   with torch.inference_mode():
    z=a(x);y=b(x);pref,cache=b(x[:,:17],cache=True);tails=[]
    for i in range(17,23):
     zz,cache=b(x[:,i:i+1],cache,cache=True);tails.append(zz)
    err=(z-y).abs().max().item();ce=(torch.cat([pref]+tails,1)-y).abs().max().item()
   steps={v['step'].item() for v in saved['optimizer']['state'].values()};assert steps==({1000.} if mode=='retain' else {200.});assert err<1e-9 and ce<1e-9
   checks.append(dict(kind=kind,seed=seed,mode=mode,optimized_maxerror=err,cached_maxerror=ce,optimizer_steps=list(steps)))
(P/'validation_v6.json').write_text(json.dumps(checks,indent=2));print('All eight checkpoint/optimizer/cache checks passed')
