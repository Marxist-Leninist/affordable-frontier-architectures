from mixed_control import *
checks=[]
x=torch.arange(4*513).reshape(4,513)
assert torch.equal(torch.sort(split128(x)[:,1:].flatten()).values,torch.sort(x[:,1:].flatten()).values)
for step in [0,1]:
 a,b=(x[:2],x[2:]) if step%2==0 else (x[2:],x[:2]);assert torch.equal(torch.sort(torch.cat([a[:,1:].flatten(),split128(b)[:,1:].flatten()])).values,torch.sort(x[:,1:].flatten()).values)
prior=json.loads((P/'historical_long.json').read_text())
for kind in ['attention','signed']:
 for seed in [1,2,3]:
  initial=torch.load(P/f'initial_{kind}_{seed}.pt',weights_only=True);assert {v['step'].item() for v in initial['optimizer']['state'].values()}=={800.}
  for mode in ['short','long','mixed']:
   saved=torch.load(P/f'{kind}_{seed}_{mode}.pt',weights_only=True);assert {v['step'].item() for v in saved['optimizer']['state'].values()}=={1000.}
   a=previous.model(kind).double().eval();a.load_state_dict(saved['model']);b=create(kind,seed,True).double().eval();state={k:(-v if kind=='signed' and k.endswith('mix.out.weight') else v) for k,v in saved['model'].items()};b.load_state_dict(state);torch.manual_seed(70);x=torch.randint(256,(2,27))
   with torch.inference_mode():
    z=a(x);zz=b(x);pref,cache=b(x[:,:20],cache=True);tail,cache=b(x[:,20:],cache,cache=True);err=(z-zz).abs().max().item();ce=(zz-torch.cat([pref,tail],1)).abs().max().item()
   assert err<1e-9 and ce<1e-9;checks.append(dict(kind=kind,seed=seed,mode=mode,optimized_error=err,cache_error=ce))
rows=json.loads((P/'results.json').read_text())['rows'];long=[]
for r in rows:
 if r['mode']=='long':
  old=next(x for x in prior if x['kind']==r['kind'] and x['seed']==r['seed']);err=max(abs(v['bpb']-old['after'][m]['bpb']) for m,v in r['after'].items());assert err<1e-12;long.append(dict(kind=r['kind'],seed=r['seed'],historic_bpb_maxerror=err))
(P/'validation.json').write_text(json.dumps(dict(target_ledger_equality=True,checkpoints=checks,long_replications=long),indent=2));print('Target equality, 18 state/cache checks and 6 exact historical loss replays passed')
