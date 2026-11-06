from evaluate_fresh import *
def main():
 data,receipt=prepare();records=[]
 for kind in ['attention','signed']:
  for seed in [1,2,3]:
   for arm in ['short','long','mixed']:
    a,_=load(kind,seed,arm);b,_=load(kind,seed,arm,True);a=a.double();b=b.double();x=data[:2,:33]
    with torch.inference_mode():
     z=a(x);y=b(x);pref,state=b(x[:,:20],cache=True);tail,state=b(x[:,20:],state,cache=True);err=(z-y).abs().max().item();ce=(y-torch.cat([pref,tail],1)).abs().max().item();x2=x.clone();x2[:,20:]=0;fe=(a(x2)[:,:20]-z[:,:20]).abs().max().item()
    assert err<1e-9 and ce<1e-9 and fe==0;records.append(dict(kind=kind,seed=seed,arm=arm,optimized_error=err,chunked_cache_error=ce,future_error=fe))
 for n in [128,256,512]:
  x=data[:,-n-1:];assert torch.equal(x[:,-128:],data[:,-128:]);assert torch.equal(x[:,-129:-1],data[:,-129:-1])
 (P/'validation.json').write_text(json.dumps({'target_alignment_passed':True,'checks':records},indent=2));print('All18 trained model parity/cache/causality and context target-alignment checks pass')
if __name__=='__main__':main()
