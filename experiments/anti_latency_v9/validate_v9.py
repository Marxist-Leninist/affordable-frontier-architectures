from profile_v9 import *
def main():
 data,receipt=prepare();records=[]
 for kind in ['attention','signed']:
  for seed in [1,2,3]:
   for arm in ['long','mixed']:
    original,_=load(kind,seed,arm);optimized,_=load(kind,seed,arm,True);original=original.double();optimized=optimized.double()
    for condition,allx in data.items():
     x=allx[:2,:-1]
     with torch.inference_mode():
      z=original(x);y=optimized(x);tail=tail_logits(optimized,x);cut=x.shape[1]//2;p,cache=optimized(x[:,:cut],cache=True);q,cache=optimized(x[:,cut:],cache,cache=True);err=(z-y).abs().max().item();te=(tail-y[:,-128:]).abs().max().item();ce=(y-torch.cat([p,q],1)).abs().max().item();altered=x.clone();altered[:,cut:]=0;fe=(original(altered)[:,:cut]-z[:,:cut]).abs().max().item()
     assert max(err,te,ce)<1e-9 and fe==0;records.append(dict(kind=kind,seed=seed,arm=arm,condition=condition,optimized_error=err,tailhead_error=te,cache_error=ce,future_error=fe))
 oracle_checks=0
 for condition,allx in data.items():
  for seq in allx[:3].tolist():
   cache=CompactCache();history=seq[:-128].copy()
   for b in history:cache.feed(b)
   for target in seq[-128:]:
    expected=np.full(256,1/256)
    for order in [16,8,4]:
     key=history[-order:];counts={}
     for j in range(order,len(history)):
      if history[j-order:j]==key:counts[history[j]]=counts.get(history[j],0)+1
     if counts:
      for b,n in counts.items():expected[b]+=n
      expected/=sum(counts.values())+1;break
    actual,_,_=cache.distribution();prob,_,_=cache.target_probability(target);assert np.allclose(actual,expected,atol=1e-14,rtol=0) and abs(actual.sum()-1)<1e-12 and abs(prob-actual[target])<1e-14
    history.append(target);cache.feed(target);oracle_checks+=1
 assert torch.equal(data['original512'][:,256:],data['donor512'][:,256:]);assert torch.equal(data['suffix256'],data['original512'][:,256:])
 (P/'validation.json').write_text(json.dumps({'neural_checks':records,'copy_distribution_oracle_checks':oracle_checks,'intervention_target_alignment':True},indent=2));print('36 full-length neural parity/cache/causal/tailhead checks and',oracle_checks,'copy oracle checks pass')
if __name__=='__main__':main()
