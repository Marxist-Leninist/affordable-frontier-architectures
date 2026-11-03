"""Bounded checks only: no optimizer training run is started here."""
import json
import numpy as np
from models import Transformer,Recurrent
from run import P,RD,TARGET,batches,evaluation_data,make_batch
from test_base_models import run_tests

def tests():
    report={'base_models':run_tests()}; rng=np.random.default_rng(941)
    for cls in (Transformer,Recurrent):
        m=cls(17,8,31);x,y=make_batch(rng,5,3);loss,g=m.loss(x,y);errors={}
        for key,value in m.p.items():
            errors[key]=0.
            for flat in rng.choice(value.size,min(10,value.size),replace=False):
                ix=np.unravel_index(flat,value.shape);old=value[ix];eps=1e-5
                value[ix]=old+eps;hi=m.loss(x,y,False)
                value[ix]=old-eps;lo=m.loss(x,y,False);value[ix]=old
                error=abs((hi-lo)/(2*eps)-g[key][ix]);errors[key]=max(errors[key],error);assert error<2e-7,(key,error)
        # Verify exact zero direct loss gradient before the query, analytic final CE.
        out=rng.normal(size=(3,5,8));direct,dl=m.output_loss(out,y)
        assert np.count_nonzero(dl[:,:-1])==0
        probs=m.output_loss(out,None)
        np.testing.assert_allclose(direct,-np.log(probs[np.arange(3),-1,y[:,-1]]).mean(),atol=1e-13)
        expected=probs[:,-1].copy();expected[np.arange(3),y[:,-1]]-=1;expected/=3
        np.testing.assert_allclose(dl[:,-1],expected,atol=1e-13)
        # Every sequence contributes equally regardless of masked prefix length.
        separate=[m.loss(x[i:i+1],y[i:i+1]) for i in range(len(x))]
        np.testing.assert_allclose(loss,np.mean([s[0] for s in separate]),atol=1e-13)
        for key in g:np.testing.assert_allclose(g[key],np.mean([s[1][key] for s in separate],axis=0),atol=1e-13)
        try:m.loss(x,np.full_like(y,-1));raise AssertionError('Empty mask accepted')
        except ValueError:pass
        report[cls.__name__]={'masked_gradient_max_absolute_errors':errors,'nonquery_direct_gradient_exact_zero':True,'query_only_ce':True,'batch_gradient_invariance':True,'empty_mask_rejected':True}
    data_hashes={}
    for split in ('validation','test'):
        for length in P['evaluation_lengths']:
            x,y=evaluation_data(length,split)
            assert x.shape==(1024,length)
            assert np.array_equal(np.bincount(y[:,-1],minlength=8),np.full(8,128))
            assert np.array_equal(x[:,0],y[:,-1]);assert (x[:,-1]==16).all();assert (y[:,:-1]==-1).all()
            assert ((x[:,1:-1]>=8)&(x[:,1:-1]<=15)).all()
            import hashlib
            data_hashes[f'{split}_{length}']=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
    assert len(set(data_hashes.values()))==len(data_hashes)
    schedules={}
    for seed in [101,1,2,3]:
        aa=list(batches(seed));bb=list(batches(seed))
        assert len(aa)==400
        assert all(np.array_equal(x,xx) and np.array_equal(y,yy) for (x,y),(xx,yy) in zip(aa,bb))
        counts={str(l):sum(len(x) for x,y in aa if x.shape[1]==l) for l in P['train_lengths']}
        tokens=sum(x.size for x,y in aa);queries=sum((y>=0).sum() for x,y in aa)
        assert tokens==192000 and queries==6400 and set(counts.values())=={1600}
        schedules[str(seed)]={'input_tokens':tokens,'queries':int(queries),'queries_by_length':counts}
    report['data']={'balanced_evaluation':True,'split_hashes':data_hashes,'paired_deterministic_train_schedules':schedules}
    report['parameters']={'transformer_width':32,'transformer_count':TARGET,'recurrent_width':RD,'recurrent_count':Recurrent(17,RD,1).count()}
    return report
if __name__=='__main__':print(json.dumps(tests(),indent=2))
