"""Predeclared extension of v9: same weights, 1024 balanced queries at 2k/4k.
Linear-memory final-query Transformer evaluation is exact for its ONE block.
No retuning. Original protocol/results are immutable inputs.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import json,time,hashlib
from pathlib import Path
from models import Transformer,Recurrent,positions,layer_norm,softmax
from run import train,evaluation_data
ROOT=Path(__file__).resolve().parent

def final_logits(m,x):
    p=m.p;b,t=x.shape;d=m.d
    if isinstance(m,Recurrent):
        state=np.zeros((b,d))
        for i in range(t):state=np.tanh(p['E'][x[:,i]]+state@p['R']+p['b'])
        return state@p['U']
    def norm(z,name):return layer_norm(z)[0]*p[name+'s']+p[name+'b']
    # Absolute indices0..t-1 exactly as reference. Only final query attends;
    # every key is causal relative to that query, so no keys are masked.
    z=p['E'][x]+positions(t,d)
    a=norm(z,'n1');h=m.heads;kdim=d//h
    q=(a[:,-1:]@p['Q']).reshape(b,1,h,kdim).transpose(0,2,1,3)
    k=(a@p['K']).reshape(b,t,h,kdim).transpose(0,2,1,3)
    v=(a@p['V']).reshape(b,t,h,kdim).transpose(0,2,1,3)
    scores=q@k.swapaxes(-1,-2)/np.sqrt(kdim)
    c=(softmax(scores)@v).transpose(0,2,1,3).reshape(b,d)
    residual=z[:,-1]+c@p['O']
    f=np.tanh(norm(residual,'n2')@p['F']+p['fb'])
    out=norm(residual+f@p['G']+p['gb'],'nf')
    return out@p['U']

def parity(m):
    report={}
    for length in (8,64,128,256):
        x,y=evaluation_data(length,'test');x=x[:3]
        fast=softmax(final_logits(m,x));full=m.loss(x,None)[:,-1]
        err=float(np.max(np.abs(fast-full)));np.testing.assert_allclose(fast,full,atol=2e-12,rtol=2e-12)
        logits=final_logits(m,x);mx=logits.max(-1);ce=np.mean(mx+np.log(np.exp(logits-mx[:,None]).sum(-1))-logits[np.arange(len(x)),y[:3,-1]])
        np.testing.assert_allclose(ce,m.loss(x,y[:3],False),atol=2e-12,rtol=2e-12)
        report[str(length)]={'probability_max_absolute_error':err,'query_ce_agrees':True}
    return report

def evaluate_long(m,length):
    x,y=evaluation_data(length,'test');correct=0;nats=0.;start=time.perf_counter();batchsize=16
    for startidx in range(0,len(x),batchsize):
        xx=x[startidx:startidx+batchsize];target=y[startidx:startidx+batchsize,-1]
        logits=final_logits(m,xx);mx=logits.max(-1)
        nats+=float(np.sum(mx+np.log(np.exp(logits-mx[:,None]).sum(-1))-logits[np.arange(len(xx)),target]))
        correct+=int((logits.argmax(-1)==target).sum())
    # Named estimates, not measured process RSS or hardware speed benchmarks.
    # Transformer keeps z,a,k,v each B*T*D; norm intermediates/BLAS add memory.
    caches=4*batchsize*length*m.d*8+2*batchsize*m.heads*length*8 if isinstance(m,Transformer) else 2*batchsize*m.d*8
    return {'length_symbols':length,'queries':len(x),'target_counts':np.bincount(y[:,-1],minlength=8).tolist(),'correct':correct,'query_accuracy':correct/len(x),'query_ce_nats':nats/len(x),'evaluation_wall_seconds':time.perf_counter()-start,'batch_size':batchsize,'data_arrays_bytes':x.nbytes+y.nbytes,'named_model_cache_bytes_estimate':caches,'cache_estimate_note':'Named main arrays only, excludes temporary normalization/BLAS allocations, parameters, Python overhead and process RSS. No speed comparison.','input_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),'extrapolation':True,'target_class_chance':.125}

def main():
    old=json.loads((ROOT/'results.json').read_text());tests={'random':{},'trained':{}};rows=[]
    for cls in (Transformer,Recurrent):tests['random'][cls.__name__]=parity(cls(17,32 if cls is Transformer else 101,79))
    for original in old['final']:
        kind=original['kind'];seed=original['seed'];lr=old['selected_rates'][kind]
        m,row=train(kind,seed,lr)
        sha=hashlib.sha256(b''.join(v.tobytes() for v in m.p.values())).hexdigest()
        assert sha==original['weights_sha256'],('Weight mismatch',kind,seed,sha,original['weights_sha256'])
        tests['trained'][f'{kind}_{seed}']=parity(m)
        item={'kind':kind,'seed':seed,'learning_rate':lr,'weights_sha256':sha,'identical_original_weights':True,'parameters':m.count(),'reconstruction_training_seconds':row['training_seconds'],'evaluations':{}}
        for length in (2048,4096):
            item['evaluations'][str(length)]=evaluate_long(m,length)
            print('evaluation',json.dumps({'kind':kind,'seed':seed,**item['evaluations'][str(length)]}),flush=True)
        rows.append(item)
        (ROOT/'extrapolation_2k4k_partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    result={'scope':'Same v9 trained weights, no retuning;3-bit single-symbol recall, not long-text memory. Unseen sinusoidal positions/context lengths may confound Transformer extrapolation. Length units are vocabulary symbols, not bytes.','original_results_sha256':hashlib.sha256((ROOT/'results.json').read_bytes()).hexdigest(),'extension_protocol_sha256':hashlib.sha256((ROOT/'extrapolation_2k4k_protocol.json').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,'summary':{kind:{str(l):{metric:float(np.mean([r['evaluations'][str(l)][metric] for r in rows if r['kind']==kind])) for metric in ('query_accuracy','query_ce_nats')} for l in (2048,4096)} for kind in ('transformer','recurrent')}}
    (ROOT/'extrapolation_2k4k_tests.json').write_text(json.dumps(tests,indent=2)+'\n')
    (ROOT/'extrapolation_2k4k_results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
