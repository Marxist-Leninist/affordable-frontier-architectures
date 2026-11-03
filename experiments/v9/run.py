import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import json, hashlib, time, platform
from pathlib import Path
import numpy as np
from models import Transformer, Recurrent
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
V=17
TARGET=Transformer(V,32,1).count()
RD=min(range(1,512),key=lambda d:abs(Recurrent(V,d,1).count()-TARGET))
def make_batch(rng,length,n=16,balanced=False):
    if balanced:
        assert n%8==0
        target=np.tile(np.arange(8),n//8); rng.shuffle(target)
    else: target=rng.integers(8,size=n)
    x=rng.integers(8,16,size=(n,length)); x[:,0]=target; x[:,-1]=16
    y=np.full(x.shape,-1,dtype=np.int64); y[:,-1]=target
    return x,y

def batches(seed):
    rng=np.random.default_rng(1000+seed)
    for block in range(P['steps']//4):
        for length in rng.permutation(P['train_lengths']):
            yield make_batch(rng,int(length),P['batch_size'])

def evaluation_data(length,split):
    return make_batch(np.random.default_rng((200000 if split=='validation' else 300000)+length),length,P['evaluation_cases_per_length'],True)

def evaluate(m,split,lengths):
    result={}
    for length in lengths:
        x,y=evaluation_data(length,split); correct=0; nats=0.
        for start in range(0,len(x),16):
            xx,yy=x[start:start+16],y[start:start+16]
            # Separate stable CE and probabilities, to avoid log(underflow).
            nats+=m.loss(xx,yy,False)*len(xx)
            prob=m.loss(xx,None)[:,-1]
            correct+=int((prob.argmax(-1)==yy[:,-1]).sum())
        result[str(length)]={'query_accuracy':correct/len(x),'query_ce_nats':nats/len(x),'correct':correct,'queries':len(x),'input_tokens':int(x.size),'extrapolation':length>64,'target_class_chance':.125}
    return result

def train(kind,seed,lr):
    m=Transformer(V,32,seed) if kind=='transformer' else Recurrent(V,RD,seed)
    ms={k:np.zeros_like(v) for k,v in m.p.items()};vs={k:np.zeros_like(v) for k,v in m.p.items()}
    trace=[];input_tokens=0;queries=0;digest=hashlib.sha256();length_counts={str(k):0 for k in P['train_lengths']}; start=time.perf_counter()
    for step,(x,y) in enumerate(batches(seed),1):
        digest.update(x.tobytes());digest.update(y.tobytes());input_tokens+=x.size;queries+=len(x);length_counts[str(x.shape[1])]+=len(x)
        loss,g=m.loss(x,y);norm=np.sqrt(sum(np.sum(v*v) for v in g.values()));scale=min(1.,1/(norm+1e-12))
        for k in m.p:
            q=g[k]*scale;ms[k]=.9*ms[k]+.1*q;vs[k]=.999*vs[k]+.001*q*q
            m.p[k]-=lr*(ms[k]/(1-.9**step))/(np.sqrt(vs[k]/(1-.999**step))+1e-8)
        if step%100==0:trace.append({'step':step,'batch_query_ce_nats':loss})
    seconds=time.perf_counter()-start
    val=evaluate(m,'validation',P['train_lengths'])
    row={'kind':kind,'seed':seed,'learning_rate':lr,'parameters':m.count(),'width':m.d,'training_input_tokens':input_tokens,'supervised_queries':queries,'train_queries_by_length':length_counts,'training_batches_sha256':digest.hexdigest(),'training_seconds':seconds,'trace':trace,'validation':val,'validation_mean_query_ce_nats':float(np.mean([v['query_ce_nats'] for v in val.values()]))}
    return m,row

def main():
    tuning=[];chosen={}
    for kind in ('transformer','recurrent'):
        for lr in P['learning_rates']:
            _,row=train(kind,P['tuning_seed'],lr);tuning.append(row);print('tuning',json.dumps(row),flush=True)
        chosen[kind]=min((r for r in tuning if r['kind']==kind),key=lambda r:r['validation_mean_query_ce_nats'])['learning_rate']
    (ROOT/'selection.json').write_text(json.dumps({'selected_rates':chosen,'tuning':tuning},indent=2)+'\n')
    rows=[]
    for seed in P['final_seeds']:
        for kind in ('transformer','recurrent'):
            m,row=train(kind,seed,chosen[kind]);row['test']=evaluate(m,'test',P['evaluation_lengths']);row['weights_sha256']=hashlib.sha256(b''.join(v.tobytes() for v in m.p.values())).hexdigest();rows.append(row);print('final',json.dumps(row),flush=True)
    summary={kind:{str(length):{metric:float(np.mean([r['test'][str(length)][metric] for r in rows if r['kind']==kind])) for metric in ('query_accuracy','query_ce_nats')} for length in P['evaluation_lengths']} for kind in chosen}
    result={'protocol':P,'recurrent_width':RD,'parameters':{'transformer':TARGET,'recurrent':Recurrent(V,RD,1).count()},'numpy':np.__version__,'platform':platform.platform(),'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('models.py','run.py','protocol.json','test_models.py')},'selected_rates':chosen,'tuning':tuning,'final':rows,'summary':summary}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
