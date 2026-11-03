import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import json,hashlib,time,platform
from pathlib import Path
import numpy as np
from models import Transformer, Recurrent
P=json.loads(Path('protocol.json').read_text())
raw={split:Path(split+'.txt').read_bytes() for split in ('train','validation','test')}
data={key:np.frombuffer(value,dtype=np.uint8).astype(np.int64) for key,value in raw.items()}
V=256
def count(m):return sum(p.size for p in m.p.values())
target=count(Transformer(V,32,1)); rd=min(range(4,160),key=lambda d:abs(count(Recurrent(V,d,1))-target))
def evaluate(m,split):
    values=data[split]; total_nats=0.; count_tokens=0
    # All complete 64-token segments, including a smaller final batch.
    nsegments=(len(values)-1)//64
    for first in range(0,nsegments,8):
        batch=min(8,nsegments-first)
        idx=(first+np.arange(batch))[:,None]*64+np.arange(64)
        count=batch*64
        total_nats+=m.loss(values[idx],values[idx+1],False)*count
        count_tokens+=count
    return float(total_nats/count_tokens/np.log(2))
def train(kind,seed,lr):
    m=Transformer(V,32,seed) if kind=='transformer' else Recurrent(V,rd,seed)
    rng=np.random.default_rng(1000+seed); arr=data['train']
    ms={k:np.zeros_like(v) for k,v in m.p.items()}; vs={k:np.zeros_like(v) for k,v in m.p.items()}; trace=[]
    start=time.perf_counter()
    for step in range(1,401):
        ix=rng.integers(len(arr)-64,size=8)[:,None]+np.arange(64)
        loss,g=m.loss(arr[ix],arr[ix+1]); norm=np.sqrt(sum(np.sum(v*v) for v in g.values())); scale=min(1.,1/(norm+1e-12))
        for k in m.p:
            q=g[k]*scale; ms[k]=.9*ms[k]+.1*q; vs[k]=.999*vs[k]+.001*q*q
            m.p[k]-=lr*(ms[k]/(1-.9**step))/(np.sqrt(vs[k]/(1-.999**step))+1e-8)
        if step%100==0:trace.append({'step':step,'batch_train_nats':float(loss)})
    seconds=time.perf_counter()-start
    row={'kind':kind,'seed':seed,'learning_rate':lr,'parameters':count(m),'training_tokens':204800,'training_seconds':seconds,'trace':trace,'validation_bpb':evaluate(m,'validation')}
    return m,row
if __name__=='__main__':
    tuning=[]; chosen={}
    for kind in ('transformer','recurrent'):
        for lr in P['learning_rates']:
            _,row=train(kind,101,lr); tuning.append(row); print('tuning',json.dumps(row),flush=True)
        chosen[kind]=min((r for r in tuning if r['kind']==kind),key=lambda r:r['validation_bpb'])['learning_rate']
    Path('selection.json').write_text(json.dumps({'selected_rates':chosen,'tuning':tuning},indent=2)+'\n')
    rows=[]
    for seed in (1,2,3):
        for kind in ('transformer','recurrent'):
            m,row=train(kind,seed,chosen[kind]); row['test_bpb']=evaluate(m,'test'); row['weights_sha256']=hashlib.sha256(b''.join(p.tobytes() for p in m.p.values())).hexdigest(); rows.append(row); print('final',json.dumps(row),flush=True)
    result={'protocol':P,'vocab':V,'recurrent_width':rd,'evaluation_tokens':{k:((len(data[k])-1)//64)*64 for k in ('validation','test')},'evaluation_target_bytes_omitted':{k:(len(data[k])-1)%64 for k in ('validation','test')},'numpy':np.__version__,'platform':platform.platform(),'data_manifest':json.loads(Path('data_manifest.json').read_text()),'split_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in raw.items()},'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ('models.py','run.py','protocol.json')},'selected_rates':chosen,'tuning':tuning,'final':rows}
    Path('results.json').write_text(json.dumps(result,indent=2)+'\n')
