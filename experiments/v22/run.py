"""Frozen causal dynamic-int8 comparison. Run freeze.py and tests before this script."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import io,json,time,hashlib,random,platform
from pathlib import Path
import numpy as np,torch
from models import LM,state_bytes
from local_attention import localize
from pruning import local_prefill
from prefill import prefill
from compact_cache import CompactCache,reachable_bytes
P=Path(__file__).resolve().parent;OLD=P.parent/'frontier_research_v21';V19=P.parent/'frontier_research_v19'
torch.set_num_threads(1);torch.backends.quantized.engine='x86'
KINDS=('transformer','transformer_reset512','local','gru');SEEDS=(1,2,3)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(P/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(kind,seed,quant=False):
    base='transformer' if kind.startswith('transformer') else kind;m=LM('transformer' if base=='local' else base)
    if base=='local':m=localize(m,128)
    f=V19/'checkpoints'/f'{base}_seed{seed}_adapt500.pt';assert sha(f)==json.loads((P/'input_checkpoint_manifest.json').read_text())[f.name]
    m.load_state_dict(torch.load(f,weights_only=True));m.eval()
    return torch.ao.quantization.quantize_dynamic(m,{torch.nn.Linear,torch.nn.GRU},dtype=torch.qint8,inplace=False) if quant else m

def data():
    man=json.loads((OLD/'data_manifest.json').read_text());assert sha(OLD/'rows.json')==man['response_sha256'];items=[];docs=[]
    for row,m in zip(json.loads((OLD/'rows.json').read_text())['rows'],man['documents']):
        raw=row['row']['text'].encode()[:4097];assert hashlib.sha256(raw).hexdigest()==m['capped_text_sha256']
        if len(raw)>=2081:docs.append((row['row_idx'],raw[:2080]))
        for s in range(0,len(raw)-2048,2048):items.append(dict(doc=row['row_idx'],start=s,raw=raw[s:s+2049]))
    return items,docs[:8]
def start(m,kind,x):
    if kind=='local':return local_prefill(m,x,1,prune=True)
    return prefill(m,x,None,1)
@torch.inference_mode()
def predict(m,kind,raw,quality=True):
    # For reset512 the same final 512 input region is consumed causally.
    begin=1536 if kind=='transformer_reset512' and quality else (1536 if kind=='transformer_reset512' else 0)
    stop=1921 if quality else 2048;end=2048 if quality else 2080
    x=torch.tensor([list(raw)]);z,state=start(m,kind,x[:,begin:stop]);ps=[z[0,-1].double().softmax(-1).numpy()]
    for i in range(stop,end):
        z,state=m(x[:,i:i+1],state,cache=True);ps.append(z[0,-1].double().softmax(-1).numpy())
    return np.array(ps),state

def tensor_inventory(m):
    weights=[];seen=set()
    for n,p in m.named_parameters():weights.append(dict(name=n,dtype=str(p.dtype),numel=p.numel(),payload_bytes=p.numel()*p.element_size(),quantized=False))
    for n,mod in m.named_modules():
        candidates={}
        if isinstance(mod,torch.ao.nn.quantized.dynamic.Linear):candidates={'weight':mod.weight()}
        elif isinstance(mod,torch.ao.nn.quantized.dynamic.GRU):candidates=mod.get_weight()
        for k,w in candidates.items():
            q=dict(name=n+'.'+k,dtype=str(w.dtype),numel=w.numel(),payload_bytes=w.numel()*w.element_size(),quantized=True,qscheme=str(w.qscheme()))
            if w.qscheme() in (torch.per_tensor_affine,torch.per_tensor_symmetric):q.update(scale=w.q_scale(),zero_point=w.q_zero_point(),logical_qparam_bytes=16)
            else:q.update(scales=w.q_per_channel_scales().tolist(),zero_points=w.q_per_channel_zero_points().tolist(),logical_qparam_bytes=w.q_per_channel_scales().numel()*16)
            weights.append(q)
    buff=io.BytesIO();torch.save(m.state_dict(),buff)
    return dict(serialized_state_dict_bytes=len(buff.getvalue()),logical_weight_payload_bytes=sum(w['payload_bytes'] for w in weights),weights=weights,modules={n:type(mod).__module__+'.'+type(mod).__name__ for n,mod in m.named_modules() if isinstance(mod,(torch.ao.nn.quantized.dynamic.Linear,torch.ao.nn.quantized.dynamic.GRU))},note='Logical unpacked quantized tensor payload, not backend packed allocation or RSS. GRU biases remain FP32 inside packed state; serialized size includes those and metadata. Embeddings, RMSNorm, activations and persistent recurrent/KV state remain FP32. Backend packed resident allocation is not measurable by this inventory.')

def main():
    p=json.loads((P/'protocol.json').read_text());assert p['source_sha256']['run.py']==sha(P/'run.py');cpu=time.process_time();items,_=data();cp=[];active=[]
    for it in items:
        cache=CompactCache();ps=[];mask=[]
        for i,b in enumerate(it['raw'][:-1]):
            cache.feed(b)
            if i>=1920:q,n,_=cache.distribution();ps.append(q[it['raw'][i+1]]);mask.append(n>0)
        cp.append(ps);active.append(mask)
    cp=np.array(cp);active=np.array(active);frozen=json.loads((P/'frozen_selection.json').read_text());arrays={};rows=[];memory={}
    for kind in KINDS:
        for seed in SEEDS:
            for quant in (False,True):
                key=f'{kind}_{seed}_{"int8" if quant else "fp32"}';m=load(kind,seed,quant);memory[key]=tensor_inventory(m);probs=[]
                for it in items:
                    if time.process_time()-cpu>650:raise RuntimeError('Scoring CPU budget reached')
                    ps,_=predict(m,kind,it['raw']);targets=np.frombuffer(it['raw'][-128:],dtype=np.uint8);probs.append(ps[np.arange(128),targets])
                probs=np.array(probs);arrays[key]=probs;w=frozen['transformer_full2048' if kind=='transformer' else kind]['weight'];hy=np.where(active,(1-w)*probs+w*cp,probs)
                rows.append(dict(kind=kind,seed=seed,precision='int8' if quant else 'fp32',base_bpb=float(-np.log2(probs).mean()),hybrid_bpb=float(-np.log2(hy).mean()),weight=w));print(rows[-1],flush=True)
    np.savez_compressed(P/'probabilities.npz',copy=cp,active=active,**arrays);write('items.json',[{k:v for k,v in i.items() if k!='raw'} for i in items]);write('model_memory.json',memory);write('results.json',dict(rows=rows,cpu_seconds=time.process_time()-cpu,windows=len(items),targets=len(items)*128,documents=len(set(i['doc'] for i in items))))
if __name__=='__main__':main()
