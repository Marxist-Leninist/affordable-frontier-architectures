"""Independent read-only scientific reconstruction; writes only audit artifacts."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import json,hashlib,time,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
import run
P=Path(__file__).parent
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
J=lambda f:json.loads((P/f).read_text())
def close(a,b,tol=1e-11):
    np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def reference(raw):
    tables={k:defaultdict(Counter) for k in (4,8,16)}; qs=[];ks=[];ns=[]
    for i in range(4096):
        for k,t in tables.items():
            if i>=k:t[raw[i-k:i]][raw[i]]+=1
        if i<3968:continue
        counts=Counter();order=0
        for k in (16,8,4):
            if raw[i-k+1:i+1] in tables[k]:counts=tables[k][raw[i-k+1:i+1]];order=k;break
        n=sum(counts.values());q=np.array([(counts[b]+1/256)/(n+1) for b in range(256)])
        close(q.sum(),1);qs.append(q);ks.append(order);ns.append(n)
    return np.array(qs),np.array(ks),np.array(ns)
def main():
    start=time.process_time();protocol=J('protocol.json');items=J('items.json');selection=J('selection.json');result=J('results.json');prov=J('provenance.json')
    for f,h in {**protocol['source_sha256'],**protocol['inputs_sha256']}.items():assert H(P/f)==h,f
    assert selection['protocol_sha256']==H(P/'protocol.json') and result['selection_sha256']==H(P/'selection.json')
    train=(run.TRAIN/'train.txt').read_bytes();val=(run.TRAIN/'validation.txt').read_bytes()
    assert hashlib.sha256(train).hexdigest()==prov['training_hash'];assert hashlib.sha256(val).hexdigest()==prov['validation_hash']
    prior={}
    for r in prov['prior_receipts']:
        p=P.parent/r['directory']/'rows.json';assert H(p)==r['response_sha256']
        for row in json.loads(p.read_text())['rows']:
            text=row['row']['text'].encode();prior[hashlib.sha256(text).hexdigest()]=text
    assert len(prior)==prov['prior_unique_documents']
    docs={r['row_idx']:r['row']['text'].encode() for r in J('rows.json')['rows']}; seen=set()
    for item in items['evaluation']:
        raw=docs[item['doc']][item['start']:item['start']+4097];target=raw[-128:]
        assert target not in train and target not in val and not any(target in p for p in prior.values())
        assert hashlib.sha256(docs[item['doc']]).hexdigest() not in prior
        assert not any(target in text for d,text in docs.items() if d!=item['doc'])
        assert target not in seen;seen.add(target)
    expected_val=[]
    for s in range(0,len(val)-4096,4096):
        if val[s:s+4097][-128:] not in train:expected_val.append(s)
        if len(expected_val)==32:break
    assert expected_val==[i['start'] for i in items['validation']]
    arrays={s:np.load(P/f) for s,f in [('validation','validation_arrays.npz'),('evaluation','losses.npz')]};max_copy_error=0.
    for split in arrays:
        for n,item in enumerate(items[split]):
            raw=(val if split=='validation' else docs[item['doc']])[item['start']:item['start']+4097]
            assert hashlib.sha256(raw).hexdigest()==item['window_sha256'];assert hashlib.sha256(raw[-128:]).hexdigest()==item['target_sha256']
            dist,order,count=reference(raw);target=np.frombuffer(raw[-128:],dtype=np.uint8);q=dist[np.arange(128),target]
            close(q,arrays[split]['copy_probability'][n]);np.testing.assert_array_equal(order,arrays[split]['order'][n]);np.testing.assert_array_equal(count,arrays[split]['count'][n])
            cache=run.CompactCache()
            for pos,b in enumerate(raw[:-1]):
                cache.feed(b)
                if pos>=3968:
                    actual,nn,kk=cache.distribution();np.testing.assert_array_equal(actual,dist[pos-3968]);assert (nn,kk)==(count[pos-3968],order[pos-3968])
    calibration={};means={};parity=[]
    with torch.inference_mode():
        for arm in ('local','gru'):
            a=arrays['validation'];pn=2.**(-a[arm+'_neural']);q=a['copy_probability'];o=a['order'];grid=np.arange(11)/10
            losses=[float((-np.log2((1-w*(o>0))*pn+w*(o>0)*q)).mean()) for w in grid]
            close(losses,selection['arms'][arm]['constant_grid_bpb']);assert grid[np.argmin(losses)]==selection['arms'][arm]['constant']
            calibrated={}
            for k in (4,8,16):
                mask=o==k
                ls=[float(-np.log2((1-w)*pn[:,mask]+w*q[mask]).mean()) for w in grid]
                close(ls,selection['arms'][arm]['order_grid_bpb'][str(k)]);assert grid[np.argmin(ls)]==selection['arms'][arm]['by_order'][str(k)];calibrated[str(k)]=float(grid[np.argmin(ls)])
            calibration[arm]=dict(constant=float(grid[np.argmin(losses)]),by_order=calibrated)
            a=arrays['evaluation'];q=a['copy_probability'];o=a['order']
            for seed in (1,2,3):
                neural=a[f'{arm}_seed{seed}_neural'];pn=2.**(-neural)
                for variant in ('neural','historical','constant','by_order'):
                    if variant=='neural':w=np.zeros_like(q)
                    elif variant=='historical':w=({'local':.3,'gru':.2}[arm])*(o>0)
                    elif variant=='constant':w=calibration[arm]['constant']*(o>0)
                    else:w=sum((o==int(k))*v for k,v in calibrated.items())
                    reconstructed=-np.log2((1-w)*pn+w*q);key=f'{arm}_seed{seed}_{variant}';close(reconstructed,a[key]);close(reconstructed.mean(),result['bpb'][key]);means[key]=float(reconstructed.mean())
                model=run.load(arm,seed)
                raw=docs[items['evaluation'][0]['doc']][:4097];tokens=torch.tensor(list(raw[:-1]))[None,:]
                full=model(tokens);chunked,_=run.prefill(model,tokens,128,128)
                error=float((full[:,-128:]-chunked).abs().max());assert error<3e-5
                pruned_error=None
                if arm=='local':
                    pruned,_=run.local_prefill(model,tokens,128,True);pruned_error=float((full[:,-128:]-pruned).abs().max());assert pruned_error<3e-5
                changed=tokens.clone();changed[:,2048:]=(changed[:,2048:]+71)%256
                causal_error=float((full[:,:2048]-model(changed)[:,:2048]).abs().max());assert causal_error==0
                loss=(-full[:,-128:].double().log_softmax(-1).gather(-1,torch.tensor(list(raw[-128:]))[None,:,None]).squeeze(-1)/math.log(2)).numpy()[0]
                close(loss,neural[0],tol=2e-5)
                parity.append(dict(arm=arm,seed=seed,full_chunked_max_logit_error=error,full_pruned_max_logit_error=pruned_error,causality_max_logit_error=causal_error,saved_neural_bpb_max_error=float(np.max(abs(loss-neural[0])))))
    docids=np.array([i['doc'] for i in items['evaluation']]);unique=np.unique(docids);rng=np.random.default_rng(32);draws=rng.integers(0,len(unique),(5000,len(unique)));primary={}
    for arm in ('local','gru'):
        a=arrays['evaluation'];d=np.mean([a[f'{arm}_seed{s}_by_order']-a[f'{arm}_seed{s}_constant'] for s in (1,2,3)],axis=0)
        sums=np.array([d[docids==doc].sum() for doc in unique]);counts=np.array([d[docids==doc].size for doc in unique]);boot=sums[draws].sum(1)/counts[draws].sum(1)
        primary[arm]=dict(delta_bpb=float(d.mean()),paired_document_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),improvement_supported=bool(np.quantile(boot,.975)<0))
    report=dict(pass_all_assertions=True,cpu_seconds=time.process_time()-start,windows=len(docids),documents=len(unique),targets=len(docids)*128,validation_windows=32,calibration=calibration,primary=primary,all_saved_loss_means=means,neural_parity=parity,checks=['Frozen source/input/checkpoint hashes','Training/validation/prior/current exact exclusions','Independent causal Counter full copy distributions for every scored validation/evaluation target','Target-probability equals complete distribution gather','Independent all calibration grid objectives and argmins','Independent all saved loss and mean reconstructions','All six checkpoint full/chunked/pruned/causal neural sample checks','Independent paired document bootstrap'],limitations=['No cryptographic chronology: files bind protocol/selection/evaluation but evaluated_test flag is self-report; reviewed execution paths never fit on test.','Exact full-document and 128-byte target dedup only; near-duplicate and semantic overlap unmeasured.','Convenience cohort; two unadjusted primary comparisons; no frontier quality or service cost claim.'])
    (P/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (P/'independent_audit.md').write_text('# Independent v32 audit\n\nAll assertions passed. Reconstructed all validation/evaluation copy distributions, calibration objectives, saved mixture losses and means. Verified provenance hashes and exact overlap exclusions. All six checkpoints passed bounded full/chunked/pruned and neural causal parity.\n\n'+json.dumps(primary,indent=2)+'\n\nLimitations: '+ ' '.join(report['limitations'])+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
