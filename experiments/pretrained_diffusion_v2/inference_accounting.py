"""Full-prefix decoder cost and cache storage in the actual converted model.

This is NOT an incremental autoregressive continuation or long-context test.
"""
import run
from run import *
from accounting import restore
def storage(tensors):
    return dict(logical_bytes=sum(x.numel()*x.element_size() for x in tensors),unique_retained_storage_bytes=sum({x.untyped_storage().data_ptr():x.untyped_storage().nbytes() for x in tensors}.values()))
@torch.inference_mode()
def main():
    x,_=getbatch(data()['test'],torch.tensor(json.loads((P/'window_manifest.json').read_text())['splits']['test']['starts'][:2]));rows=[]
    for kind in ['normal_ce','direct_kd']:
        m=restore(kind,11,'equal_data');m.eval();g=torch.Generator().manual_seed(912);noises=[torch.randn(BS,T,D,generator=g) for _ in range(4)]
        if kind=='normal_ce':
            cache=m(x,use_cache=True).past_key_values
            tensors=[]
            for layer in range(6):tensors.extend(cache[layer])
            cachebytes=storage(tensors)
            def ordinary():return m(x,use_cache=False).logits.log_softmax(-1)
            ordinary();t=time.process_time();w=time.perf_counter()
            for _ in range(5):ordinary()
            rows.append(dict(kind=kind,particles=1,cpu_seconds=(time.process_time()-t)/5,wall_seconds=(time.perf_counter()-w)/5,kv_cache=cachebytes));del cache,tensors
        else:
            ctx=m.prepare_context(x);tensors=[a for c in ctx.values() for kv in c[1] for a in kv];cachebytes=storage(tensors);del ctx,tensors
            for particles in [1,4]:
                def converted():
                    contexts=m.prepare_context(x);acc=None
                    for z in noises[:particles]:
                        lp=m.predict(x,z,'normalized_rollout',contexts).log_softmax(-1)
                        acc=lp if acc is None else torch.logaddexp(acc,lp)
                    return acc-math.log(particles)
                converted();t=time.process_time();w=time.perf_counter()
                for _ in range(5):converted()
                rows.append(dict(kind=kind,particles=particles,cpu_seconds=(time.process_time()-t)/5,wall_seconds=(time.perf_counter()-w)/5,kv_cache=cachebytes))
        del m
    write('inference_accounting.json',dict(scope='Warmed FP32,one CPU thread,batch2,context32,full-prefix distributions. Charges clean-context preparation on every converted repetition, reuses only across denoising steps/particles within that fixed prefix. Does not measure incremental decoding. KV backing-storage counts include view-retained unused allocation; logical bytes count tensor elements. No cross-block KV sharing.',rows=rows));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
