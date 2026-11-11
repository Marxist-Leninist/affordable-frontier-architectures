import gc,json,resource,sys,time
from pathlib import Path
import torch
from models import LanguageModel,loss
P=Path(__file__).resolve().parent

def storage(t):
    s=t.untyped_storage();return (str(t.device),s.data_ptr(),s.nbytes())
def main(mode,depth):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(8)
    base=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    m=LanguageModel(mode,depth=depth);opt=torch.optim.AdamW(m.parameters(),lr=.002)
    x=torch.randint(256,(8,64));y=torch.randint(256,(8,64));params={storage(p) for p in m.parameters()};saved=set()
    def pack(t):
        k=storage(t)
        if k not in params:saved.add(k)
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):l=loss(m(x),y)
    # Only graph retained immediately after forward is counted, not recompute temporaries.
    retained=sum(s[2] for s in saved);l.backward();opt.step();del l;gc.collect()
    times=[]
    for i in range(13):
        c=time.process_time();w=time.perf_counter();opt.zero_grad(set_to_none=True);loss(m(x),y).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        if i>=3:times.append({'cpu_seconds':time.process_time()-c,'wall_seconds':time.perf_counter()-w})
    states=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor))
    out={'mode':mode,'depth':depth,'batch':8,'context':64,'parameters':sum(p.numel() for p in m.parameters()),'parameter_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),'optimizer_tensor_bytes':states,'retained_forward_nonparameter_storage_bytes':retained,'unique_saved_storages':len(saved),'baseline_import_peak_rss_bytes':base,'whole_process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'timings':times,'caveats':['Saved storage is union retained by forward autograd, not whole step peak or allocator/GPU memory.','RSS is whole process high-water mark and includes runtime, model, gradients, optimizer and all13 updates. Baseline subtraction is a rough diagnostic.','Fresh process per mode/depth, one CPU thread; timing includes clip and optimizer but excludes data generation.']}
    (P/f'memory_{mode}_d{depth}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
