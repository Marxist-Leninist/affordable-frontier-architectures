"""Measure saved-tensor storage and warmed CPU timings, excluding parameters."""
import run
from run import *
def main():
    d=data();x,y=batch(d['train'],torch.Generator().manual_seed(71));rows=[]
    for kind in ['normal_finetune','pretrained_diffusion']:
        torch.manual_seed(1);orig=load_original();m=DiffusionBlocks(orig) if kind=='pretrained_diffusion' else frozen_maps(orig)
        params=[p for p in m.parameters() if p.requires_grad]
        persistent={p.untyped_storage().data_ptr() for p in list(m.parameters())+list(m.buffers())}
        noise=torch.randn(BS,CONTEXT,D);s=sample_sigma(1,BS,torch.Generator().manual_seed(2))
        seen={}
        def pack(t):
            st=t.untyped_storage();key=st.data_ptr()
            if key not in persistent:seen[key]=st.nbytes()
            return t
        def step():
            m.zero_grad(set_to_none=True)
            logits=m.training_logits(x,y,noise,s,1) if kind=='pretrained_diffusion' else m(x,use_cache=False).logits
            F.cross_entropy(logits.flatten(0,1),y.flatten()).backward()
        for _ in range(2):step()
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):step()
        t=time.process_time()
        for _ in range(8):step()
        training=(time.process_time()-t)/8
        m.eval()
        with torch.inference_mode():
            def inference():return m.predict(x,noise) if kind=='pretrained_diffusion' else m(x,use_cache=False).logits
            inference();t=time.process_time()
            for _ in range(4):inference()
            infer=(time.process_time()-t)/4
        row=dict(method=kind,saved_nonparameter_storage_bytes=sum(seen.values()),trainable_parameters=trainable_count(m),total_parameters=sum(p.numel() for p in m.parameters()),forward_backward_cpu_seconds=training,single_particle_inference_cpu_seconds=infer)
        print(row,flush=True);rows.append(row);del m,orig,params
    write('accounting.json',dict(scope='CPU one thread,BS2,context32,warmed; activations=unique storages saved for backward excluding model parameters/buffers,NOT GPU peak allocated memory. Timing excludes optimizer and is descriptive,not endpoint budget. Decoder1particle; primary evaluates4.',rows=rows))
if __name__=='__main__':main()
