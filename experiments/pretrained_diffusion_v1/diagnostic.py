"""Post-hoc VALIDATION ONLY decoder sensitivity; never select on test."""
import run
from run import *
@torch.inference_mode()
def score(m,d,mode,particles,n=8):
    starts=torch.linspace(0,len(d)-33,n).long();out=[];g=torch.Generator().manual_seed(449874)
    for ss in starts.split(2):
        a=d[ss[:,None]+torch.arange(33)[None,:]];x,y=a[:,:-1],a[:,1:];vals=[]
        for _ in range(particles):
            z=torch.randn(2,32,128,generator=g)
            logits=m.denoise(x,z*80,z.new_full((2,1,1),80.),0) if mode=='high_noise_head' else m.predict(x,z)
            vals.append(logits.log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1))
        ll=torch.logsumexp(torch.stack(vals),0)-math.log(particles)
        out.extend((-ll[:,-16:]).mean(1).tolist())
    return dict(mode=mode,particles=particles,nll_nats=float(np.mean(out)),scored_validation_targets=n*16)
def main():
    torch.manual_seed(1);m=DiffusionBlocks(load_original());state=torch.load(P/'checkpoints'/'pretrained_diffusion_1_equal_cpu.pt',weights_only=True)
    result=m.load_state_dict(state,strict=False)
    assert not result.unexpected_keys;assert set(state)=={n for n,p in m.named_parameters() if p.requires_grad}
    m.eval();d=data()['validation'];rows=[]
    for mode in ['high_noise_head','iterative']:
        for particles in [4,32]:
            r=score(m,d,mode,particles);rows.append(r);print(r,flush=True)
    write('diagnostic.json',dict(scope='Post-hoc validation-only diagnosis, seed1 CPUmatched checkpoint,128targets; no hyperparameter or checkpoint selection and not a replacement main result',results=rows))
if __name__=='__main__':main()
