"""Post-hoc validation-only decoder sensitivity. No tuning or retraining."""
import json,math,time,resource
from run import *
resource.setrlimit(resource.RLIMIT_CPU,(180,185));data=source_data()['validation'];starts=torch.linspace(0,len(data)-65,16).long();z=data[starts[:,None]+torch.arange(65)[None,:]];x,y=z[:,:-1],z[:,1:]
rows=[];before=time.process_time()
for kind in ['bptt','diffusion_weighted','diffusion_ce']:
 endpoint='equal_data' if kind=='bptt' else 'capped_cpu'
 f=P/'checkpoints'/f'{kind}_1_{endpoint}.pt';m=Recurrent();m.load_state_dict(torch.load(f,weights_only=True));m.eval()
 with torch.inference_mode():
  code=m.codebook();cos=code@code.T;cos=(cos.sum()-256)/(256*255)
  for k in [1,4,32]:
   g=torch.Generator().manual_seed(99172);probs=0
   for particle in range(1,33):
    noise=torch.randn(16,64,D,generator=g);probs=probs+m.predict(x,noise,k=k).double().softmax(-1)
    if particle in [4,32]:
     bp=-(probs[:,-32:]/particle).gather(-1,y[:,-32:,None]).squeeze(-1).log().mean()/math.log(2)
     rows.append(dict(kind=kind,seed=1,endpoint=endpoint,depth=k,particles=particle,bpb=float(bp),targets=512,mean_codebook_offdiag_cosine=float(cos)))
  noise=torch.randn(16,64,D,generator=g)
  for sigma in [.002,.03,.3,3.,80.]:
   logits=m.train_denoise(x,y,noise,torch.full((16,1,1),sigma));bp=F.cross_entropy(logits[:,-32:].reshape(-1,256),y[:,-32:].reshape(-1))/math.log(2)
   rows.append(dict(kind=kind,seed=1,endpoint=endpoint,diagnostic='target-conditioned denoising CE: NOT predictive loss',sigma=sigma,bpb=float(bp),targets=512))
write('diagnostic.json',dict(posthoc=True,split='validation',no_model_selection=True,rows=rows,cpu_seconds=time.process_time()-before,limits='Only16 validation windows and seed1; changing inference depth tests an altered decoder. Finite-mixture log-score remains biased relative to continuous marginal.'))
print(json.dumps(rows,indent=2))
