from bench import *
k=sys.argv[1];assert k in KINDS
torch.manual_seed(17);m=Model('ar');o=torch.optim.AdamW(m.parameters(),lr=.002);d=read('train_524288.bin')
idx=np.random.default_rng(17).integers(0,len(d)-N,size=(64,B));warm=12
wall=[];cpu=[];mem={};start=0
for j,i in enumerate(idx):
 x,y=grab(d,i)
 if j==warm:start=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 s=time.perf_counter();c=time.process_time();o.zero_grad(set_to_none=True)
 _l,p=grad(m,x,y,k,probe=j==warm)
 nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
 if j>=warm:wall.append(time.perf_counter()-s);cpu.append(time.process_time()-c)
 if j==warm:mem=p
st=dict(mode=k,steps=len(wall),wall_p50_s=float(np.median(wall[1:])),cpu_p50_s=float(np.median(cpu[1:])),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,prefill_rss_kib=start,activation_proxy=mem)
(OUT/f'profile_{k}.json').write_text(json.dumps(st,indent=2)+'\n')
print(st,flush=True)
