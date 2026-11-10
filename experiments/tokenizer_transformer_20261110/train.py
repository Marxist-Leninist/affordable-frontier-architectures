import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch, torch.nn as nn, torch.nn.functional as F, time, json, math, pathlib, random, argparse, hashlib, resource, platform
from transformers import AutoTokenizer
P=pathlib.Path(__file__).parent
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
class Block(nn.Module):
 def __init__(self,d):
  super().__init__();self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d,bias=False);self.proj=nn.Linear(d,d,bias=False);self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
 def forward(self,x):
  b,t,d=x.shape;q,k,v=self.qkv(self.n1(x)).reshape(b,t,3,4,d//4).permute(2,0,3,1,4).unbind(0)
  x=x+self.proj(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,d))
  return x+self.ff(self.n2(x))
class LM(nn.Module):
 def __init__(self,v,seed,d=64):
  super().__init__();torch.manual_seed(seed);self.emb=nn.Embedding(v,d);nn.init.normal_(self.emb.weight,std=.02)
  torch.manual_seed(seed+10000);self.pos=nn.Embedding(64,d);self.blocks=nn.Sequential(Block(d),Block(d));self.norm=nn.LayerNorm(d)
 def forward(self,x):
  h=self.blocks(self.emb(x)+self.pos(torch.arange(x.shape[1])));return F.linear(self.norm(h),self.emb.weight)
def batch(items,eos):
 seqs=[z["ids"] for z in items];n=max(map(len,seqs));x=torch.full((len(seqs),n),eos,dtype=torch.long);y=torch.full_like(x,-100)
 for i,s in enumerate(seqs):x[i,:len(s)]=torch.tensor([eos]+s[:-1]);y[i,:len(s)]=torch.tensor(s)
 return x,y,sum(z["bytes"] for z in items),sum(map(len,seqs))
@torch.no_grad()
def evaluate(model,items,eos):
 model.eval();nll=0.;nt=0;nb=0;correct=0;rows=[]
 for z in items:
  x,y,b,t=batch([z],eos);logits=model(x);loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction="sum")
  nll+=loss.item();nb+=b;nt+=t;correct+=(logits.argmax(-1)==y).sum().item();rows.append({"nll":loss.item(),"bytes":b,"tokens":t})
 model.train();return {"canonical_bits_per_utf8_byte":nll/(math.log(2)*nb),"token_nll_noncomparable":nll/nt,"token_accuracy_noncomparable":correct/nt,"bytes":nb,"tokens":nt,"rows":rows}
def run(name,seed,budget):
 tok=AutoTokenizer.from_pretrained(str(P/("gpt2" if name=="gpt2_padded" else name)),trust_remote_code=False)
 if tok.eos_token_id is None: tok.eos_token="<|endoftext|>"
 data={s:[{"ids":tok.encode(t,add_special_tokens=False),"bytes":len(t.encode())} for t in json.loads((P/(s+".json")).read_text())] for s in ["train","validation","test"]}
 model=LM(151669 if name=="gpt2_padded" else len(tok),seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.95),weight_decay=.1)
 out={"tokenizer":name,"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),"trunk_parameters":sum(p.numel() for n,p in model.named_parameters() if not n.startswith("emb.")),"training_cpu_budget_seconds":budget,"hardware":platform.processor(),"torch":torch.__version__,"precision":"float32"}
 # Verify causality and round-trip: changing future token cannot alter previous logits.
 model.eval();z=torch.tensor([[tok.eos_token_id,10,11,12]]);a=model(z);z[0,-1]=13;b=model(z)
 assert torch.equal(a[:,:-1],b[:,:-1]),"causality failure"
 model.train();order=list(range(len(data["train"])));random.Random(seed).shuffle(order);idx=0;epoch=0;cpu=0.;wall=0.;steps=0;tokens=0;bytes_seen=0;milestone=False
 out["initial_validation"]=evaluate(model,data["validation"],tok.eos_token_id)
 while cpu<budget:
  ids=order[idx:idx+4]
  if len(ids)<4:
   epoch+=1;random.Random(seed+epoch).shuffle(order);idx=0;ids=order[:4]
  items=[data["train"][i] for i in ids];idx+=4
  t0=time.process_time();w0=time.monotonic()
  x,y,b,n=batch(items,tok.eos_token_id);opt.zero_grad(set_to_none=True)
  loss=F.cross_entropy(model(x).flatten(0,1),y.flatten(),reduction="sum")/b
  assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  cpu+=time.process_time()-t0;wall+=time.monotonic()-w0;steps+=1;tokens+=n;bytes_seen+=b
  if steps==128:
   out["matched_512_raw_segments"]={"cpu_seconds":cpu,"wall_seconds":wall,"steps":steps,"tokens":tokens,"bytes_seen":bytes_seen,"validation":evaluate(model,data["validation"],tok.eos_token_id),"test":evaluate(model,data["test"],tok.eos_token_id)};milestone=True
  if steps%100==0:print(name,seed,steps,round(cpu,2),flush=True)
 out["fixed_cpu_endpoint"]={"cpu_seconds":cpu,"wall_seconds":wall,"steps":steps,"tokens":tokens,"bytes_seen":bytes_seen,"validation":evaluate(model,data["validation"],tok.eos_token_id),"test":evaluate(model,data["test"],tok.eos_token_id)}
 out["max_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 torch.save({"state_dict":model.state_dict(),"tokenizer":name,"seed":seed,"steps":steps},P/f"{name}_seed{seed}.pt")
 (P/f"{name}_seed{seed}.json").write_text(json.dumps(out,indent=2))
 print("COMPLETE",name,seed,out["fixed_cpu_endpoint"]["test"]["canonical_bits_per_utf8_byte"],flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--tokenizer",choices=["gpt2","qwen3","gpt2_padded"],required=True);ap.add_argument("--seed",type=int,required=True);ap.add_argument("--budget",type=float,default=240);a=ap.parse_args();run(a.tokenizer,a.seed,a.budget)
