# Exploratory broader test sampling, added after two main seeds. No training/model changes.
import json,random,hashlib,pathlib,pandas as pd,torch
from train import LM,evaluate
from transformers import AutoTokenizer
P=pathlib.Path(__file__).parent
toks={n:AutoTokenizer.from_pretrained(str(P/n),trust_remote_code=False) for n in ["gpt2","qwen3"]}
toks["gpt2"].eos_token="<|endoftext|>"
exclude=set()
for s in ["train","validation","test"]:exclude.update(json.loads((P/(s+".json")).read_text()))
pool=[];seen=set(exclude)
for row in pd.read_parquet(P/"test.parquet")["text"]:
 for start in range(0,len(row),160):
  s=row[start:start+160]
  if len(s)<100 or s.isspace() or s in seen:continue
  enc={k:t.encode(s,add_special_tokens=False) for k,t in toks.items()}
  if any(not 1<=len(v)<=63 for v in enc.values()):continue
  assert all(toks[k].decode(v,clean_up_tokenization_spaces=False)==s for k,v in enc.items())
  pool.append(s);seen.add(s)
random.Random(1234).shuffle(pool);texts=pool[:256]
assert len(texts)==256
meta={"sampling":"256 random eligible segments across full official test split, seed1234; excludes all retained main train/validation/test exact segments","posthoc":True,"eligible_pool":len(pool),"bytes":sum(len(s.encode()) for s in texts),"text_sha256":hashlib.sha256("\n".join(texts).encode()).hexdigest(),"results":{}}
for name in ["gpt2","qwen3","gpt2_padded"]:
 tok=toks["gpt2" if name=="gpt2_padded" else name]
 data=[{"ids":tok.encode(s,add_special_tokens=False),"bytes":len(s.encode())} for s in texts]
 for seed in [1,2,3]:
  d=torch.load(P/f"{name}_seed{seed}.pt",map_location="cpu",weights_only=True)
  model=LM(151669 if name=="gpt2_padded" else len(tok),seed);model.load_state_dict(d["state_dict"])
  r=evaluate(model,data,tok.eos_token_id);meta["results"][f"{name}_seed{seed}"]=r
  print(name,seed,r["canonical_bits_per_utf8_byte"],flush=True)
(P/"replication_eval.json").write_text(json.dumps(meta,indent=2))
print("REPLICATION_COMPLETE")
