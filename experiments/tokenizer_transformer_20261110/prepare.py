import requests, json, hashlib, pathlib, pandas as pd
from transformers import AutoTokenizer
P=pathlib.Path(__file__).parent
DS="b08601e04326c79dfdd32d625aee71d232d685c3"
specs={"gpt2":("openai-community/gpt2","607a30d783dfa663caf39e06633721c8d4cfcd7e"),"qwen3":("Qwen/Qwen3-235B-A22B","8efa61729e24bd65b1d152b5ab5409052aa80e65")}
manifest={"dataset":"Salesforce/wikitext","config":"wikitext-2-raw-v1","revision":DS,"license":"CC-BY-SA-3.0 and GFDL per card","tokenizers":{},"splits":{}}
def get(url,path):
 r=requests.get(url,timeout=60); r.raise_for_status(); path.write_bytes(r.content);return hashlib.sha256(r.content).hexdigest()
toks={}
for name,(repo,rev) in specs.items():
 d=P/name;d.mkdir(exist_ok=True)
 for f in ["tokenizer.json","tokenizer_config.json"]:
  h=get(f"https://huggingface.co/{repo}/resolve/{rev}/{f}",d/f)
  manifest["tokenizers"].setdefault(name,{"repo":repo,"revision":rev,"files":{}})["files"][f]=h
 # config omitted: tokenizer_config declares class
 t=AutoTokenizer.from_pretrained(str(d),trust_remote_code=False);toks[name]=t
 manifest["tokenizers"][name].update(vocab_size=len(t),eos=t.eos_token_id)
small=get("https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json",P/"qwen06_tokenizer.json")
manifest["qwen06_tokenizer_same_as_235b"]=small==manifest["tokenizers"]["qwen3"]["files"]["tokenizer.json"]
seen=set()
for split,limit in [("train",10000),("validation",128),("test",128)]:
 path=P/(split+".parquet");h=get(f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{DS}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet",path)
 texts=pd.read_parquet(path)["text"].tolist()
 samples=[]; duplicates=0
 for row in texts:
  # Segments of 160 Unicode characters, no normalization; complete tokenizer round trips.
  for start in range(0,len(row),160):
   s=row[start:start+160]
   if len(s)<100 or s.isspace():continue
   hs=hashlib.sha256(s.encode()).hexdigest()
   if hs in seen:duplicates+=1;continue
   encoded={k:t.encode(s,add_special_tokens=False) for k,t in toks.items()}
   if any(not 1<=len(v)<=63 for v in encoded.values()):continue
   if any(toks[k].decode(v,clean_up_tokenization_spaces=False)!=s for k,v in encoded.items()):raise ValueError("roundtrip")
   seen.add(hs);samples.append(s)
   if len(samples)>=limit:break
  if len(samples)>=limit:break
 (P/(split+".json")).write_text(json.dumps(samples,ensure_ascii=False))
 manifest["splits"][split]={"parquet_sha256":h,"samples":len(samples),"bytes":sum(len(s.encode()) for s in samples),"duplicates_removed":duplicates,"text_sha256":hashlib.sha256("\n".join(samples).encode()).hexdigest(),"tokens":{k:sum(len(t.encode(s,add_special_tokens=False)) for s in samples) for k,t in toks.items()}}
(P/"manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)
