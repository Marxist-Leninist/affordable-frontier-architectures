import json, pathlib, math, statistics, collections, hashlib
from transformers import AutoTokenizer
P=pathlib.Path(__file__).parent
manifest=json.loads((P/"manifest.json").read_text())
output={"arms":{},"unigram_full_train":{},"scope":"Small from-scratch language modelling; not an intelligence benchmark"}
for name in ["gpt2","qwen3","gpt2_padded"]:
 results=[json.loads((P/f"{name}_seed{s}.json").read_text()) for s in [1,2,3]]
 output["arms"][name]={"parameters":results[0]["parameters"],"trunk_parameters":results[0]["trunk_parameters"]}
 for endpoint in ["matched_512_raw_segments","fixed_cpu_endpoint"]:
  rows=[r[endpoint] for r in results]
  bpb=[r["test"]["canonical_bits_per_utf8_byte"] for r in rows]
  output["arms"][name][endpoint]={"test_bpb_seeds":bpb,"test_bpb_mean":statistics.mean(bpb),"test_bpb_stdev":statistics.stdev(bpb),"validation_bpb_seeds":[r["validation"]["canonical_bits_per_utf8_byte"] for r in rows],"training_cpu_seconds":[r["cpu_seconds"] for r in rows],"training_wall_seconds":[r["wall_seconds"] for r in rows],"steps":[r["steps"] for r in rows],"raw_bytes_seen":[r["bytes_seen"] for r in rows],"tokens_seen":[r["tokens"] for r in rows]}
 # A cheap full-corpus unigram reference contextualizes how undertrained the Transformer is.
 tok=AutoTokenizer.from_pretrained(str(P/("gpt2" if name=="gpt2_padded" else name)),trust_remote_code=False)
 texts=json.loads((P/"train.json").read_text());counts=collections.Counter()
 for s in texts:counts.update(tok.encode(s,add_special_tokens=False))
 vocab=151669 if name=="gpt2_padded" else len(tok);total=sum(counts.values())+vocab
 texts=json.loads((P/"test.json").read_text());nll=0.;nbytes=0
 for s in texts:
  nll-=sum(math.log((counts[t]+1)/total) for t in tok.encode(s,add_special_tokens=False));nbytes+=len(s.encode())
 output["unigram_full_train"][name]={"test_bpb":nll/(math.log(2)*nbytes),"smoothing":1,"note":"Trained on all retained training segments, not compute/data matched to Transformer endpoints."}
for endpoint in ["matched_512_raw_segments","fixed_cpu_endpoint"]:
 q=output["arms"]["qwen3"][endpoint]["test_bpb_seeds"]
 for baseline in ["gpt2","gpt2_padded"]:
  g=output["arms"][baseline][endpoint]["test_bpb_seeds"];d=[a-b for a,b in zip(q,g)]
  output.setdefault("paired_differences",{})[endpoint+"_qwen_minus_"+baseline]={"seeds":d,"mean":statistics.mean(d),"stdev":statistics.stdev(d),"interpretation":"Positive means worse canonical text likelihood for Qwen3 in this experiment only."}
(P/"summary.json").write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
