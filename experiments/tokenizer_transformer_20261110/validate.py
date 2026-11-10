import json, pathlib, hashlib, math, torch
from train import LM, batch
from transformers import AutoTokenizer
P=pathlib.Path(__file__).parent
manifest=json.loads((P/"manifest.json").read_text())
allsets=[set(json.loads((P/f"{s}.json").read_text())) for s in ["train","validation","test"]]
assert not (allsets[0]&allsets[1] or allsets[0]&allsets[2] or allsets[1]&allsets[2])
checks=[]
for name in ["gpt2","qwen3","gpt2_padded"]:
 tok=AutoTokenizer.from_pretrained(str(P/("gpt2" if name=="gpt2_padded" else name)),trust_remote_code=False)
 if tok.eos_token_id is None:tok.eos_token="<|endoftext|>"
 for seed in [1,2,3]:
  r=json.loads((P/f"{name}_seed{seed}.json").read_text());assert "matched_512_raw_segments" in r
  assert r["fixed_cpu_endpoint"]["cpu_seconds"]>=240
  for e in ["matched_512_raw_segments","fixed_cpu_endpoint"]:
   for split in ["validation","test"]:
    rows=r[e][split]["rows"];bpb=sum(x["nll"] for x in rows)/(math.log(2)*sum(x["bytes"] for x in rows))
    assert abs(bpb-r[e][split]["canonical_bits_per_utf8_byte"])<1e-10
    assert sum(x["bytes"] for x in rows)==manifest["splits"][split]["bytes"]
  cp=P/f"{name}_seed{seed}.pt";d=torch.load(cp,map_location="cpu",weights_only=True);m=LM(151669 if name=="gpt2_padded" else len(tok),seed);m.load_state_dict(d["state_dict"]);m.eval()
  sample=json.loads((P/"test.json").read_text())[0];x,y,nb,nt=batch([{"ids":tok.encode(sample,add_special_tokens=False),"bytes":len(sample.encode())}],tok.eos_token_id)
  with torch.no_grad():loss=torch.nn.functional.cross_entropy(m(x).flatten(0,1),y.flatten(),reduction="sum").item()
  assert abs(loss-r["fixed_cpu_endpoint"]["test"]["rows"][0]["nll"])<1e-5
  checks.append({"arm":name,"seed":seed,"checkpoint_sha256":hashlib.sha256(cp.read_bytes()).hexdigest(),"reload_nll_matches":True,"causality_passed_during_training":True})
for seed in [1,2,3]:
 rows=[json.loads((P/f"{n}_seed{seed}.json").read_text())["matched_512_raw_segments"] for n in ["gpt2","qwen3","gpt2_padded"]]
 assert len({r["bytes_seen"] for r in rows})==1
 assert len({r["steps"] for r in rows})==1
(P/"validation_checks.json").write_text(json.dumps({"passed":True,"checks":checks,"exact_cross_split_duplicate_check":True},indent=2));print("ALL_VALIDATIONS_PASSED")
