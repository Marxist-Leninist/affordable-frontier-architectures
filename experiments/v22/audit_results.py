"""Independent artifact-level reconstruction; does not re-run neural inference."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
r=json.loads((P/'results.json').read_text());a=np.load(P/'probabilities.npz');proto=json.loads((P/'protocol.json').read_text())
for f,h in proto['source_sha256'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h
assert a['copy'].shape==(91,128)
maximum=0
for row in r['rows']:
 p=a[f"{row['kind']}_{row['seed']}_{row['precision']}"];w=row['weight'];h=np.where(a['active'],(1-w)*p+w*a['copy'],p)
 for key,prob in [('base_bpb',p),('hybrid_bpb',h)]:
  assert np.isfinite(prob).all() and (prob>0).all() and (prob<=1).all();err=abs(-np.log2(prob).mean()-row[key]);maximum=max(maximum,err);assert err<1e-12
m=json.loads((P/'model_memory.json').read_text())
for kind in proto['kinds']:
 for seed in proto['seeds']:
  f=m[f'{kind}_{seed}_fp32'];q=m[f'{kind}_{seed}_int8'];assert q['serialized_state_dict_bytes']<f['serialized_state_dict_bytes'];assert all(x['dtype']=='torch.qint8' for x in q['weights'] if x['quantized'])
if (P/'timings.json').exists():
 t=json.loads((P/'timings.json').read_text());assert len(t['rows'])==1152
 groups={}
 for x in t['rows']:
  assert x['probability_sum_error']<1e-12
  k=(x['kind'],x['seed'],x['doc'],x['rep']);groups.setdefault(k,set()).add((x['precision'],x['hybrid']))
 assert all(len(v)==4 for v in groups.values())
print(json.dumps(dict(aggregate_rows=len(r['rows']),maximum_loss_reconstruction_error=maximum,source_hashes_checked=True,memory_checked=True,timing_checked=(P/'timings.json').exists()),indent=2))
