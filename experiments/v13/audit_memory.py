"""Owned Python object graph accounting; excludes interpreter/allocator overhead."""
import sys,json,hashlib
from collections import Counter,deque
from pathlib import Path
import numpy as np
from ngram import Ngram
P=Path(__file__).parent;V=P.parent/'frontier_research_v12';D=P.parent/'frontier_research_v8'
raw=(D/'train.txt').read_bytes();starts=np.load(V/'train_starts.npy');m=Ngram()
for s in starts.flatten():m.add(raw[s:s+129])
seen=set();types=Counter();todo=[m];total=0
while todo:
 o=todo.pop();i=id(o)
 if i in seen:continue
 seen.add(i);size=sys.getsizeof(o);total+=size;types[type(o).__name__]+=size
 if isinstance(o,dict):todo.extend(o.keys());todo.extend(o.values())
 elif isinstance(o,(list,tuple,set,frozenset,deque)):todo.extend(o)
 elif hasattr(o,'__dict__') and not isinstance(o,type):todo.append(vars(o))
result={'method':'Identity-deduplicated recursive sys.getsizeof of rebuilt Ngram instance, instance dict, containers, keys, values. Shared identities counted once. Does not follow class/type objects or defaultdict factory types. Excludes Python interpreter, allocator fragmentation, transient inference allocations, neural runtime, and RSS.','ngram_object_graph_bytes':total,'unique_objects':len(seen),'bytes_by_type':dict(types),'ngram_serialized_checkpoint_bytes':(V/'checkpoints/ngram.pkl').stat().st_size,'ledger_sha256':hashlib.sha256((V/'train_starts.npy').read_bytes()).hexdigest(),'training_targets':int(starts.size*128)}
(P/'audit_memory.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
