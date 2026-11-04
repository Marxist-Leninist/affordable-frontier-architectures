"""Rebuild statistical control; audit serialized versus live Python object bytes."""
import sys,json,pickle,os,hashlib
from pathlib import Path
from collections import Counter,deque
import numpy as np
from ngram import Ngram
P=Path(__file__).parent;D=Path(os.environ.get('WIKITEXT_DATA',str(P.parent/'frontier_research_v8')))
def main():
 raw=(D/'train.txt').read_bytes();manifest=json.loads((P/'data_manifest.json').read_text());assert hashlib.sha256(raw).hexdigest()==manifest['splits']['train']['text_sha256']
 starts=np.load(P/'train_starts.npy');protocol=json.loads((P/'protocol.json').read_text());assert hashlib.sha256((P/'train_starts.npy').read_bytes()).hexdigest()==protocol['ledger_sha256']
 m=Ngram()
 for s in starts.flatten():m.add(raw[s:s+129])
 m.order=json.loads((P/'selection.json').read_text())['ngram_order'];seen=set();types=Counter();todo=[m];total=0
 while todo:
  o=todo.pop();i=id(o)
  if i in seen:continue
  seen.add(i);size=sys.getsizeof(o);total+=size;types[type(o).__name__]+=size
  if isinstance(o,dict):todo.extend(o.keys());todo.extend(o.values())
  elif isinstance(o,(list,tuple,set,frozenset,deque)):todo.extend(o)
  elif hasattr(o,'__dict__') and not isinstance(o,type):todo.append(vars(o))
 result=dict(method='Identity-deduplicated sys.getsizeof Python object graph. Excludes interpreter, allocator fragmentation, temporary activations and process RSS.',ngram_object_graph_bytes=total,unique_objects=len(seen),bytes_by_type=dict(types),serialized_bytes=len(pickle.dumps(m,protocol=5)),training_target_presentations=int(starts.size*128))
 (P/'memory_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
