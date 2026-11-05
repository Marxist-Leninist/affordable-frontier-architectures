"""Exact counts for complete context/follower pairs in a sliding byte suffix."""
from collections import deque
import numpy as np
class BoundedCache:
 def __init__(self,limit,orders=(4,8,16)):
  if not isinstance(limit,int) or limit<max(orders)+1:raise ValueError('Invalid limit')
  self.limit=limit;self.orders=tuple(orders);self.text=deque(maxlen=max(orders));self.tables={k:{} for k in orders};self.events={k:deque() for k in orders};self.seen=0
 def feed(self,byte):
  if not isinstance(byte,int) or not 0<=byte<=255:raise ValueError('Invalid byte')
  history=bytes(self.text);j=self.seen
  for k in self.orders:
   table=self.tables[k];events=self.events[k]
   if j>=k:
    key=history[-k:];counts=table.setdefault(key,{})
    counts[byte]=counts.get(byte,0)+1;events.append((j-k,key,byte))
   cutoff=j+1-self.limit
   while events and events[0][0]<cutoff:
    _,key,b=events.popleft();counts=table[key];counts[b]-=1
    if not counts[b]:del counts[b]
    if not counts:del table[key]
  self.text.append(byte);self.seen+=1
 def distribution(self):
  history=bytes(self.text);q=np.full(256,1/256,dtype=np.float64)
  for k in reversed(self.orders):
   if self.seen<k:continue
   counts=self.tables[k].get(history[-k:])
   if counts:
    n=sum(counts.values())
    for b,v in counts.items():q[b]+=v
    return q/(n+1),n,k
  return q,0,0

def reachable_bytes(obj):
 """Reachable CPython memory including deque entries, not RSS."""
 import sys
 seen=set()
 def size(o):
  if id(o) in seen:return 0
  seen.add(id(o));n=sys.getsizeof(o)
  if isinstance(o,dict):n+=sum(size(k)+size(v) for k,v in o.items())
  elif isinstance(o,(tuple,list,deque)):n+=sum(size(x) for x in o)
  elif hasattr(o,'__dict__'):n+=size(vars(o))
  return n
 return size(obj)
