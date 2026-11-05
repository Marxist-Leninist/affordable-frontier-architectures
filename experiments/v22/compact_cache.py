"""Exact full-history byte-copy counts, packed sparse follower lists.
No eviction, approximation, target lookahead, or context selection change.
"""
from array import array
import numpy as np
class CompactCache:
    def __init__(self,orders=(4,8,16)):
        self.orders=tuple(orders)
        if not self.orders or tuple(sorted(set(self.orders)))!=self.orders or min(self.orders)<1:raise ValueError('Orders must be increasing positive integers')
        self.masks=tuple((1<<(8*k))-1 for k in self.orders)
        self.tables=tuple({} for _ in self.orders)
        self.head=array('i');self.total=array('I');self.byte=array('B');self.count=array('I');self.next=array('i')
        self.rolling=0;self.seen=0
    def feed(self,byte):
        if not isinstance(byte,int) or not 0<=byte<256:raise ValueError('Expected a byte integer')
        for k,mask,table in zip(self.orders,self.masks,self.tables):
            if self.seen<k:continue
            key=self.rolling&mask;ctx=table.get(key)
            if ctx is None:
                ctx=len(self.head)
                if ctx>=2147483647:raise OverflowError('Context index overflow')
                table[key]=ctx;self.head.append(-1);self.total.append(0)
            if self.total[ctx]==4294967295:raise OverflowError('Context count overflow')
            node=self.head[ctx]
            while node!=-1 and self.byte[node]!=byte:node=self.next[node]
            if node==-1:
                node=len(self.byte)
                if node>=2147483647:raise OverflowError('Follower index overflow')
                self.byte.append(byte);self.count.append(1);self.next.append(self.head[ctx]);self.head[ctx]=node
            else:self.count[node]+=1
            self.total[ctx]+=1
        self.rolling=((self.rolling<<8)|byte)&self.masks[-1];self.seen+=1
    def query(self):
        for k,mask,table in reversed(tuple(zip(self.orders,self.masks,self.tables))):
            if self.seen<k:continue
            ctx=table.get(self.rolling&mask)
            if ctx is not None:return ctx,int(self.total[ctx]),k
        return -1,0,0
    def distribution(self):
        ctx,n,k=self.query();q=np.full(256,1/256,dtype=np.float64)
        if ctx!=-1:
            node=self.head[ctx]
            while node!=-1:q[self.byte[node]]+=self.count[node];node=self.next[node]
        q/=n+1
        return q,n,k
    def target_probability(self,target):
        ctx,n,k=self.query();count=0
        if ctx!=-1:
            node=self.head[ctx]
            while node!=-1:
                if self.byte[node]==target:count=self.count[node];break
                node=self.next[node]
        return (count+1/256)/(n+1),n,k

def reachable_bytes(obj):
    """CPython reachable object bytes, not peak RSS or process allocation delta."""
    import sys
    seen=set()
    def size(o):
        if id(o) in seen:return 0
        seen.add(id(o));n=sys.getsizeof(o)
        if isinstance(o,dict):n+=sum(size(k)+size(v) for k,v in o.items())
        elif isinstance(o,(list,tuple)):n+=sum(size(x) for x in o)
        elif hasattr(o,'__dict__'):n+=size(vars(o))
        # array.__sizeof__ already includes its allocated data buffer.
        return n
    return size(obj)
