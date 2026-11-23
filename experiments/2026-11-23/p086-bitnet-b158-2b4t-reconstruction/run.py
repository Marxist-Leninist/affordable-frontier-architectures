"""Portable reproduction helpers from P079; no training entry point."""
import argparse, hashlib, json, math, os, statistics, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES=((256,128),)+((384,128),(128,128),(512,128),(128,512))*3
D=128; T=128; B=16; STEPS=1800; SEEDS=(1867,1871); DEV_EVERY=150
ARMS=('grad','gate','packed4')
HASH={'train':'c58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8',
'validation':'714a24a5de7459876f7cfdc64837e551dd7cde2a51f95b763ee083fff273e307',
'test':'aca2f46735043bcfd0a44eca981d04627b9cdf74c4c9a04bf0856d04066f58fc'}
def digest(x):return hashlib.sha256(x).hexdigest()
def atomic(p,writer):
    q=p.with_suffix('.partial')
    with q.open('wb') as f:writer(f);f.flush();os.fsync(f.fileno())
    os.replace(q,p)
def save(p,x):atomic(p,lambda f:f.write((json.dumps(x,allow_nan=False,indent=2)+'\n').encode()))

def bits_pack(b):
    shifts=torch.arange(8,device=b.device)
    return (b.flatten().view(-1,8).to(torch.int16)<<shifts).sum(-1).to(torch.uint8)
def bits_open(p):
    shifts=torch.arange(8,device=p.device)
    return ((p[:,None].to(torch.int16)>>shifts)&1).flatten().bool()
def nib_pack(q):
    c=(q.flatten().to(torch.int16)+8).view(-1,2)
    return (c[:,0]|(c[:,1]<<4)).to(torch.uint8)
def nib_open(p):
    c=torch.stack((p&15,p>>4),-1)
    return c.flatten().to(torch.int16)-8

class Net(nn.Module):
    def __init__(self,arm,seed,device):
        super().__init__(); self.arm=arm
        self.register_buffer('bytepos',self.pos(device),persistent=False)
        rng=torch.Generator(device='cpu').manual_seed(seed)
        for i,(o,d) in enumerate(SHAPES):
            b=torch.randint(2,(o*d,),generator=rng).bool().to(device)
            p=nib_pack(2*b.to(torch.int16)-1) if arm=='packed4' else bits_pack(b)
            self.register_buffer(f'p{i}',p)
    @staticmethod
    def pos(dev):
        freq=torch.exp(-math.log(10000)*torch.arange(0,D,2,device=dev)/D)
        angles=torch.arange(T,device=dev)[:,None]*freq
        return torch.stack((angles.sin(),angles.cos()),-1).reshape(T,D)*.08
    def packed(self):return tuple(getattr(self,f'p{i}') for i in range(len(SHAPES)))
    def materialize(self,requires=False):
        a=[]
        for p,(o,d) in zip(self.packed(),SHAPES):
            x=(nib_open(p)>0) if self.arm=='packed4' else bits_open(p)
            a.append((2*x.float().view(o,d)-1).requires_grad_(requires))
        return a
    def forward(self,x,w=None):
        w=self.materialize() if w is None else w
        h=.15*F.embedding(x,w[0])+self.bytepos[:x.shape[1]]
        norm=lambda u:F.layer_norm(u,(D,))
        lin=lambda u,p:F.linear(u,p)/math.sqrt(p.shape[-1])
        for i in range(1,len(w),4):
            q,k,v=(a.unflatten(-1,(4,32)).transpose(1,2) for a in lin(norm(h),w[i]).chunk(3,-1))
            z=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).flatten(-2)
            h=h+.25*lin(z,w[i+1])
            h=h+.25*lin(F.gelu(lin(norm(h),w[i+2])),w[i+3])
        return lin(norm(h),w[0])

def load_data(p):
    raw={}
    for s,h in HASH.items():
        raw[s]=(p/f'{s}.bin').read_bytes()
        assert digest(raw[s])==h,s
    return raw
def offsets(n,k):return np.linspace(0,n//(T+1)-1,k,dtype=np.int64)*(T+1)
def pull(tape,i):
    ii=torch.as_tensor(i,device=tape.device)[:,None]+torch.arange(T+1,device=tape.device)[None,:]
    z=tape[ii].long()
    return z[:,:-1],z[:,1:]


def sync(dev):
    if dev=='cuda':torch.cuda.synchronize()
