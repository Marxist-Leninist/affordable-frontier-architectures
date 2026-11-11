"""Tiny synthetic cache equivalence/corruption test; no corpus cache changed."""
import json,tempfile
from pathlib import Path
import numpy as np
import torch
import data_long as d
P=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
 d.ROOT=Path(tmp)
 source=bytes(np.random.default_rng(7).integers(32,127,128*20,dtype=np.uint8));train=b'other training bytes'
 d.raw=lambda s:train if s=='train' else source
 direct=d.eval_windows(d.articles(source),train,limit=7)
 cold=d.text_sets('validation',7);hot=d.text_sets('validation',7)
 for cached in [cold,hot]:
  assert torch.equal(direct[0],cached[0]) and torch.equal(direct[1],cached[1]) and np.array_equal(direct[2],cached[2]) and direct[3]==cached[3]
 cache=next((d.ROOT/'data_cache').glob('*.npz'))
 with np.load(cache,allow_pickle=False) as z:payload={k:z[k] for k in z.files}
 payload['p'][0,0]+=1
 with cache.open('wb') as f:np.savez_compressed(f,**payload)
 rejected=False
 try:d.text_sets('validation',7)
 except AssertionError as e:rejected='cache integrity mismatch' in str(e)
 assert rejected
r={'passed':True,'cold_equals_uncached':True,'hot_equals_uncached':True,'same_evenly_spaced_limit_and_metadata':True,'corrupt_token_cache_rejected':True,'scope':'Tiny synthetic arrays in temporary directory; no real corpus cache changed.','limitation':'Cache content digest verifies concatenated prefix/target arrays, not groups/metadata separately; atomic locally generated cache makes this acceptable for current use, but whole-cache digest would harden future document-bootstrap provenance.'}
(P/'cache_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
