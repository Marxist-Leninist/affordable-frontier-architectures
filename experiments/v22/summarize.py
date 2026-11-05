from run import *
def main():
 r=json.loads((P/'results.json').read_text());a=np.load(P/'probabilities.npz');items=json.loads((P/'items.json').read_text());docs=np.array([x['doc'] for x in items]);unique=np.unique(docs);rng=np.random.default_rng(2026110522);out=[]
 for kind in KINDS:
  w=next(x['weight'] for x in r['rows'] if x['kind']==kind)
  for hybrid in (False,True):
   losses={}
   for prec in ('fp32','int8'):
    ps=[a[f'{kind}_{s}_{prec}'] for s in SEEDS]
    if hybrid:ps=[np.where(a['active'],(1-w)*p+w*a['copy'],p) for p in ps]
    losses[prec]=np.mean([-np.log2(p).mean(1) for p in ps],axis=0)
   delta=losses['int8']-losses['fp32'];sums=np.array([delta[docs==d].sum() for d in unique]);counts=np.array([(docs==d).sum() for d in unique]);indices=rng.integers(0,len(unique),(5000,len(unique)));bs=sums[indices].sum(1)/counts[indices].sum(1)
   out.append(dict(kind=kind,hybrid=hybrid,fp32_bpb=float(losses['fp32'].mean()),int8_bpb=float(losses['int8'].mean()),delta_int8_minus_fp32=float(delta.mean()),paired_document_bootstrap95=np.quantile(bs,[.025,.975]).tolist(),descriptive_abs_delta_le_002=bool(abs(delta.mean())<=.02)))
 write('summary.json',dict(quality=out,bootstrap='5000 paired document clusters; average checkpoint seeds before sampling; token-weighted same-sample exploratory intervals, not population guarantees. Fixed seed2026110522.'))
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
