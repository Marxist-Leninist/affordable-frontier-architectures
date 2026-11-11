"""Supplementary untrained-length inference scaling; not long-context quality."""
import json,time
from pathlib import Path
import numpy as np
import torch
from models import NAT
P=Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
@torch.inference_mode()
def main():
    assert (P/'results.json').exists(), 'Run only after the main experiment finishes.'
    models={}
    for kind in ['transformer','mamba']:
        m=NAT(kind);m.load_state_dict(torch.load(P/'checkpoints'/f'final_{kind}_1.pt',weights_only=True));models[kind]=m.eval()
    rng=np.random.default_rng(1234);rows=[]
    for length in [48,128,512]:
        for batch in [1,8]:
            p=torch.from_numpy(rng.integers(0,256,(batch,length-16),dtype=np.int64))
            for m in models.values():
                for _ in range(3):m(p).argmax(-1)
            for trial in range(30):
                for kind in rng.permutation(list(models)):
                    st=time.perf_counter();models[kind](p).argmax(-1);sec=time.perf_counter()-st
                    rows.append({'kind':kind,'total_positions':length,'batch':batch,'trial':trial,'ms':sec*1000})
    out={'scope':'Seed1, CPU FP32, pure forward+16-output argmax. Length128/512 not trained or quality-tested. Not GPU speed. Randomized family order.','trials':rows}
    (P/'length_timings.json').write_text(json.dumps(out,indent=2)+'\n')
    for length in [48,128,512]:
        for batch in [1,8]:
            print(length,batch,{k:round(float(np.median([r['ms'] for r in rows if r['kind']==k and r['total_positions']==length and r['batch']==batch])),3) for k in models})
if __name__=='__main__':main()
