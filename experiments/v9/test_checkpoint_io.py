"""Only untrained random weights are saved here; no historical model recreated."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import tempfile,json
from pathlib import Path
import numpy as np
from models import Transformer,Recurrent
from checkpoint_io import save_checkpoint,load_checkpoint,verify_prediction_reload

def main():
    report={'scope':'Random untrained fixtures only. No historical trained checkpoints available.','models':{}}
    with tempfile.TemporaryDirectory(prefix='v9_checkpoint_test_') as tmp:
        for cls in (Transformer,Recurrent):
            m=cls(17,32 if cls is Transformer else 101,79);p=Path(tmp)/(cls.__name__+'.npz')
            manifest=save_checkpoint(m,p,{'fixture':'random untrained; not experimental result'})
            x=np.random.default_rng(73).integers(17,size=(3,64))
            checks=verify_prediction_reload(m,p,x)
            try:save_checkpoint(m,p);raise AssertionError('Overwrite accepted')
            except FileExistsError:checks['overwrite_rejected']=True
            data=p.read_bytes();p.write_bytes(data+b'changed')
            try:load_checkpoint(p);raise AssertionError('Corruption accepted')
            except ValueError:checks['corruption_rejected']=True
            report['models'][cls.__name__]={'checks':checks,'parameter_order':manifest['parameter_order'],'random_fixture_ordered_hash':manifest['ordered_weights_sha256']}
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
