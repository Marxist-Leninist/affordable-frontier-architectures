"""Future-run checkpoint serialization. Does not reconstruct unavailable weights.
Hashes preserve model.p insertion order, matching v9 result hashes exactly.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from models import Transformer,Recurrent

def ordered_weight_sha256(model):
    return hashlib.sha256(b''.join(value.tobytes(order='C') for value in model.p.values())).hexdigest()

def save_checkpoint(model,path,metadata=None):
    path=Path(path);manifest_path=path.with_suffix(path.suffix+'.json')
    if path.exists() or manifest_path.exists():raise FileExistsError('Refusing to overwrite checkpoint')
    kind='transformer' if isinstance(model,Transformer) else 'recurrent' if isinstance(model,Recurrent) else None
    if kind is None:raise TypeError(type(model))
    manifest={'format':'v9-numpy-model-weights-v1','kind':kind,'vocab':model.vocab,'width':model.d,'parameters':model.count(),'parameter_order':list(model.p),'parameter_shapes':{k:list(v.shape) for k,v in model.p.items()},'parameter_dtypes':{k:str(v.dtype) for k,v in model.p.items()},'ordered_weights_sha256':ordered_weight_sha256(model),'metadata':metadata or {},'contents':'Model weights only; no optimizer/RNG state, so not an exact training-resume checkpoint'}
    with path.open('xb') as f:np.savez(f,**model.p)
    manifest['file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with manifest_path.open('x') as f:f.write(json.dumps(manifest,indent=2)+'\n')
    loaded=load_checkpoint(path)
    assert ordered_weight_sha256(loaded)==manifest['ordered_weights_sha256']
    return manifest

def load_checkpoint(path):
    path=Path(path);manifest=json.loads(path.with_suffix(path.suffix+'.json').read_text())
    if manifest['format']!='v9-numpy-model-weights-v1':raise ValueError('Unknown format')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['file_sha256']:raise ValueError('File hash mismatch')
    cls={'transformer':Transformer,'recurrent':Recurrent}[manifest['kind']]
    model=cls(manifest['vocab'],manifest['width'],0)
    if list(model.p)!=manifest['parameter_order']:raise ValueError('Parameter order mismatch')
    with np.load(path,allow_pickle=False) as arrays:
        if set(arrays.files)!=set(model.p):raise ValueError('Parameter set mismatch')
        for k,v in model.p.items():
            a=arrays[k]
            if a.shape!=v.shape or list(a.shape)!=manifest['parameter_shapes'][k] or a.dtype!=v.dtype or str(a.dtype)!=manifest['parameter_dtypes'][k]:raise ValueError('Parameter shape/dtype mismatch')
            model.p[k]=a.copy()
    if ordered_weight_sha256(model)!=manifest['ordered_weights_sha256']:raise ValueError('Ordered weights hash mismatch')
    return model

def verify_prediction_reload(original,path,x):
    loaded=load_checkpoint(path)
    before=original.loss(x,None);after=loaded.loss(x,None)
    if not np.array_equal(before,after):raise AssertionError('Prediction reload differs')
    return {'ordered_hash_matches':ordered_weight_sha256(original)==ordered_weight_sha256(loaded),'predictions_bitwise_equal':True,'max_absolute_prediction_difference':float(np.max(np.abs(before-after)))}
