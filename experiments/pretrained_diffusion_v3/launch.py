"""Initialize CPU/NumPy before Torch/Transformers, matching the completed v2 run.

Runtime-only launcher. No data, model, optimizer, loss or sampling change.
"""
import os,sys,runpy
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy
P=Path(__file__).parent
allowed={'checks.py','experiment.py','accounting.py','validate.py','startup_parity.py'}
script=sys.argv[1]
if script not in allowed:raise ValueError('Unknown experiment entry point')
sys.argv=[str(P/script)]+sys.argv[2:]
runpy.run_path(str(P/script),run_name='__main__')
