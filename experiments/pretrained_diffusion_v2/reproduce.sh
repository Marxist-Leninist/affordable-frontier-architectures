#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python prepare.py
python checks.py
python extra_checks.py
# Preserve the released frozen protocol and window manifest.
python run.py untouched 0
for seed in 11 22 33; do
  python setup.py "$seed"
  for kind in normal_ce normal_kd direct_ce direct_kd; do
    python run.py "$kind" "$seed"
  done
done
python accounting.py export
python accounting.py normal
python accounting.py block
python inference_accounting.py
python diagnostic.py
python validate.py
python summarize.py
