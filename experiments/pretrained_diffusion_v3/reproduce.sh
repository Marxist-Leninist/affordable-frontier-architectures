#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python prepare.py
python launch.py checks.py
python launch.py experiment.py untouched 0
for seed in 101 202 303; do
  for kind in normal masked_hybrid_parent masked_hybrid_all masked_local_all direct_hybrid_all direct_local_all; do
    python launch.py experiment.py "$kind" "$seed"
  done
done
python launch.py startup_parity.py
python launch.py accounting.py export
for family in normal masked direct; do python launch.py accounting.py "$family"; done
python launch.py validate.py
python summarize.py
