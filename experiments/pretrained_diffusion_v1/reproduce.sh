#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python prepare.py
python checks.py
# Use frozen protocol in release; do not overwrite it to conceal source changes.
python run.py untouched 0
for seed in 1 2 3; do
  python run.py normal_finetune "$seed"
  python run.py pretrained_diffusion "$seed"
done
python diagnostic.py
python accounting.py
python generate.py
python validate.py
python summarize.py
