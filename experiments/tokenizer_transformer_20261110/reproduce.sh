#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python prepare.py
for seed in 1 2 3; do
 for tok in gpt2 qwen3 gpt2_padded; do
  python train.py --tokenizer "$tok" --seed "$seed" --budget 240
 done
done
python validate.py
python analyze.py
python replicate_eval.py
