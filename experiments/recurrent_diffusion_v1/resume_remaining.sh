#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for seed in 1 2 3; do
  for kind in bptt diffusion_weighted diffusion_ce transformer; do
    if [ "$kind" = bptt ] && [ -f "bptt_${seed}_equal_data.json" ]; then continue; fi
    if [ -f "completed_${kind}_${seed}.json" ]; then continue; fi
    python -u run_one.py "$kind" "$seed"
  done
done
