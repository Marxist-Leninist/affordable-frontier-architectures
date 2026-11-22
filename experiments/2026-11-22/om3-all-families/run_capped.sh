#!/bin/bash
# Run a command at nice 19 on one CPU thread with no GPU; kill it if its resident memory passes 2 GiB or it runs 15 min.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
nice -n 19 "$@" &
pid=$!
start=$(date +%s)
while kill -0 $pid 2>/dev/null; do
  rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null || echo 0)
  if [ "${rss:-0}" -gt 2097152 ]; then echo "RUN_CAPPED: killed at ${rss} kB RSS" >&2; kill -9 $pid; wait $pid; exit 137; fi
  if [ $(( $(date +%s) - start )) -gt 900 ]; then echo "RUN_CAPPED: killed after 900 s" >&2; kill -9 $pid; wait $pid; exit 124; fi
  sleep 0.5
done
wait $pid
