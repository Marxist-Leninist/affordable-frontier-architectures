#!/bin/bash
# P039-L v2 pipeline on one GTX 1660 SUPER. v2 = rotary positions for every arm (no learned position table);
# diffusion loss = $DL ('mean' or 'elbo', picked on the 2,000-step dev test). Launches every unfinished v2 run when
# the GPU has room for it (free-memory gate) and relaunches a dead run (resumes from latest.pt; at most 3 launches),
# then evaluates, scores AR arms with the P045 protocol, also evaluates the v1 absolute-position attention-AR runs
# (P045-parity reference), times everything on a quiet GPU in 3 rotated rounds and writes the summary.
# Stop: touch STOP_P039L in the work dir (running trainers are left alone).
W=/workspace/p039l_claude_20261122; cd "$W" || exit 1
OUT=runs_v2; DL=${DL:-elbo}
exec >> "$W/logs/pipeline.log" 2>&1
ts() { date -u +%FT%TZ; }
echo "[$(ts)] pipeline start pid $$ out=$OUT diff_loss=$DL"
running() { pgrep -f "[p]039l.py train --arm $1 --seed $2 --out $W/$OUT" > /dev/null; }
need_mb() { case $1 in hybrid_diff) echo 1500;; hybrid_ar) echo 1250;; *) echo 900;; esac; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
launch() {
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -u p039l.py train --arm $1 --seed $2 --out $W/$OUT --pos rope --diff-loss $DL >> logs/v2_train_$1_$2.log 2>&1 &
  echo "[$(ts)] launched $1 $2 pid $! (free before: $3 MiB)"; sleep 25
}
declare -A tries
while true; do
  [ -f STOP_P039L ] && { echo "[$(ts)] STOP file present, exiting"; exit 0; }
  pending=0; alive=0; launchable=0
  for s in 17 43; do for a in hybrid_diff hybrid_ar attn_diff attn_ar; do
    k=${a}_${s}
    [ -f $OUT/$k/done.json ] && continue
    pending=$((pending + 1))
    if running $a $s; then alive=$((alive + 1)); continue; fi
    t=${tries[$k]:-0}
    [ "$t" -ge 3 ] && continue
    launchable=$((launchable + 1))
    f=$(free_mb)
    if [ "${f:-0}" -ge "$(need_mb $a)" ]; then
      tries[$k]=$((t + 1))
      [ -d $OUT/$k ] && echo "[$(ts)] $k is neither running nor done: (re)launch $((t + 1))"
      launch $a $s "$f"; alive=$((alive + 1))
    fi
  done; done
  [ $pending -eq 0 ] && break
  [ $alive -eq 0 ] && [ $launchable -eq 0 ] && { echo "[$(ts)] unfinished runs cannot be relaunched; continuing with what is done"; break; }
  sleep 60
done
echo "[$(ts)] v2 training phase over"
python3 -u p039l.py evaluate --out $W/$OUT >> logs/v2_eval.log 2>&1; echo "[$(ts)] v2 evaluate rc=$?"
python3 -u p039l.py p045 --out $W/$OUT --only attn_ar_17,attn_ar_43,hybrid_ar_17,hybrid_ar_43 >> logs/v2_p045.log 2>&1; echo "[$(ts)] v2 p045 rc=$?"
for i in $(seq 1 240); do
  if [ -f runs/attn_ar_17/done.json ] && [ -f runs/attn_ar_43/done.json ]; then break; fi
  pgrep -f "[p]039l.py train --arm attn_ar --seed" > /dev/null || break; sleep 30
done
python3 -u p039l.py evaluate --out $W/runs --only attn_ar_17,attn_ar_43 >> logs/v1_eval.log 2>&1; echo "[$(ts)] v1 evaluate rc=$?"
python3 -u p039l.py p045 --out $W/runs --only attn_ar_17,attn_ar_43 >> logs/v1_p045.log 2>&1; echo "[$(ts)] v1 p045 rc=$?"
for i in $(seq 1 240); do
  others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
  [ -z "$others" ] && break
  echo "[$(ts)] timing waits for a quiet GPU: $others"; sleep 30
done
python3 -u p039l.py timing --out $W/$OUT --rounds 3 >> logs/v2_timing.log 2>&1; echo "[$(ts)] v2 timing rc=$?"
python3 -u summarize.py _v2 >> logs/v2_summary.log 2>&1; echo "[$(ts)] v2 summary rc=$?"
touch PIPELINE_DONE; echo "[$(ts)] pipeline done"
