#!/bin/bash
# End-to-end gate for the lctx hotmod v1.1: the REAL trainer source (agillm128b.py sha 1241dead..., the live 888B
# trainer module) on CPU, tiny preset, blocks under non-reentrant checkpointing (aaa_force_ckpt helper),
# TERNTF_ATTN_T = sqrt(hd).
# A no mod | B v1.1 at launch | B2 resume | S v1 at launch, swapped to v1.1 mid-run (the live upgrade path) |
# P PAUSE mid-run pins the gate | D kill switch | E kill -9 inside a step: the relaunch installs again (first strike),
# a second kill -9 inside a step: the next relaunch refuses | F marker deleted: installs and resumes.
D=/root/claude_lctx_20261116/e2e
cd "$D" || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TERNTF_ATTN_T=8.0
MOD=/root/claude_lctx_20261116/hotmod/zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz_lctx.persist.py
MODV1=/root/claude_lctx_20261116/hotmod_v1/v1_ffba5188.persist.py
NAME=$(basename $MOD)
TR="nice -n 10 python3 -u $D/agillm128b.py train --preset tiny --batch 2 --ctx 64 --data file:$D/agillm128b.py --save-every 1 --log-every 1 --seed 0 --save-min 0"
put() { mkdir -p "$2/hotmods" && cp "$1" "$2/hotmods/.lctx.tmp" && mv "$2/hotmods/.lctx.tmp" "$2/hotmods/$NAME"; }
put_ck() { mkdir -p "$1/hotmods" && cp $D/aaa_force_ckpt.persist.py "$1/hotmods/"; }
wait_log() { local t=0; while ! grep -q -E "$2" "$1" 2>/dev/null; do sleep 0.2; t=$((t+1)); [ $t -gt $(( $3 * 5 )) ] && { echo "TIMEOUT waiting for $2 in $1"; return 1; }; done; return 0; }
tpid() { pgrep -f "agillm128b.py train .*--run $D/$1" | head -1; }
kill_in_step() {  # kill -9 the trainer of run $1 while its crash marker is present
  for i in $(seq 1 1000); do [ -f "$D/$1/lctx.inflight" ] && break; sleep 0.01; done
  cat "$D/$1/lctx.inflight" 2>&1 | cut -c1-160; echo
  local p=$(tpid $1); kill -9 $p; sleep 1; tpid $1 >/dev/null && echo "STILL RUNNING" || echo "killed pid $p"; }
rm -rf runA runB runS runP runD runE ./*.e2e.log
sha256sum agillm128b.py $MOD $MODV1 | cut -c1-80
python3 -V
echo "== A baseline (checkpointed blocks, no mod), 16 steps"; put_ck runA; timeout 900 $TR --run $D/runA --steps 16 > A.e2e.log 2>&1; echo "rc=$?"
echo "== B v1.1 at launch, 16 steps"; put_ck runB; put $MOD runB; timeout 900 $TR --run $D/runB --steps 16 > B.e2e.log 2>&1; echo "rc=$?"
echo "== B2 resume, 6 steps"; timeout 900 $TR --run $D/runB --resume --steps 6 > B2.e2e.log 2>&1; echo "rc=$?"
echo "== S v1 at launch, v1.1 swapped in after step 6, 14 steps"; put_ck runS; put $MODV1 runS
$TR --run $D/runS --steps 14 > S.e2e.log 2>&1 & ps_=$!
wait_log S.e2e.log '^step 6 ' 600 && put $MOD runS
wait $ps_; echo "rc=$?"
echo "== P PAUSE after step 5 for ~8 s, then on to 10 steps"; put_ck runP; put $MOD runP
$TR --run $D/runP --steps 10 > P.e2e.log 2>&1 & pp=$!
wait_log P.e2e.log '^step 5 ' 600 && touch runP/PAUSE
wait_log P.e2e.log 'pinned the gate' 300; sleep 3; rm -f runP/PAUSE
wait $pp; echo "rc=$?"
echo "== D kill switch after step 6, 12 steps"; put_ck runD; put $MOD runD
$TR --run $D/runD --steps 12 > D.e2e.log 2>&1 & pd=$!
wait_log D.e2e.log '^step 6 ' 600 && touch runD/hotmods/lctx.off
wait $pd; echo "rc=$?"
echo "== E two strikes"; put_ck runE; put $MOD runE
$TR --run $D/runE --steps 0 > E1.e2e.log 2>&1 &
wait_log E1.e2e.log '^step 4 ' 600; kill_in_step runE
$TR --run $D/runE --resume --steps 0 > E2.e2e.log 2>&1 &
# kill E2 inside its FIRST step (before any clean backward resets the strike): its own marker says post_crash
for i in $(seq 1 6000); do grep -q '"post_crash": true' runE/lctx.inflight 2>/dev/null && break; sleep 0.05; done
kill_in_step runE
timeout 600 $TR --run $D/runE --resume --steps 3 > E3.e2e.log 2>&1; echo "E3 rc=$?"
echo "== F marker deleted: installs again, 4 steps"; rm -f runE/lctx.inflight
timeout 600 $TR --run $D/runE --resume --steps 4 > F.e2e.log 2>&1; echo "rc=$?"
echo "== results"
python3 /root/claude_lctx_20261116/tests/e2e_check.py $D
