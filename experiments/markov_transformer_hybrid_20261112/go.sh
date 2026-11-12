#!/bin/bash
# go.sh <cmd> [args] -- entry points on GETH for the Markov chain + transformer experiment.
# Everything runs from this folder, single-threaded per process, at low priority.
cd /root/markov_xfmr_20261112 || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4
mkdir -p logs runs data
case "$1" in
  prep)    nohup nice -n 10 python3 prep_data.py --name "$2" > "logs/prep_$2.log" 2>&1 &
           echo "started prep $2" ;;
  blocks)  nohup nice -n 10 python3 prep_blocks.py --name "$2" > "logs/blocks_$2.log" 2>&1 &
           echo "started blocks $2" ;;
  tune)    nohup nice -n 15 python3 tune_discounts.py --name "$2" > "logs/tune_$2.log" 2>&1 &
           echo "started tune $2" ;;
  test)    nice -n 10 python3 tests/test_all.py > logs/test.log 2>&1
           echo "exit=$?"; tail -n 45 logs/test.log ;;
  jobs)    shift; python3 make_jobs.py "$@" ;;
  grid)    nohup setsid nice -n 19 python3 launch_grid.py "$2" "$3" $4 > "logs/grid_$(basename "$2" .json).out" 2>&1 &
           echo "started grid $2 with $3 slots ${4:+after $4}" ;;
  ugrid)   # like grid, but in its own systemd unit. A shell started through the SG exec route lives in geth-mcp.service,
           # and a restart of that service kills everything in it (that is how wave w4 died at 06:58Z on 12 Nov 2026).
           # The unit is capped (memory, CPU weight, idle IO) so it can never push GETH into trouble.
           t="$(basename "$2" .json)"; u="markov-grid-$(echo "$t" | tr '_' '-')"
           systemd-run --unit="$u" --collect --quiet -p Nice=19 -p CPUWeight=20 -p IOSchedulingClass=idle \
             -p MemoryHigh=6G -p MemoryMax=8G -p MemorySwapMax=0 -p WorkingDirectory=/root/markov_xfmr_20261112 \
             -p StandardOutput=append:/root/markov_xfmr_20261112/logs/grid_$t.out -p StandardError=inherit \
             --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
             /usr/bin/python3 launch_grid.py "$2" "$3" $4
           echo "started unit $u ($2, $3 slots${4:+, after $4}); stop: systemctl stop $u  or  touch STOP" ;;
  bench)   shift; nohup nice -n 5 python3 bench_speed.py "$@" > logs/bench.log 2>&1 &
           echo "started bench" ;;
  one)     shift; nohup nice -n 19 python3 train.py "$@" > logs/one.log 2>&1 &
           echo "started one run" ;;
  sum)     shift; python3 summarize.py "$@" ;;
  status)  for f in grid_status_*.json; do [ -f "$f" ] && cat "$f" && echo; done
           uptime | sed 's/.*load/load/'; free -m | sed -n 2p ;;
  *)       echo "usage: go.sh prep|test|jobs|grid|bench|one|sum|status" ;;
esac
