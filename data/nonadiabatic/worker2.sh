#!/bin/bash
# reduced grid (the crossing E_odd = E_even is linear in U to 1e-2 between the nodes used here)
cd /home/user/cQED_current/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
US=0.4,0.6,0.8,0.9,1.0,1.2
GS=0.0,0.2,0.4,0.6,0.8,1.0
run() { python3 run_nonadiabatic.py 10 $1 $2 $3 200 $4 ../data/nonadiabatic/grid_om$1.json >> ../data/nonadiabatic/grid_om$1.log 2>&1; }
( run 0.1 $US $GS 24 ) &
( run 0.3 $US $GS 24 ) &
( run 1.0 $US $GS 20 ) &
( while pgrep -f ring_ephi.py > /dev/null; do sleep 30; done; run 0.03 0.6,0.8,0.9,1.0 0.0,0.4,0.8 36; run 0.01 0.6,0.8,0.9,1.0 0.0,0.4,0.8 48 ) &
wait
