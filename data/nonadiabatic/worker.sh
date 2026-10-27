#!/bin/bash
# non-adiabatic grid: 3 workers, one omega list each; U and g lists shared
cd "$(dirname "$0")/../../code"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
US=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2
GS=0.0,0.2,0.4,0.6,0.8,1.0
run() { python3 run_nonadiabatic.py 10 $1 $US $GS 200 $2 ../data/nonadiabatic/grid_om$1.json > ../data/nonadiabatic/grid_om$1.log 2>&1; }
(run 0.1 24; run 0.03 36) &
(run 0.3 24; run 1.0 20) &
(run 0.01 48) &
wait
