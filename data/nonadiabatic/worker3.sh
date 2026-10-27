#!/bin/bash
# completion and checks: omega=0.01 remaining g; omega=3 (anti-adiabatic check); chi/Nph convergence; U-linearity
cd /home/user/cQED_current/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { python3 run_nonadiabatic.py 10 $1 $2 $3 $4 $5 ../data/nonadiabatic/$6 >> ../data/nonadiabatic/${6%.json}.log 2>&1; }
( run 0.01 0.6,0.8,0.9,1.0 0.2,0.6,1.0 200 48 grid_om0.01.json ) &
( run 3.0 0.4,0.6,0.8,0.9,1.0,1.2 0.0,0.2,0.4,0.6,0.8,1.0 200 16 grid_om3.0.json ) &
( run 0.1 0.9 0.6,1.0 300 24 check_chi300_om0.1.json; run 1.0 0.9 1.0 300 20 check_chi300_om1.0.json; run 0.1 0.9 1.0 200 40 check_Nph40_om0.1.json; run 0.01 0.9 0.8 200 80 check_Nph80_om0.01.json ) &
( run 0.1 0.7,0.85,0.95 0.6 200 24 check_linearity_om0.1.json; run 1.0 0.5,0.7 0.8 200 20 check_linearity_om1.0.json ) &
wait
