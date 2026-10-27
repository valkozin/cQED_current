#!/bin/bash
cd /home/user/cQED_current/code
export OMP_NUM_THREADS=1
G1=0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.18,0.20
python3 run_full_grid.py 10 2.0 0.5 7.39e-4 1e-3 odd $G1 400 50 ../data/full_chi400/U2.0_odd.json 40 > ../data/full_chi400/U2.0_odd.log 2>&1
python3 run_full_grid.py 10 2.0 0.5 7.39e-4 1e-4 odd $G1 300 50 ../data/full_seed1e-4/U2.0_odd.json 40 > ../data/full_seed1e-4/U2.0_odd.log 2>&1
python3 run_full_grid.py 10 1.25 0.5 7.39e-4 1e-3 odd 0.12,0.14 300 80 ../data/full_chi400/U1.25_Nph80.json 40 > ../data/full_chi400/U1.25_Nph80.log 2>&1
