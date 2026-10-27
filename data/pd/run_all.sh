#!/bin/bash
# full-DMRG (U,g) grid; 4 workers x 1 thread. Resumable (each json is updated per point).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/user/cQED_current/code
G=0.06,0.10,0.13,0.16,0.20,0.25,0.30,0.40
run(){ python run_dmrg_phase_diagram.py "$@" ; }
( run 1.0 odd $G 256 50 ../data/pd/U1.0.json   >> ../data/pd/U1.0.log 2>&1
  run 6.0 odd 0.10,0.20,0.25,0.30,0.35,0.40 256 50 ../data/pd/U6.0.json >> ../data/pd/U6.0.log 2>&1 ) &
( run 1.5 odd $G 256 50 ../data/pd/U1.5.json   >> ../data/pd/U1.5.log 2>&1
  run 8.0 odd 0.10,0.20,0.30,0.40 256 50 ../data/pd/U8.0.json >> ../data/pd/U8.0.log 2>&1 ) &
( run 3.0 odd $G 256 50 ../data/pd/U3.0.json   >> ../data/pd/U3.0.log 2>&1
  run 2.0 odd 0.06,0.25,0.30,0.40 256 50 ../data/pd/U2.0.json >> ../data/pd/U2.0.log 2>&1 ) &
( run 4.0 odd $G 256 50 ../data/pd/U4.0.json   >> ../data/pd/U4.0.log 2>&1
  run 0.5 even 0.10,0.20,0.40 256 50 ../data/pd/U0.5.json >> ../data/pd/U0.5.log 2>&1 ) &
wait
