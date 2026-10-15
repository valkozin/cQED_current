#!/bin/bash
cd /home/user/cQED_current/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PHN=-3.1416,-2.7489,-2.3562,-1.9635,-1.5708,-1.1781,-0.7854,-0.3927
PHP=0.0,0.3927,0.7854,1.1781,1.5708,1.9635,2.3562,2.7489,3.1416
( python3 run_so.py uscan 0.5,0.75,1.0,1.25,1.5 0.6 ../data/so/uscan_a0.6.json > ../data/so/uscan_a0.6.log 2>&1;
  python3 run_so.py ephi 2.0 0.6 $PHN ../data/so/ephi_U2_a0.6_neg.json > ../data/so/ephi_U2_a0.6_neg.log 2>&1 ) &
( python3 run_so.py ephi 2.0 0.6 $PHP ../data/so/ephi_U2_a0.6_pos.json > ../data/so/ephi_U2_a0.6_pos.log 2>&1 ) &
( python3 run_so.py ephi 2.0 0.0 $PHP ../data/so/ephi_U2_a0_pos.json > ../data/so/ephi_U2_a0_pos.log 2>&1 ) &
( for d in 0.0 0.1 0.2 0.3 0.4 0.5; do
    EPS_SHIFT=$d python3 run_nonadiabatic.py 10 0.1 1.1 0.0 200 4 ../data/gate/gate_U1.1_d$d.json > ../data/gate/gate_U1.1_d$d.log 2>&1
    EPS_SHIFT=$d python3 run_nonadiabatic.py 10 0.1 1.1 0.6 200 24 ../data/gate/gate_U1.1_d$d.json >> ../data/gate/gate_U1.1_d$d.log 2>&1
  done ) &
wait
