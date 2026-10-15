#!/bin/bash
cd /home/user/cQED_current/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
read OM G < ../data/so/cat_params.txt
( python3 run_so.py cat 2.0 0.6 $OM $G 40 ../data/so/cat_U2_a0.6_beta2.json > ../data/so/cat_U2_a0.6_beta2.log 2>&1 ) &
( python3 run_so.py switch 1.2,1.3 0.6 0.1 0.0,0.6 24 ../data/so/switch_a0.6_A.json > ../data/so/switch_a0.6_A.log 2>&1 ) &
( python3 run_so.py switch 1.4,1.5 0.6 0.1 0.0,0.6 24 ../data/so/switch_a0.6_B.json > ../data/so/switch_a0.6_B.log 2>&1 ) &
( python3 run_so.py switch 1.0,1.1 0.6 0.1 0.0,0.6 24 ../data/so/switch_a0.6_C.json > ../data/so/switch_a0.6_C.log 2>&1 ) &
wait
