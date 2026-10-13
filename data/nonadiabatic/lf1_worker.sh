#!/bin/bash
# Lang-Firsov (omega -> infinity) reference with the vacuum average e^{-g^2/2} on the dressed bond (0,1) only
cd /home/user/cQED_current/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
US=0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2
one() { TJ_PH=$(python3 -c "import math; print('%.6f' % (0.5*math.exp(-$1*$1/2)))") python3 run_nonadiabatic.py 10 1.0 $US 0.0 200 4 ../data/nonadiabatic/lf1_g$1.json > ../data/nonadiabatic/lf1_g$1.log 2>&1; }
( one 0.2; one 1.0 ) &
( one 0.4 ) &
( one 0.6 ) &
( one 0.8 ) &
wait
