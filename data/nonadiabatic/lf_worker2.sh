#!/bin/bash
cd /home/user/cQED_current/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
US=0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2
for g in 0.6 0.8 1.0; do
  tJ=$(python3 -c "import math; print('%.6f' % (0.5*math.exp(-$g*$g/2)))")
  python3 ring_ephi.py 10 $US $tJ 0.0 200 ../data/nonadiabatic/lf_g$g.json > ../data/nonadiabatic/lf_g$g.log 2>&1
done
