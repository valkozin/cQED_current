#!/bin/bash
# usage: worker.sh "sector:U sector:U ..."
cd /home/user/cQED_current/code
export OMP_NUM_THREADS=1
for job in $1; do
  sec=${job%%:*}; U=${job##*:}
  python3 run_full_grid.py 10 $U 0.5 7.39e-4 1e-3 $sec 0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.32,0.34,0.36,0.38,0.40 300 50 ../data/full/U${U}_${sec}.json 40 > ../data/full/U${U}_${sec}.log 2>&1
done
