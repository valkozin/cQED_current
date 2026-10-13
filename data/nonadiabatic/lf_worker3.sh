#!/bin/bash
cd /home/user/cQED_current/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
tJ=$(python3 -c "import math; print('%.6f' % (0.5*math.exp(-0.5)))")
python3 ring_ephi.py 10 0.1,0.2 $tJ 0.0 200 ../data/nonadiabatic/lf_g1.0_ext.json > ../data/nonadiabatic/lf_g1.0_ext.log 2>&1
python3 - <<'PY'
import json
a=json.load(open('../data/nonadiabatic/lf_g1.0.json')); b=json.load(open('../data/nonadiabatic/lf_g1.0_ext.json'))
json.dump(a+b, open('../data/nonadiabatic/lf_g1.0.json','w'))
PY
rm -f ../data/nonadiabatic/lf_g1.0_ext.json
