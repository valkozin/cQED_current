#!/bin/bash
# Run a job list on the local machine with P parallel single-threaded workers (no scheduler).
# Job line: U g omega seed [options of code/run_dmrg_point.py];
# output data/om<omega>[_tJ<tJ>]/dmrg_U<U>_g<g>.json (the suffix only when --tJ is given).
# Points whose output already holds the last requested chi stage are skipped (safe to restart).
#   bash cluster/run_local.sh cluster/jobs_cloud_om0.003.txt 4
JOBS=${1:-cluster/jobs_cloud_om0.003.txt}; P=${2:-4}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
grep -v '^\s*$' "$JOBS" | xargs -P "$P" -L 1 bash -c '
  U=$0; G=$1; OM=$2; SEED=$3; shift 3
  cd '"$ROOT"'/code
  TJ=$(echo "$@" | sed -n "s/.*--tJ \([0-9.]*\).*/\1/p")
  D=../data/om${OM}${TJ:+_tJ$TJ}; mkdir -p $D/logs; OUT=$D/dmrg_U${U}_g${G}.json
  LAST=$(echo "$@" | sed -n "s/.*--chis [0-9,]*,\([0-9]*\).*/\1/p")
  if [ -f "$OUT" ] && grep -q "\"chi\": $LAST," "$OUT" && { ! echo "$@" | grep -q -- --excited || grep -q "\"excited\"" "$OUT"; }; then exit 0; fi
  python run_dmrg_point.py "$U" "$G" "$OM" "$SEED" "$OUT" --sweeps 20 "$@" > $D/logs/U${U}_g${G}.log 2>&1'
