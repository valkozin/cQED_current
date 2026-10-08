#!/bin/bash
# Run a job list on the local machine with P parallel single-threaded workers (no scheduler).
# Points whose output already holds all requested chi stages are skipped (safe to restart).
#   bash cluster/run_local.sh cluster/jobs_cloud_om0.01.txt 4
JOBS=${1:-cluster/jobs_cloud_om0.01.txt}; P=${2:-4}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/data/om0.01/logs"
grep -v '^\s*$' "$JOBS" | xargs -P "$P" -L 1 bash -c '
  U=$0; G=$1; shift 1
  OUT=../data/om0.01/dmrg_U${U}_g${G}.json
  cd '"$ROOT"'/code
  if [ -f "$OUT" ] && grep -q "\"excited\"\|\"chi\": 128" "$OUT" && ! echo "$@" | grep -q -- "--excited"; then exit 0; fi
  if [ -f "$OUT" ] && grep -q "\"excited\"" "$OUT"; then exit 0; fi
  python run_dmrg_point.py "$U" "$G" 0.01 1e-3 "$OUT" --Nph 20 --sweeps 20 "$@" > ../data/om0.01/logs/U${U}_g${G}.log 2>&1'
