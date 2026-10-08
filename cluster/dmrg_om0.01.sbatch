#!/bin/bash
# Full electron+photon DMRG on the coarse (U, g) grid at hbar*omega = 0.01 t, seed phi_cl = 1e-3.
# One array task per line of cluster/jobs_om0.01.txt (U g [--excited]); output: data/om0.01/dmrg_U<U>_g<g>.json
# Submit from the repository root:   sbatch cluster/dmrg_om0.01.sbatch
#SBATCH --job-name=cqed_om0.01
#SBATCH --array=1-96%48
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --qos=6hours
#SBATCH --output=cluster/logs/%x_%a.out
module purge 2>/dev/null || true
module load Python 2>/dev/null || true
source "$HOME/venvs/cqed/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
read -r U G EXTRA < <(sed -n "${SLURM_ARRAY_TASK_ID}p" cluster/jobs_om0.01.txt)
OUT=$PWD/data/om0.01/dmrg_U${U}_g${G}.json
cd code
python run_dmrg_point.py "$U" "$G" 0.01 1e-3 "$OUT" --chis 256,512,768 --Nph 20 --sweeps 30 $EXTRA
