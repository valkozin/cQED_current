#!/bin/bash
# One-time setup on the cluster login node (sciCORE): Python venv with the pinned TeNPy version.
#   bash cluster/setup_env.sh
# Adjust the module line to a Python >= 3.10 available on the cluster (`module spider Python`).
set -e
module purge 2>/dev/null || true
module load Python 2>/dev/null || true
python3 -m venv "$HOME/venvs/cqed"
source "$HOME/venvs/cqed/bin/activate"
pip install --upgrade pip
pip install numpy scipy matplotlib physics-tenpy==1.1.1
python -c "import tenpy; print('tenpy', tenpy.__version__)"
