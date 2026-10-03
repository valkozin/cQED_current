import sys; sys.path.insert(0,'..')
sys.path.insert(0, '.')
import numpy as np, time
from dmrg_ring import *
from bench_params import nb_params
from model import DEFAULTS, electron_energy
for chi in [256, 512, 1024]:
    m = RingCavityModel(nb_params(0.0, 0.0))
    t0=time.time(); E, psi = run_dmrg(m, ['up'] + ['up', 'down'] * 5, chi=chi, sweeps=12)
    print(chi, E, E-electron_energy(DEFAULTS,1.0,0.0), time.time()-t0, flush=True)
