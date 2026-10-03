import numpy as np, time
from dmrg_ring import *
from model import DEFAULTS, electron_energy
import logging; logging.basicConfig(level=logging.WARNING)
P = DEFAULTS; Nr = 1 + P['NumSCsites']
def nb_params(g=0.0, phi=0.0, Nph=4):
    return dict(Nr=Nr, t=[P['tR0']] + [P['t0']] * (Nr - 2) + [P['tL0']], tJ=P['tR0'],
                eps_up=[P['ED0']] + [0] * (Nr - 1), eps_dn=[P['ED0'] + P['DeltaZ0']] + [0] * (Nr - 1),
                U=[0] * Nr, Delta=[0] + [P['Delta0']] * (Nr - 1), omega=P['omegaR'], g=g, phi_cl=phi, Nph=Nph)
# NB: bond Nr-1 (site Nr-1 -> 0) uses t[Nr-1]=tL ; bonds 1..Nr-2 are SC bonds t0
for phi in [0.0, 1.0, np.pi]:
    m = RingCavityModel(nb_params(0.0, phi))
    res = []
    for st in (['up'] + ['up', 'down'] * 5, ['empty'] + ['up', 'down'] * 5, ['down'] + ['up', 'down'] * 5):
        t0 = time.time(); E, psi = run_dmrg(m, st, chi=200); res.append(E)
    o = observables(m, psi)
    print('phi=%.3f  DMRG E(odd,Sz+)=%.8f E(even)=%.8f E(odd,Sz-)=%.8f  BdG=%.8f  (%.1fs)' %
          (phi, *res, electron_energy(P, 1.0, phi), time.time() - t0), flush=True)
