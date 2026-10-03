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
