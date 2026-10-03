"""Common definitions: SC ring (1 dot + N_SC superconducting sites) threaded by cavity flux.

Exact Python port of Optimised_SC_ring_with_JJ_cavity_mean_field.nb (Mathematica).
Site 0 = dot (normal site), sites 1..N_SC = superconducting chain.  The ring closes
via dot-site1 (tR, carries Peierls phase) and dot-site N_SC (tL).
Units: energies in the same units as the notebook (t0 = 2).
"""
import numpy as np
from scipy.linalg import expm, eigh

DEFAULTS = dict(ED0=-0.3, DeltaZ0=4.0, t0=2.0, Delta0=0.6, tR0=0.5, tL0=0.5,
                omegaR=0.02, phiClassical=1e-3, NumSCsites=10)


def photon_ops(Nphot):
    a = np.diag(np.sqrt(np.arange(1, Nphot)), 1).astype(complex)
    adag = a.conj().T
    return a, adag, adag @ a, a + adag


def h_spin(p, eps_dot):
    Ns = 1 + p['NumSCsites']
    h = np.zeros((Ns, Ns), complex)
    h[0, 0] = eps_dot
    for j in range(1, Ns - 1):
        h[j, j + 1] = h[j + 1, j] = -p['t0']
    h[0, Ns - 1] = h[Ns - 1, 0] = -p['tL0']
    return h


def hbdg(p, alpha, phi=None):
    """BdG matrix in basis (c_up, c_dn^dagger), right bond renormalised by alpha=<D>."""
    phi = p['phiClassical'] if phi is None else phi
    Ns = 1 + p['NumSCsites']
    hu = h_spin(p, p['ED0'])
    hd = h_spin(p, p['ED0'] + p['DeltaZ0'])
    for h in (hu, hd):
        h[0, 1] = -p['tR0'] * np.exp(-1j * phi) * np.conj(alpha)
        h[1, 0] = -p['tR0'] * np.exp(1j * phi) * alpha
    D = np.diag([0.0] + [p['Delta0']] * (Ns - 1))
    return np.block([[hu, D], [D, -hd.conj()]])


def electron_energy(p, alpha, phi=None):
    vals = eigh(hbdg(p, alpha, phi), eigvals_only=True)
    return vals[vals < 0].sum() + p['ED0'] + p['DeltaZ0']
