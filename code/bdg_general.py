"""BdG for the general ring used in the DMRG code (non-interacting, spin-diagonal)."""
import numpy as np
from scipy.linalg import eigh


def ring_h(p, eps, phi):
    Nr = p['Nr']
    h = np.zeros((Nr, Nr), complex)
    h[np.arange(Nr), np.arange(Nr)] = eps
    for r in range(1, Nr):
        j = (r + 1) % Nr
        h[j, r] += -p['t'][r]; h[r, j] += -p['t'][r]
    h[1, 0] += -p['tJ'] * np.exp(1j * phi); h[0, 1] += -p['tJ'] * np.exp(-1j * phi)
    return h


def bdg_energy(p, phi, return_spec=False):
    hu = ring_h(p, np.array(p['eps_up'], float), phi)
    hd = ring_h(p, np.array(p['eps_dn'], float), phi)
    D = np.diag(np.array(p['Delta'], float))
    H = np.block([[hu, D], [D, -hd.conj()]])
    v = eigh(H, eigvals_only=True)
    E = v[v < 0].sum() + np.trace(hd).real
    return (E, v) if return_spec else E


def harmonics(p, M=12, npts=96):
    ph = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    E = np.array([bdg_energy(p, x) for x in ph])
    c = np.fft.rfft(E) / npts
    c[1:] *= 2
    return c[:M + 1]


def sc_ring(Nr, t=1.0, Delta=0.5, tJ=0.5, eps_dot=None, Zdot=0.0, Udot=0.0, dot=False):
    """All-SC ring with weak bond tJ between sites 0,1; if dot: site 0 is a normal dot
    (Delta=0, eps_dot, Zeeman Zdot, Hubbard Udot) coupled by tJ to both neighbours."""
    tt = [tJ] + [t] * (Nr - 1)
    Dl = [Delta] * Nr
    eu = [0.0] * Nr; ed = [0.0] * Nr; U = [0.0] * Nr
    if dot:
        tt[Nr - 1] = tJ
        Dl[0] = 0.0
        e0 = -Udot / 2 if eps_dot is None else eps_dot
        eu[0] = e0; ed[0] = e0 + Zdot; U[0] = Udot
    return dict(Nr=Nr, t=tt, tJ=tJ, eps_up=eu, eps_dn=ed, U=U, Delta=Dl)
