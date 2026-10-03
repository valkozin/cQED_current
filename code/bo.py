"""Born-Oppenheimer (adiabatic) treatment: the photon moves in the exact electronic
ground-state energy landscape V(phi) = E_el(phi_cl + g X).  Exact in the limit
omega << Andreev gap; it keeps all harmonics of V and all photon correlations."""
import numpy as np
from scipy.linalg import eigh
from model import DEFAULTS, electron_energy


def harmonics(p, M=16, npts=128):
    ph = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    E = np.array([electron_energy(p, 1.0, x) for x in ph])
    c = np.fft.rfft(E) / npts
    c[1:] *= 2
    return c[:M + 1]  # E(phi) = Re sum_m c_m e^{i m phi}


def V_of(c, phi):
    m = np.arange(len(c))
    return np.real(np.exp(1j * np.outer(phi, m)) @ c)


def dV_of(c, phi):
    m = np.arange(len(c))
    return np.real((1j * m) * np.exp(1j * np.outer(phi, m)) @ c)


def solve(c, g, omega, phi_cl=0.0, Nphot=200, nstates=4):
    n = np.arange(Nphot)
    X = np.diag(np.sqrt(n[1:]), 1) + np.diag(np.sqrt(n[1:]), -1)
    x, U = eigh(X)  # DVR in the eigenbasis of X
    H = omega * np.diag(n) + (U * V_of(c, phi_cl + g * x)) @ U.T
    E, psi = eigh(H)
    psi = psi[:, :nstates]
    # observables in each state
    cur = (U * (-dV_of(c, phi_cl + g * x))) @ U.T   # I = -dE/dphi
    obs = {}
    obs['E'] = E[:nstates]
    obs['I'] = np.einsum('ik,ij,jk->k', psi, cur, psi)
    obs['I2'] = np.einsum('ik,ij,jk->k', psi, cur @ cur, psi)
    obs['X'] = np.einsum('ik,ij,jk->k', psi, X, psi)
    obs['X2'] = np.einsum('ik,ij,jk->k', psi, X @ X, psi)
    obs['n'] = np.einsum('ik,i,ik->k', psi, n.astype(float), psi)
    obs['Pdist'] = (x, (U.T @ psi[:, 0]) ** 2)
    return obs


if __name__ == '__main__':
    p = dict(DEFAULTS)
    c = harmonics(p)
    np.save('../data/harmonics_notebook_model.npy', c)
    print('harmonics', np.round(c.real[:5], 6))
    rows = []
    for g in np.arange(0, 1.2001, 0.02):
        o0 = solve(c, g, p['omegaR'], 0.0)
        o1 = solve(c, g, p['omegaR'], p['phiClassical'])
        rows.append([g, o0['E'][0], o0['E'][1] - o0['E'][0], o0['I2'][0], o0['X2'][0], o0['n'][0],
                     o1['I'][0], o1['X'][0], o1['E'][0]])
        print('g=%.2f E0=%.6f split=%.3e sqrt<I2>=%.4f <X2>=%.3f n=%.3f | seed: I=%.2e X=%.3e'
              % tuple(rows[-1][:6] + rows[-1][6:8]), flush=True)
    np.save('../data/bo_notebook_model.npy', np.array(rows))
