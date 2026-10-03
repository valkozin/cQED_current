"""The notebook's mean-field (electron x photon product state) algorithm for the general ring model."""
import numpy as np
from scipy.linalg import expm, eigh
from bdg_general import ring_h


def hbdg_alpha(p, alpha, phi):
    Nr = p['Nr']
    hu = ring_h(p, np.array(p['eps_up'], float), 0.0)
    hd = ring_h(p, np.array(p['eps_dn'], float), 0.0)
    for h in (hu, hd):
        h[1, 0] = -p['tJ'] * np.exp(1j * phi) * alpha
        h[0, 1] = np.conj(h[1, 0])
    D = np.diag(np.array(p['Delta'], float))
    return np.block([[hu, D], [D, -hd.conj()]]), np.trace(hd).real


def solve_mf(p, g, omega, phi, Nphot=120, nsteps=200, alpha0=1.0, mix=0.5, tol=1e-12):
    n = np.arange(Nphot)
    X = np.diag(np.sqrt(n[1:]), 1) + np.diag(np.sqrt(n[1:]), -1)
    Dop = expm(1j * g * X)
    Nr = p['Nr']
    alpha = alpha0
    for step in range(nsteps):
        H, c0 = hbdg_alpha(p, alpha, phi)
        vals, vecs = eigh(H)
        neg, pos = vals < 0, vals > 0
        chi = np.sum(np.conj(vecs[1, neg]) * vecs[0, neg]) + np.sum(vecs[1 + Nr, pos] * np.conj(vecs[Nr, pos]))
        eta = np.exp(1j * phi) * p['tJ'] * chi
        Hph = omega * np.diag(n) - (eta * Dop + np.conj(eta) * Dop.conj().T)
        ev, evec = eigh(Hph)
        psi = evec[:, 0]
        alphaNew = np.conj(psi) @ Dop @ psi
        if abs(alphaNew - alpha) < tol:
            alpha = alphaNew
            break
        alpha = mix * alpha + (1 - mix) * alphaNew
    Eel = vals[neg].sum() + c0
    return dict(E=Eel + ev[0] + 2 * np.real(eta * alpha), I=-2 * np.imag(eta * alpha),
                n=np.real(np.conj(psi) @ (n * psi)), X=np.real(np.conj(psi) @ X @ psi), alpha=alpha)
