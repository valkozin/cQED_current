"""Self-consistent mean-field (product state) solution, line-by-line port of the .nb."""
import numpy as np, sys, json
from scipy.linalg import expm, eigh
from model import DEFAULTS, photon_ops, hbdg


def run(p=DEFAULTS, glist=np.arange(0, 1.2001, 0.02), Nphot=200, nsteps=20, alpha0=1.0,
        mix=0.5, verbose=False):
    a, adag, N, X = photon_ops(Nphot)
    Ns = 1 + p['NumSCsites']
    alpha = alpha0
    out = []
    for g in glist:
        Dop = expm(1j * g * X)
        for step in range(nsteps):
            vals, vecs = eigh(hbdg(p, alpha))
            neg, pos = vals < 0, vals > 0
            chiUp = np.sum(np.conj(vecs[1, neg]) * vecs[0, neg])
            chiDn = np.sum(vecs[1 + Ns, pos] * np.conj(vecs[0 + Ns, pos]))
            eta = np.exp(1j * p['phiClassical']) * p['tR0'] * (chiUp + chiDn)
            Hph = p['omegaR'] * N - (eta * Dop + np.conj(eta) * Dop.conj().T)
            ev, evec = eigh(Hph)
            Eph, psi = ev[0], evec[:, 0]
            alphaNew = np.conj(psi) @ Dop @ psi
            alpha = mix * alpha + (1 - mix) * alphaNew
        Eel = vals[vals < 0].sum() + p['ED0'] + p['DeltaZ0']
        Etot = Eel + Eph + 2 * np.real(eta * alpha)
        n = np.real(np.conj(psi) @ N @ psi)
        x = np.real(np.conj(psi) @ X @ psi)
        I = -2 * np.imag(eta * alpha)
        out.append([g, Eel, Etot, n, I, x, abs(alpha), np.angle(alpha), ev[1] - ev[0]])
        if verbose:
            print(f"g={g:.3f} Eel={Eel:.6f} Etot={Etot:.6f} n={n:.3f} I={I:.5f} X={x:.3f} "
                  f"|a|={abs(alpha):.4f} arg={np.angle(alpha):.4f} gap_ph={ev[1]-ev[0]:.2e}", flush=True)
    return np.array(out)


if __name__ == '__main__':
    res = run(verbose=True)
    np.save('../data/mf_notebook_port.npy', res)
