"""Harmonics of the DMRG ground-state energy E(phi) of the interacting-dot ring (fixed parity sector)."""
import numpy as np, json


def load(fname, U, sector='odd'):
    d = [r for r in json.load(open(fname)) if abs(r['U'] - U) < 1e-9]
    phi = np.array([r['phi'] for r in d]); E = np.array([r[sector] for r in d])
    o = np.argsort(phi)
    return phi[o], E[o]


def harmonics_from_half(phi, E, M=8):
    """phi on [0, pi] (uniform, including both ends); E(-phi)=E(phi). Cosine series via least squares."""
    A = np.cos(np.outer(phi, np.arange(M + 1)))
    c, *_ = np.linalg.lstsq(A, E, rcond=None)
    return c.astype(complex)  # compatible with bo.V_of (Re sum c_m e^{i m phi})


if __name__ == '__main__':
    import sys
    for U in (2.0, 4.0):
        try:
            phi, E = load('../data/udot_phi_tJ0.5.json', U)
        except Exception as e:
            continue
        if len(phi) < 17:
            continue
        c = harmonics_from_half(phi, E)
        Epp = -np.sum(np.arange(len(c)) ** 2 * c.real)
        print('U=%.1f harmonics:' % U, np.round(c.real[1:6], 7), " E''(0)=%.5f  EJ_eff=|E''|/4=%.5f" % (Epp, abs(Epp) / 4),
              'fit resid %.1e' % np.max(abs(np.cos(np.outer(phi, np.arange(len(c)))) @ c.real - E)))
        np.save('../data/harmonics_udot_U%.0f.npy' % U, c)
