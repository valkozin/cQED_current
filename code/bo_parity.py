"""Parity-resolved adiabatic (Born-Oppenheimer) solver for H = omega/4 (P^2 + X^2) + E(g X) with E(phi) = E(-phi).
The flux parity X -> -X is a symmetry; the two lowest states are the even and odd ground states.  Diagonalizing
the two parity blocks separately (and dropping the constant E_0) gives the tunnel splitting delta = E_odd - E_even
as a difference of two independently converged eigenvalues, with an absolute floor ~1e-15 * ||H|| instead of the
~1e-8 omega floor of the full diagonalization (where delta is the difference of two nearly degenerate eigenvalues
of one matrix)."""
import numpy as np
from scipy.linalg import eigh


def V_of(c, phi):
    m = np.arange(len(c))
    return np.real(np.exp(1j * np.outer(phi, m)) @ c)


def solve_parity(c, g, omega, Nphot=200):
    """returns dict(E_even, E_odd, delta, X01, Phi_star (eigenvalue of X projected on the doublet), n_even)"""
    c = np.array(c, complex); c0 = c[0].real; c[0] = 0.0
    n = np.arange(Nphot)
    X = np.diag(np.sqrt(n[1:]), 1) + np.diag(np.sqrt(n[1:]), -1)
    out = {}
    vecs = {}
    for par, sel in (('even', n % 2 == 0), ('odd', n % 2 == 1)):
        # the photon-number parity (-1)^n is the X -> -X parity; V(gX) is even so H is block diagonal
        H = omega * np.diag(n.astype(float))
        # potential in the Fock basis via DVR of the full X, then project onto the parity block
        x, U = eigh(X)
        Vfull = (U * V_of(c, g * x)) @ U.T
        Hb = (H + Vfull)[np.ix_(sel, sel)]
        E, W = eigh(Hb)
        out['E_' + par] = E[0] + c0
        v = np.zeros(Nphot); v[sel] = W[:, 0]; vecs[par] = v
        out['n_' + par] = float(np.sum(n * v ** 2))
    out['delta'] = out['E_odd'] - out['E_even']
    out['X01'] = float(abs(vecs['even'] @ X @ vecs['odd']))
    return out


if __name__ == '__main__':
    import sys, json, os
    D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
    c = np.load(os.path.join(D, 'harmonics_udot_U2_Nr10.npy'), allow_pickle=True)
    EJ = c[2].real
    print('E_J = %.5e' % EJ)
    from bo import solve
    res = {}
    for ratio in (2, 5, 10, 40):
        omega = EJ / ratio
        rows = []
        for beta in np.round(np.arange(0.25, 4.01, 0.25), 3):
            g = np.sqrt(beta * omega / (8 * EJ))
            o = solve_parity(c, g, omega, Nphot=200 if ratio < 40 else 300)
            of = solve(c, g, omega, 0.0, Nphot=200 if ratio < 40 else 300)
            rows.append(dict(beta=float(beta), g=float(g), delta=o['delta'] / omega, delta_full=(of['E'][1] - of['E'][0]) / omega,
                             Phi=o['X01'] * g / (np.pi / 2)))
            print('EJ/om=%2d beta=%.2f  delta/om parity=%.3e  full=%.3e  Phi*=%.4f' % (ratio, beta, rows[-1]['delta'], rows[-1]['delta_full'], rows[-1]['Phi']), flush=True)
        res[str(ratio)] = rows
    json.dump(dict(EJ=EJ, rows=res), open(os.path.join(D, 'bo_parity_U2_Nr10.json'), 'w'))
