"""Unrestricted Hartree-Fock (normal + anomalous) BdG for the SC ring with an interacting dot,
optionally combined with the notebook's photon product-state mean field (the user's method).

Dot: eps_d = -U/2, U n_up n_dn  ->  U<n_dn> n_up + U<n_up> n_dn + U F c^dag_up c^dag_dn + h.c.,  F = <c_dn c_up>
"""
import numpy as np
from scipy.linalg import eigh, expm
from bdg_general import sc_ring, ring_h


def hf_bdg(p, U, nu, nd, F, alpha, phi):
    Nr = p['Nr']
    hu = ring_h(p, np.array(p['eps_up'], float), 0.0)
    hd = ring_h(p, np.array(p['eps_dn'], float), 0.0)
    hu[0, 0] += U * nd; hd[0, 0] += U * nu
    for h in (hu, hd):
        h[1, 0] = -p['tJ'] * np.exp(1j * phi) * alpha
        h[0, 1] = np.conj(h[1, 0])
    D = np.diag(np.array(p['Delta'], complex)); D[0, 0] = U * F
    H = np.block([[hu, D], [D.conj(), -hd.conj()]])
    return H, np.trace(hd).real


def electron_step(p, U, nu, nd, F, alpha, phi):
    Nr = p['Nr']
    H, c0 = hf_bdg(p, U, nu, nd, F, alpha, phi)
    E, W = eigh(H)
    neg = E < 0
    u, v = W[:Nr, neg], W[Nr:, neg]
    nu_new = np.sum(np.abs(u[0]) ** 2)
    nd_new = 1 - np.sum(np.abs(v[0]) ** 2)
    F_new = np.sum(u[0] * np.conj(v[0]))
    # bond expectation <c^dag_1 c_0>_up + <c^dag_1 c_0>_dn  (as in the notebook)
    chi_up = np.sum(np.conj(u[1]) * u[0])
    pos = ~neg
    chi_dn = np.sum(W[Nr + 1, pos] * np.conj(W[Nr, pos]))
    Eel = E[neg].sum() + c0 - U * nu * nd + U * abs(F) ** 2
    return Eel, nu_new, nd_new, F_new, chi_up + chi_dn


def hf_energy(p, U, phi, magnetic=True, iters=400, mix=0.5):
    nu, nd, F = (1.0, 0.0, 0.0) if magnetic else (0.5, 0.5, 0.1)
    for _ in range(iters):
        E, a, b, c, _ = electron_step(p, U, nu, nd, F, 1.0, phi)
        if abs(a - nu) + abs(b - nd) + abs(c - F) < 1e-12:
            break
        nu, nd, F = mix * nu + (1 - mix) * a, mix * nd + (1 - mix) * b, mix * F + (1 - mix) * c
    return E, nu, nd, F


def mf_point(p, U, g, omega, phi, state, Dop, N, X, iters=600, mix=0.5, tol=1e-11):
    """notebook-type product state (HF-BdG electrons x exact photon), warm-started from `state`."""
    nu, nd, F, alpha = state
    for it in range(iters):
        Eel, a, b, c, chi = electron_step(p, U, nu, nd, F, alpha, phi)
        eta = np.exp(1j * phi) * p['tJ'] * chi
        Hph = omega * N - (eta * Dop + np.conj(eta) * Dop.conj().T)
        ev, evec = eigh(Hph)
        psi = evec[:, 0]
        alpha_new = np.conj(psi) @ Dop @ psi
        err = abs(a - nu) + abs(b - nd) + abs(c - F) + abs(alpha_new - alpha)
        nu, nd, F = mix * nu + (1 - mix) * a, mix * nd + (1 - mix) * b, mix * F + (1 - mix) * c
        alpha = mix * alpha + (1 - mix) * alpha_new
        if err < tol:
            break
    Etot = Eel + ev[0] + 2 * np.real(eta * alpha)
    I = -2 * np.imag(eta * alpha)
    n = np.real(np.conj(psi) @ (np.diag(N) * psi))
    x = np.real(np.conj(psi) @ X @ psi)
    return dict(E=Etot, I=I, n=n, X=x, nu=nu, nd=nd, F=F, alpha=alpha, it=it), (nu, nd, F, alpha)


if __name__ == '__main__':
    for U in [0.0, 1.0, 2.0, 4.0]:
        p = sc_ring(14, 1.0, 1.0, 0.5, Udot=U, dot=True)
        for mag in (True, False):
            E0, nu, nd, F = hf_energy(p, U, 0.0, mag)
            E1, *_ = hf_energy(p, U, np.pi / 2, mag)
            print('U=%.1f magnetic=%d  E(0)=%.6f  m=%.3f F=%.3f  [E(pi/2)-E(0)]/2=%+.5f' % (U, mag, E0, nu - nd, F.real, (E1 - E0) / 2))
