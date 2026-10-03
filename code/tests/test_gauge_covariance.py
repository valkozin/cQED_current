"""Gauge check: all Peierls phase on one junction bond with real Delta (our model) is identical to a uniform
vector potential along the whole ring with the pairing dressed covariantly, Delta_j -> Delta e^{-2i theta_j}
(the prescription of Dmytruk & Schiro, arXiv:2310.01296). A uniform A with *real* Delta is NOT equivalent."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scipy.linalg import eigh
from bdg_general import sc_ring

p = sc_ring(14, 1.0, 1.0, 0.5, dot=True, eps_dot=0.0); L = p['Nr']


def E(phi, mode):
    aours = np.zeros(L); aours[0] = phi
    a = aours.copy() if mode == 'ours' else np.full(L, phi / L)
    th = np.zeros(L)
    for j in range(L - 1):
        th[j + 1] = th[j] + (aours[j] - a[j])          # c_j = e^{i th_j} c~_j
    h = np.zeros((L, L), complex)
    for b in range(L):
        i, j = b, (b + 1) % L
        t = p['tJ'] if b in (0, L - 1) else 1.0
        h[j, i] += -t * np.exp(1j * a[b]); h[i, j] += -t * np.exp(-1j * a[b])
    D = np.array(p['Delta'], complex)
    if mode == 'uniform_covariant':
        D = D * np.exp(-2j * th)
    H = np.block([[h, np.diag(D)], [np.diag(D).conj(), -h.conj()]])
    v = eigh(H, eigvals_only=True)
    return v[v < 0].sum()


for phi in [0.0, 0.5, 1.0, np.pi / 2]:
    e0, e1, e2 = E(phi, 'ours'), E(phi, 'uniform_covariant'), E(phi, 'uniform_real')
    assert abs(e0 - e1) < 1e-10
    print('phi=%.3f ours=%.10f uniform+covariant=%.10f uniform+real Delta=%.10f' % (phi, e0, e1, e2))
