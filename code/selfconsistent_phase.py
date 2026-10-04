"""Self-consistent BdG check: where does the gauge-invariant phase drop sit in a SC ring with a weak link?
Delta_j = -U_a <c_dn c_up>_j is solved self-consistently (complex), so current conservation is restored and
no assumption about the phase profile is made. Flux enters as the total Peierls phase (placed on the junction bond;
the result is gauge invariant)."""
import numpy as np
from scipy.linalg import eigh

L, t, tJ = 14, 1.0, 0.5


def bdg(D, a):
    h = np.zeros((L, L), complex)
    for b in range(L):
        i, j = b, (b + 1) % L
        tb = tJ if b == 0 else t
        h[j, i] += -tb * np.exp(1j * a[b]); h[i, j] += -tb * np.exp(-1j * a[b])
    return np.block([[h, np.diag(D)], [np.diag(D).conj(), -h.conj()]])


def solve(phi, Ua, D0, iters=3000, mix=0.3):
    a = np.zeros(L); a[0] = phi           # electron Peierls phase; pair (Cooper) flux = 2*phi
    D = D0.copy()
    for _ in range(iters):
        E, W = eigh(bdg(D, a)); neg = E < 0
        u, v = W[:L, neg], W[L:, neg]
        F = np.sum(u * v.conj(), axis=1)   # <c_dn c_up>_j
        Dn = -Ua * F
        if np.max(abs(Dn - D)) < 1e-12: break
        D = (1 - mix) * D + mix * Dn
    # bond currents  I_b = 2 Im[ t_b e^{i a_b} <c^dag_{b+1} c_b> ] summed over spin (pos/neg convention irrelevant for ratios)
    rho_up = u.conj() @ u.T                  # <c^dag_i c_j>_up  -> rho_up[i,j]
    I = []
    for b in range(L):
        i, j = b, (b + 1) % L
        tb = tJ if b == 0 else t
        I.append(2 * 2 * np.imag(tb * np.exp(1j * a[b]) * rho_up[j, i]))   # x2 spin
    th = np.angle(D)
    gamma = np.array([np.angle(np.exp(1j * (th[(b + 1) % L] - th[b] - 2 * a[b]))) for b in range(L)])  # gauge-inv. pair phase drop
    return D, np.array(I), gamma


# calibrate U_a so that the bulk gap is ~1
Ua = 3.0
for _ in range(30):
    D, I, g = solve(0.0, Ua, np.ones(L, complex))
    Ua *= 1.0 / np.mean(abs(D)) ** 0.5
D0 = D
print('U_attr = %.3f, |Delta| bulk = %.3f' % (Ua, np.mean(abs(D0))))
for phi in [0.2, 0.4, 0.6]:
    D, I, gamma = solve(phi, Ua, D0.astype(complex))
    print('pair flux 2phi=%.2f: bond currents min/max = %.5f / %.5f (conserved: %s)' % (2 * phi, I.min(), I.max(), np.allclose(I, I.mean(), rtol=1e-3)))
    print('   phase drop on junction = %.4f, on all 13 lead bonds = %.4f  -> fraction on junction = %.3f' %
          (gamma[0], gamma[1:].sum(), gamma[0] / gamma.sum()))
