"""Kinetic inductance of the superconducting arms.

Eq. (1) of the paper keeps the pair field Delta fixed and real on every site of the chain: the phase of the
condensate is pinned (proximity picture) and the whole gauge-invariant phase drop sits on the junction.
If instead the chain is a self-supporting superconductor, Delta_j = -U_a <c_dn c_up>_j is determined
self-consistently, the phase drop spreads over the arms, and the arms act as a kinetic inductance L_K in
series with the junction.  This script measures L_K for the 14-site chain (t=1, |Delta|=1) and checks the
series rule  1/E''_sc = 1/E''_J + 1/K_lead  on the weak link (t_J = 0.5), then quotes the resulting
beta_K = 2 pi L_K I_c / Phi_0 = 4 E_J(U) / K_lead for the interacting-dot pi junction at each U node.
"""
import numpy as np, json
from scipy.linalg import eigh

L, t, tJ = 14, 1.0, 0.5


def bdg(D, a, tt):
    h = np.zeros((L, L), complex)
    for b in range(L):
        i, j = b, (b + 1) % L
        h[j, i] += -tt[b] * np.exp(1j * a[b]); h[i, j] += -tt[b] * np.exp(-1j * a[b])
    return np.block([[h, np.diag(D)], [np.diag(D).conj(), -h.conj()]])


def solve(phi, Ua, D0, tt, iters=5000, mix=0.3, selfcons=True):
    """total electron Peierls phase phi on bond 0; returns the mean-field energy (incl. |Delta|^2/U_a) and Delta."""
    a = np.zeros(L); a[0] = phi
    D = D0.copy()
    for _ in range(iters if selfcons else 1):
        E, W = eigh(bdg(D, a, tt)); neg = E < 0
        u, v = W[:L, neg], W[L:, neg]
        F = np.sum(u * v.conj(), axis=1)
        Dn = -Ua * F
        if not selfcons:
            break
        if np.max(abs(Dn - D)) < 1e-13:
            break
        D = (1 - mix) * D + mix * Dn
    Etot = E[neg].sum() + np.sum(abs(D) ** 2) / Ua       # BdG mean-field energy (constant trace term dropped)
    return Etot, D


def curvature(f, h=0.02):
    return (f(h) - 2 * f(0.0) + f(-h)) / h ** 2


if __name__ == '__main__':
    tt_wl = np.array([tJ] + [t] * (L - 1)); tt_uni = np.full(L, t)
    # calibrate U_a so that |Delta| = 1 in the uniform ring
    Ua = 3.0
    for _ in range(40):
        _, D = solve(0.0, Ua, np.ones(L, complex), tt_uni)
        Ua *= 1.0 / np.mean(abs(D)) ** 0.5
    Duni = D
    print('U_attr = %.4f  ->  |Delta| = %.4f (uniform ring)' % (Ua, np.mean(abs(Duni))))
    # weak link: self-consistent Delta at phi=0
    _, Dwl = solve(0.0, Ua, Duni, tt_wl)
    print('weak link, |Delta| on sites 0,1,7: %.3f %.3f %.3f' % (abs(Dwl[0]), abs(Dwl[1]), abs(Dwl[7])))
    Erig = lambda p: solve(p, Ua, Dwl, tt_wl, selfcons=False)[0]   # pinned (rigid) phase: Delta frozen, real
    Esc = lambda p: solve(p, Ua, Dwl, tt_wl, selfcons=True)[0]
    Euni = lambda p: solve(p, Ua, Duni, tt_uni, selfcons=True)[0]  # uniform ring: pure kinetic inductance of 14 bonds
    Krig, Ksc, Kuni = curvature(Erig), curvature(Esc), curvature(Euni)
    Klead = Kuni * L / (L - 1)       # 13 bonds in series instead of 14
    print("E''(0): pinned phase %.5f | self-consistent %.5f | uniform ring (14 bonds) %.5f -> K_lead(13 bonds) %.5f" % (Krig, Ksc, Kuni, Klead))
    print('series rule: 1/Ksc = %.3f ; 1/Krig + 1/Klead = %.3f' % (1 / Ksc, 1 / Krig + 1 / Klead))
    print('fraction of the phase drop on the junction = Ksc/Krig = %.3f' % (Ksc / Krig))
    # harmonics of the full self-consistent E(phi) vs pinned
    ph = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for name, f in (('pinned', Erig), ('self-consistent', Esc)):
        E = np.array([f(x) for x in ph]); c = np.fft.rfft(E) / len(ph); c[1:] *= 2
        print('%-16s E1=%+.5f E2=%+.5f E4=%+.5f' % (name, c[1].real, c[2].real, c[4].real))
    # beta_K for the interacting-dot pi junction: 4 E_J(U) / K_lead with E_J = |E''(0)|/4 of the doublet sector
    from exact_phase_diagram import load_curves, cos_fit
    curves = load_curves()
    out = dict(Ua=Ua, Krig=Krig, Ksc=Ksc, Klead=Klead, betaK={})
    print('\nU      E_J(odd)   beta_K = 4E_J/K_lead   g_c pinned   g_c with L_K   (omega = 7.61e-4)')
    for U in sorted(curves):
        c = cos_fit(*curves[U]['odd'], M=6); Epp = -np.sum(np.arange(7) ** 2 * c.real); EJ = abs(Epp) / 4
        bK = 4 * EJ / Klead
        om = 7.61e-4
        gc0 = np.sqrt(om / (2 * abs(Epp)))
        gc1 = np.sqrt(om * (1 - bK) / (2 * abs(Epp))) if bK < 1 else float('nan')
        out['betaK'][str(U)] = bK
        print('%-6.2f %.5f    %.3f                  %.4f       %.4f' % (U, EJ, bK, gc0, gc1))
    json.dump(out, open('../data/lead_inductance.json', 'w'), indent=1)
