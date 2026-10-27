"""rf-SQUID + quantum LC circuit (the original notebook rfSQUID_with_quantum_LC_circtuit.nb):
mean field of the notebook (coherent-state resonator, exact SQUID mode) reproduced, versus the exact
two-mode ground state.  Units E_C = 1.  H = E_C N^2 + E_L (phi - phi_ext - g X)^2 - E_J cos phi + w_r a^dag a,
X = a + a^dag, notebook parameter g1^2 = g^2 E_L / w_r, c = g1^2/(1/4 + g1^2).
Run from talk/:  python make_rfsquid_fig.py  -> figs/rfsquid_mf.pdf"""
import os
import numpy as np
from scipy.linalg import eigh, expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, GREEN, PURPLE, GREY = '#2a6fdb', '#e8692b', '#1aa37a', '#5b3fa8', '#888888'

EC, EL, EJ = 1.0, 0.5, 2.0
phiext = np.pi - 1e-3
pz = (EC / EL) ** 0.25 / np.sqrt(2)


def squid_ops(Nb):
    b = np.diag(np.sqrt(np.arange(1, Nb)), 1); nb = b.T @ b; phi = pz * (b + b.T)
    E = expm(1j * phi)
    H0 = 2 * np.sqrt(EC * EL) * nb - 2 * EL * phiext * phi - EJ * np.real(E)   # E_C N^2 + E_L(phi-phiext)^2 - EJ cos, up to const
    return nb, phi, np.imag(E), H0


def mean_field(g1s, Nb=200, steps=60):
    nb, phi, sinphi, H0 = squid_ops(Nb)
    e, v = eigh(H0); psi = v[:, 0]; ph = psi @ phi @ psi
    out = []
    for g1 in g1s:
        c = g1 ** 2 / (0.25 + g1 ** 2)
        for _ in range(steps):
            H = H0 - 2 * EL * c * (ph - phiext) * (phi - phiext * np.eye(Nb))
            e, v = eigh(H); psi = v[:, 0]; ph = psi @ phi @ psi
        out.append((g1, ph, 2 * EL * (ph - phiext) * (1 - 2 * c), EJ * (psi @ sinphi @ psi)))
    return np.array(out)


def exact(g1, wr, Nb=40, Na=70):
    g = g1 * np.sqrt(wr / EL)
    a = np.diag(np.sqrt(np.arange(1, Na)), 1); X = a + a.T; na = a.T @ a
    nb, phi, sinphi, Hb = squid_ops(Nb)
    Ib, Ia = np.eye(Nb), np.eye(Na)
    H = (np.kron(Hb, Ia) + np.kron(Ib, wr * na) - 2 * EL * g * np.kron(phi - phiext * Ib, X) + EL * g * g * np.kron(Ib, X @ X))
    e, v = eigh(H); psi = v[:, 0]
    return psi @ np.kron(phi, Ia) @ psi, EJ * (psi @ np.kron(sinphi, Ia) @ psi), psi @ np.kron(Ib, na) @ psi, e[1] - e[0]


if __name__ == '__main__':
    g1s = np.arange(0, 1.5001, 0.025)
    mf = mean_field(g1s)
    g1x = np.arange(0, 1.5001, 0.1)
    ex = {wr: np.array([exact(g1, wr) for g1 in g1x]) for wr in (0.5, 0.1)}
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
    ax[0].plot(mf[:, 0], mf[:, 1] / np.pi, '-', color=ORANGE, lw=2, label='mean field (notebook)')
    for (wr, d), mk in zip(ex.items(), ('o', 's')):
        ax[0].plot(g1x, d[:, 0] / np.pi, mk, color=BLUE if wr == 0.5 else GREEN, ms=6, mfc='w', mew=1.5, label=r'exact, $\omega_r=%.1fE_C$' % wr)
    ax[0].axhline(1, color=GREY, lw=0.6, ls=':'); ax[0].axhline(0, color=GREY, lw=0.6, ls=':')
    ax[0].set_xlabel(r'$g_1=g\sqrt{E_L/\omega_r}$'); ax[0].set_ylabel(r'$\langle\varphi\rangle/\pi$'); ax[0].legend(frameon=False, fontsize=9.5)
    ax[0].set_title(r'loop phase ($\varphi_{\rm ext}=\pi-10^{-3}$)', fontsize=11)
    ax[1].plot(mf[:, 0], mf[:, 3] / EJ, '-', color=ORANGE, lw=2, label=r'mean field: $E_J\langle\sin\varphi\rangle$')
    ax[1].plot(mf[:, 0], -mf[:, 2] / EJ, '--', color=ORANGE, lw=1.2, label=r"notebook's $J$ (extra factor 2)")
    for (wr, d), mk in zip(ex.items(), ('o', 's')):
        ax[1].plot(g1x, d[:, 1] / EJ, mk, color=BLUE if wr == 0.5 else GREEN, ms=6, mfc='w', mew=1.5, label=r'exact, $\omega_r=%.1fE_C$' % wr)
    ax[1].axhline(0, color=GREY, lw=0.6, ls=':')
    ax[1].set_xlabel(r'$g_1$'); ax[1].set_ylabel(r'current $/\,I_c$'); ax[1].set_ylim(-1.3, 1.3); ax[1].legend(frameon=False, fontsize=9)
    ax[1].set_title(r'persistent current', fontsize=11)
    fig.suptitle(r'rf-SQUID ($E_C=1$, $E_L=0.5$, $E_J=2$, $\beta_{\rm loop}=E_J/2E_L=2$) + quantum $LC$ circuit', fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figs', 'rfsquid_mf.pdf')); plt.close(fig)
    for wr, d in ex.items():
        print('w_r=%.1f: max current/I_c = %.3f at g1=%.2f; <n_a> up to %.2f; gap at g1=1: %.4f' % (wr, d[:, 1].max() / EJ, g1x[np.argmax(d[:, 1])], d[:, 2].max(), d[g1x == 1.0, 3][0]))
    print('MF: max current/I_c = %.3f at g1=%.3f' % (mf[:, 3].max() / EJ, mf[np.argmax(mf[:, 3]), 0]))
