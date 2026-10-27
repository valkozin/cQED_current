"""Pedagogical figures for the seminar talk (pure pi junction, E(phi) = E_J cos 2phi).
Run from talk/:  python make_talk_figs.py   -> figs/*.pdf"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, GREEN, PURPLE, GREY = '#2a6fdb', '#e8692b', '#1aa37a', '#5b3fa8', '#888888'
SEQ = ['#a8c8f0', '#6fa0e6', '#2a6fdb', '#123f8c']


def bo_levels(beta, r, k=2, N=3000):
    """H = -hw d^2/dX^2 + hw X^2/4 + E_J cos(2 g X), hw = 1, E_J = r, beta = 8 g^2 E_J."""
    g = np.sqrt(beta / (8 * r))
    Xmax = max(8.0, 1.6 * np.pi / (2 * g) + 6)
    X = np.linspace(-Xmax, Xmax, N); h = X[1] - X[0]
    V = X ** 2 / 4 + r * np.cos(2 * g * X)
    E, W = eigh_tridiagonal(2 / h ** 2 + V, -np.ones(N - 1) / h ** 2, select='i', select_range=(0, k - 1))
    W /= np.sqrt(h)
    return X, g, V, E, W


# ---------------------------------------------------------------- 1. Landau landscape + pitchfork
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
phi = np.linspace(-1.35, 1.35, 600) * np.pi / 2
for b, c in zip([0.5, 1.0, 1.6, 2.5], SEQ):
    F = 2 * phi ** 2 / b + np.cos(2 * phi) - 1
    ax[0].plot(phi / (np.pi / 2), F, color=c, lw=2, label=r'$\beta=%.1f$' % b)
ax[0].set_ylim(-1.0, 1.6); ax[0].axhline(0, color=GREY, lw=0.6)
ax[0].set_xlabel(r'$\Phi/(\Phi_0/2)=\varphi/(\pi/2)$'); ax[0].set_ylabel(r'$[\mathcal{F}(\varphi)-\mathcal{F}(0)]/E_J$')
ax[0].legend(frameon=False, fontsize=9, loc='upper center', ncol=2)
ax[0].set_title(r'$\mathcal{F}/E_J = 2\varphi^2/\beta + \cos 2\varphi$', fontsize=11)
bb = np.linspace(0, 4, 400)
th = np.array([0.0 if b <= 1 else brentq(lambda t: t - b * np.sin(t), 1e-9, np.pi) for b in bb])
ax[1].plot(bb, th / np.pi, color=BLUE, lw=2, label=r'$\theta^\star/\pi$  (flux)')
ax[1].plot(bb, -th / np.pi, color=BLUE, lw=2)
ax[1].plot(bb, np.sin(th), color=ORANGE, lw=2, label=r'$I^\star/I_c=\sin\theta^\star$')
ax[1].axvline(1, color=GREY, ls='--', lw=1); ax[1].axhline(0, color=GREY, lw=0.6)
ax[1].set_xlabel(r'$\beta = 8g^2E_J/\hbar\omega$'); ax[1].legend(frameon=False, fontsize=9, loc='center right', bbox_to_anchor=(1.0, 0.3))
ax[1].set_title(r'$\theta^\star = \beta\sin\theta^\star$  ($\theta=2\varphi$)', fontsize=11)
fig.tight_layout(); fig.savefig('figs/landau.pdf'); plt.close(fig)

# ---------------------------------------------------------------- 2. quantum double well: cat states
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=False)
for a, (b, r) in zip(ax, [(0.5, 10), (2.0, 2), (2.0, 10)]):
    X, g, V, E, W = bo_levels(b, r)
    x = g * X / (np.pi / 2)
    m = np.abs(x) < 1.35
    V0 = V - V.min()
    a.plot(x[m], V0[m], color='k', lw=1.6)
    barrier0 = V0[np.argmin(np.abs(X))]
    scale = 0.45 * max(1.5, barrier0) / np.abs(W).max()
    for n, c, ls in zip([0, 1], [BLUE, ORANGE], ['-', '--']):
        En = E[n] - V.min()
        a.axhline(En, color=c, lw=0.8, ls=':', xmax=1)
        a.plot(x[m], En + scale * W[m, n] * np.sign(W[np.argmax(X > 0.3 / g) if b > 1 else np.argmax(np.abs(W[:, n])), n]), color=c, lw=1.8, ls=ls,
               label=r'$\psi_%d$' % n)
    d = E[1] - E[0]
    if d > 1e-2:
        ds = r'%.2f' % d
    else:
        ex = int(np.floor(np.log10(d))); ds = r'%.0f\times10^{%d}' % (d / 10 ** ex, ex)
    a.set_title(r'$\beta=%.1f,\ E_J/\hbar\omega=%d$' % (b, r) + '\n' + r'$\delta=%s\,\hbar\omega$' % ds, fontsize=10)
    a.set_xlabel(r'$\Phi/(\Phi_0/2)$'); a.set_xlim(-1.35, 1.35)
    barrier = V0[np.argmin(np.abs(X))]
    a.set_ylim(-0.2 if barrier < 0.5 else -0.25 * barrier - 0.6, 4.5 if barrier < 0.5 else 2.0 * barrier + 1.5)
ax[0].set_ylabel(r'$V_{\rm BO}/\hbar\omega$'); ax[0].legend(frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig('figs/quantum_well.pdf'); plt.close(fig)

# ---------------------------------------------------------------- 3. soft mode: gap vs beta
fig, ax = plt.subplots(figsize=(4.6, 3.4))
bb = np.linspace(0.02, 3, 70)
for r, c in zip([2, 10, 40], SEQ[1:]):
    ax.semilogy(bb, [np.diff(bo_levels(b, r)[3])[0] for b in bb], color=c, lw=2, label=r'$E_J/\hbar\omega=%d$' % r)
bc = np.linspace(0, 0.999, 200)
ax.semilogy(bc, np.sqrt(1 - bc), color=PURPLE, ls='--', lw=1.6, label=r'classical soft mode $\sqrt{1-\beta}$')
ax.plot([0.5, 1.0, 1.5], [0.70, 0.24, 2.65e-4], 'o', color=ORANGE, ms=7, label=r'full DMRG, $U=2\Delta$')
ax.set_ylim(1e-8, 2); ax.set_xlabel(r'$\beta$'); ax.set_ylabel(r'$(E_1-E_0)/\hbar\omega$')
ax.legend(frameon=False, fontsize=9.5, loc='lower left')
fig.tight_layout(); fig.savefig('figs/soft_mode.pdf'); plt.close(fig)
print('done')
