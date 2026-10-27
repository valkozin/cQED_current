"""Figure for the short talk: persistent current and photon number vs coupling g at U = 2 Delta
(full DMRG from data/full, circuit model from data/effective_Nr10.json, classical Landau theory).
Run from talk/:  python make_short_figs.py  -> figs/current_vs_g.pdf"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, '..', 'code')); sys.path.insert(0, '.')
import make_phase_figs as mpf

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, GREEN, PURPLE = '#2a6fdb', '#e8692b', '#1aa37a', '#5b3fa8'

full, eff = mpf.load_full(), mpf.load_eff()
U = 2.0
d = full[U]; ic = mpf.Ic_of(eff)(U); ke = eff['Us'].index(U)
EJ = abs(np.array(eff['harm'][str(U)]['odd'])[2]); om = eff['omega']

gg = np.linspace(0.02, 0.4, 300)
beta = 8 * gg ** 2 * EJ / om
th = np.array([0.0 if b <= 1 else brentq(lambda t: t - b * np.sin(t), 1e-9, np.pi) for b in beta])

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
ax[0].plot(gg, np.sin(th), '--', color=PURPLE, lw=1.6, label=r'classical: $\theta^\star=\beta\sin\theta^\star$')
ax[0].plot(eff['gs'], np.abs(eff['I'][ke]) / ic, '-', color=GREEN, lw=2, label='circuit model, exact photon')
ax[0].plot(d['g'], np.abs(d['I']) / ic, 'o', ms=6, color=BLUE, mfc='w', mew=1.6, label='full DMRG', zorder=5)
gc = np.sqrt(om / (8 * EJ))
ax[0].axvline(gc, color=PURPLE, lw=0.8, ls=':')
ax[0].text(gc + 0.005, 0.05, r'$\beta=1$', color=PURPLE, fontsize=10)
ax[0].set_xlabel(r'coupling $g$   ($g^2=\pi Z/4R_Q$)'); ax[0].set_ylabel(r'$|\langle I\rangle|/I_c$')
ax[0].set_xlim(0.02, 0.4); ax[0].set_ylim(-0.03, 1.1); ax[0].legend(frameon=False, fontsize=9.5, loc='upper right')
ax[1].plot(eff['gs'], eff['n'][ke], '-', color=GREEN, lw=2)
ax[1].plot(d['g'], d['n'], 'o', ms=6, color=BLUE, mfc='w', mew=1.6, zorder=5)
ax[1].axvline(gc, color=PURPLE, lw=0.8, ls=':')
ax[1].set_xlabel(r'coupling $g$'); ax[1].set_ylabel(r'photons $\langle a^\dagger a\rangle$'); ax[1].set_xlim(0.02, 0.4)
fig.suptitle(r'$U=2\Delta$ ($\pi$ junction), $\hbar\omega=E_J/10$, no applied flux', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figs', 'current_vs_g.pdf')); plt.close(fig)
print('gc_cl=%.3f  Imax/Ic=%.3f at g=%.3f  n_max=%.2f' % (gc, (np.abs(d['I']) / ic).max(), d['g'][np.argmax(np.abs(d['I']))], d['n'].max()))
