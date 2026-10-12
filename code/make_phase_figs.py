"""Figures for the (U, g) phase diagram section: full DMRG vs effective circuit-QED model vs mean field.
Reads data/full/U*_{even,odd}.json (full DMRG), data/effective_Nr10.json, data/mf_grid_Nr10.json."""
import json, glob, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../figs')
OMEGA = 7.39e-4
C = dict(dmrg='#1b1b1b', eff='#2a9d4f', mf='#e07b00', cl='#ffffff')
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'legend.fontsize': 7, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.4, 'legend.frameon': False, 'figure.dpi': 150})


def load_full():
    """ground state over sectors at every (U, g): dict U -> dict of arrays"""
    recs = {}
    for f in glob.glob(os.path.join(D, 'full', 'U*_*.json')):
        d = json.load(open(f))
        recs.setdefault(d['U'], {})[d['sector']] = {r['g']: r for r in d['rows']}
    out = {}
    for U, secs in recs.items():
        gs = sorted(set.intersection(*[set(s) for s in secs.values()])) if len(secs) > 1 else sorted(list(secs.values())[0])
        rows = []
        for g in gs:
            best = min(secs, key=lambda s: secs[s][g]['E0'])
            r = dict(secs[best][g]); r['sector'] = best
            if len(secs) > 1:
                r['dE_sec'] = max(secs[s][g]['E0'] for s in secs) - r['E0']
            rows.append(r)
        out[U] = dict(g=np.array([r['g'] for r in rows]),
                      I=np.array([r['I_bond'] for r in rows]), Ivir=np.array([r['I_vir'] for r in rows]),
                      X=np.array([r['X'] for r in rows]), X2=np.array([r['X2'] for r in rows]), n=np.array([r['n'] for r in rows]),
                      E=np.array([r['E0'] for r in rows]), sector=[r['sector'] for r in rows],
                      Xlo=np.array([min(r['X_starts'].values()) for r in rows]), Xhi=np.array([max(r['X_starts'].values()) for r in rows]),
                      dEst=np.array([max(r['E_starts'].values()) - min(r['E_starts'].values()) for r in rows]),
                      pmax=np.array([r['pn'][-1] for r in rows]), terr=np.array([r['trunc_err'] for r in rows]),
                      dE_sec=np.array([r.get('dE_sec', np.nan) for r in rows]), time=np.array([r['time'] for r in rows]))
    return out


def load_eff():
    d = json.load(open(os.path.join(D, 'effective_Nr10.json')))
    return d


def load_mf():
    return json.load(open(os.path.join(D, 'mf_grid_Nr10.json')))


def Ic_of(eff, sector='odd'):
    """critical current of the pi (doublet) sector junction, I_c = max_phi |dE_odd/dphi|, interpolated in U"""
    Us = np.array(eff['Us']); ph = np.linspace(0, 2 * np.pi, 4001); m = np.arange(5)
    Ic = []
    for U in eff['Us']:
        c = np.array(eff['harm'][str(U)][sector])
        Ic.append(np.max(np.abs(np.sum(-m[None, :] * c[None, :] * np.sin(np.outer(ph, m)), 1))))
    return PchipInterpolator(Us, np.array(Ic))


def grid_map(Us, gs, vals, Uf, gf):
    """interpolate a (U, g) table (list of rows, one per U) onto a fine grid; linear in g, pchip in U"""
    V = np.array(vals, float)
    A = np.array([np.interp(gf, gs, row) for row in V])          # (nU, ngf)
    B = PchipInterpolator(np.array(Us), A, axis=0)(Uf)            # (nUf, ngf)
    return B


def fig_phase_diagram(full, eff, mf):
    Icf = Ic_of(eff)
    gf = np.linspace(0.02, 0.40, 191); Uf = np.linspace(0.5, 8.0, 301)
    # --- DMRG map
    Us = sorted(U for U in full if len(full[U]['g']) >= 30 and full[U]['g'].max() >= 0.39); rows = []
    for U in Us:
        d = full[U]; rows.append(np.interp(gf, d['g'], np.abs(d['I'])) / Icf(U))
    Zd = PchipInterpolator(np.array(Us), np.array(rows), axis=0)(np.clip(Uf, min(Us), max(Us)))
    Zd[(Uf < min(Us)) | (Uf > max(Us))] = np.nan
    # --- effective & MF maps
    Ze = grid_map(eff['Us'], eff['gs'], [np.abs(r) / Icf(U) for U, r in zip(eff['Us'], eff['I'])], Uf, gf)
    gm = np.array(mf['gs']); Im = [np.abs(np.array(r)) for r in mf['I']]
    for r in Im: r[0] = r[1]   # the decoupled g=0 MF point is not meaningful
    Zm = grid_map(mf['Us'], gm, [r / Icf(U) for U, r in zip(mf['Us'], Im)], Uf, gf)
    # classical threshold beta = 8 g^2 E_J / omega = 1 with E_J = |E_2| of the ground sector (pi sector only)
    E2 = np.array([max(eff['harm'][str(U)]['odd'][2], 0) for U in eff['Us']])
    Ucl = np.array(eff['Us']); gcl = np.sqrt(OMEGA / (8 * np.maximum(E2, 1e-9)))
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), sharey=True, constrained_layout=True)
    norm = Normalize(0, 1)
    for ax, Z, ttl in zip(axs, (Zd, Ze, Zm), ('(a) DMRG, full model', '(b) effective circuit model', '(c) mean field')):
        im = ax.pcolormesh(Uf, gf, Z.T, cmap='Blues', norm=norm, shading='gouraud', rasterized=True)
        ax.contour(Uf, gf, Ze.T, levels=[0.5], colors=C['eff'], linewidths=1.2, linestyles='dashed')
        ax.contour(Uf, gf, Zm.T, levels=[0.5], colors=C['mf'], linewidths=1.2, linestyles='dotted')
        ax.plot(Ucl[E2 > 0], gcl[E2 > 0], color='#7b3294', lw=0.9, ls=(0, (4, 2, 1, 2)))
        # DMRG threshold at the computed U values: first g with |I| > 0.5 I_c (half a grid step as error bar)
        gcd = [(U, full[U]['g'][np.argmax(np.abs(full[U]['I']) / Icf(U) > 0.5)]) for U in Us if np.max(np.abs(full[U]['I']) / Icf(U)) > 0.5]
        if gcd:
            ax.errorbar([u for u, _ in gcd], [g - 0.005 for _, g in gcd], yerr=0.005, fmt='o', color=C['dmrg'], ms=3, mfc='w', mew=0.8, elinewidth=0.7, capsize=0, zorder=5)
            ax.plot([u for u, _ in gcd], [g - 0.005 for _, g in gcd], '-', color=C['dmrg'], lw=0.8, zorder=4)
        ax.set_xlabel(r'$U/\Delta$'); ax.set_title(ttl, fontsize=8.5, loc='left')
        ax.set_xlim(0.5, 8); ax.set_ylim(0.02, 0.40)
        ax2 = ax.secondary_xaxis('top', functions=(lambda u: u, lambda u: u)); ax2.set_xticks([])
    axs[0].set_ylabel(r'$g$')
    for ax in axs:
        sec = ax.secondary_yaxis('right', functions=(lambda g: 4 * g ** 2 / np.pi, lambda z: np.sqrt(np.pi * z) / 2))
        if ax is axs[-1]:
            sec.set_ylabel(r'$Z/R_Q$'); sec.set_yticks([0.01, 0.05, 0.1, 0.2])
        else:
            sec.set_yticks([])
    cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.1, aspect=25); cb.set_label(r'$|\langle I\rangle|/I_c(U)$')
    # markers for the DMRG U values
    for U in Us:
        axs[0].plot(U, 0.025, marker='^', color='k', ms=3, clip_on=False)
    fig.savefig(os.path.join(F, 'fig_phase_UG.pdf')); fig.savefig(os.path.join(F, 'fig_phase_UG.png'))
    return dict(Uf=Uf, gf=gf, Zd=Zd, Ze=Ze, Zm=Zm)


def fig_slices(full, eff, mf):
    Icf = Ic_of(eff)
    Uslices = [u for u in (1.25, 2.0, 4.0) if u in full]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
    ax = axs[0]
    marks = ['o', 's', 'D']
    for U, mk in zip(Uslices, marks):
        d = full[U]; ic = Icf(U)
        ke = eff['Us'].index(U) if U in eff['Us'] else None
        km = mf['Us'].index(U) if U in mf['Us'] else None
        ln, = ax.plot(d['g'], np.abs(d['I']) / ic, mk, ms=3.2, mfc='w', mew=0.9, label=r'$U=%g\Delta$' % U)
        col = ln.get_color()
        if ke is not None:
            ax.plot(eff['gs'], np.abs(eff['I'][ke]) / ic, '-', color=col, lw=1.1)
        if km is not None:
            ax.plot(mf['gs'][1:], np.abs(mf['I'][km][1:]) / ic, ':', color=col, lw=1.1)
    ax.set_xlabel(r'$g$'); ax.set_ylabel(r'$|\langle I\rangle|/I_c$'); ax.set_xlim(0.02, 0.40); ax.legend(loc='lower right')
    ax.set_title('(a) cuts at fixed $U$', fontsize=8.5, loc='left')
    # (b) cuts at fixed g
    ax = axs[1]
    for gsl, mk in zip((0.15, 0.25), ('o', 's')):
        Us = sorted(full); Id = []
        for U in Us:
            d = full[U]; Id.append(np.interp(gsl, d['g'], np.abs(d['I'])) / Icf(U))
        ln, = ax.plot(Us, Id, mk, ms=3.2, mfc='w', mew=0.9, label=r'$g=%g$' % gsl)
        col = ln.get_color()
        ke = eff['gs'].index(gsl); km = mf['gs'].index(gsl)
        ax.plot(eff['Us'], [abs(eff['I'][k][ke]) / Icf(U) for k, U in enumerate(eff['Us'])], '-', color=col, lw=1.1)
        ax.plot(mf['Us'], [abs(mf['I'][k][km]) / Icf(U) for k, U in enumerate(mf['Us'])], ':', color=col, lw=1.1)
    ax.set_xlabel(r'$U/\Delta$'); ax.set_xlim(0.4, 8); ax.legend(loc='upper right'); ax.set_title('(b) cuts at fixed $g$', fontsize=8.5, loc='left')
    # (c) photon number and flux at U=2
    ax = axs[2]
    if 2.0 in full:
        d = full[2.0]; ke = eff['Us'].index(2.0); km = mf['Us'].index(2.0)
        ax.plot(d['g'], d['n'], 'o', ms=3.2, mfc='w', mew=0.9, color=C['dmrg'], label='DMRG')
        ax.plot(eff['gs'], eff['n'][ke], '-', color=C['eff'], label='effective')
        ax.plot(mf['gs'][1:], mf['n'][km][1:], ':', color=C['mf'], label='mean field')
        ax.set_xlabel(r'$g$'); ax.set_ylabel(r'$\langle a^\dagger a\rangle$'); ax.set_xlim(0.02, 0.4); ax.legend()
        ax.set_title(r'(c) photon number, $U=2\Delta$', fontsize=8.5, loc='left')
    fig.savefig(os.path.join(F, 'fig_phase_slices.pdf')); fig.savefig(os.path.join(F, 'fig_phase_slices.png'))


def fig_convergence(full, eff):
    Icf = Ic_of(eff)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
    # (a) sum rule over the whole grid
    ax = axs[0]
    for U in sorted(full):
        d = full[U]
        ax.semilogy(d['g'], np.abs(d['I'] - d['Ivir']) / Icf(U) + 1e-12, '.', ms=3, color='0.3')
    ax.set_xlabel('$g$'); ax.set_ylabel(r'$|\langle I\rangle-\hbar\omega\langle X\rangle/2g|/I_c$'); ax.set_title('(a) current sum rule', fontsize=8.5, loc='left')
    # (b) bracket between the two starts
    ax = axs[1]
    for U in sorted(full):
        d = full[U]; dPhi = (d['Xhi'] - d['Xlo']) * d['g'] / (np.pi / 2)
        res = d['dEst'] > 3e-7
        ax.plot(d['g'][res], dPhi[res], 'o', ms=3, mfc='w', mew=0.7, color='0.5')
        ax.plot(d['g'][~res], dPhi[~res], 'o', ms=3, color='#c0392b')
    ax.plot([], [], 'o', ms=3, mfc='w', mew=0.7, color='0.5', label=r'resolved by energy ($\Delta E>3\times10^{-7}$)')
    ax.plot([], [], 'o', ms=3, color='#c0392b', label=r'bracket ($\Delta E<3\times10^{-7}$)')
    ax.legend(loc='upper right'); ax.set_ylim(-0.02, 0.6)
    ax.set_xlabel('$g$'); ax.set_ylabel(r'$|\Phi_{\rm C}-\Phi_{\rm A}|/(\Phi_0/2)$'); ax.set_title('(b) two initial states', fontsize=8.5, loc='left')
    # (c) chi dependence at U=2 if available
    ax = axs[2]
    for f, lab in (('bench/full2_Nr10_U2_chi300.json', r'$\chi=300$ (no annealing)'), ('full/U2.0_odd.json', r'$\chi=300$'), ('full_chi400/U2.0_odd.json', r'$\chi=400$'), ('full_seed1e-4/U2.0_odd.json', r'seed $10^{-4}$')):
        p = os.path.join(D, f)
        if os.path.exists(p):
            dd = json.load(open(p)); rr = dd['rows']
            ax.plot([r['g'] for r in rr], [abs(r['I_bond']) / Icf(2.0) for r in rr], 'o-', ms=3, mfc='w', lw=0.8, label=lab)
    ax.set_xlabel('$g$'); ax.set_ylabel(r'$|\langle I\rangle|/I_c$'); ax.set_xlim(0.08, 0.2); ax.legend(); ax.set_title(r'(c) $U=2\Delta$', fontsize=8.5, loc='left')
    fig.savefig(os.path.join(F, 'fig_phase_convergence.pdf')); fig.savefig(os.path.join(F, 'fig_phase_convergence.png'))


if __name__ == '__main__':
    full = load_full(); eff = load_eff(); mf = load_mf()
    print('DMRG U values:', sorted(full), {U: len(full[U]['g']) for U in full})
    fig_phase_diagram(full, eff, mf)
    fig_slices(full, eff, mf)
    fig_convergence(full, eff)
