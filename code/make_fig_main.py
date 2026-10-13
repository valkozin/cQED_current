"""Main phase-diagram figure of the PRL version: (a) full-DMRG persistent current (seeded), (b) seed-free order
parameter of the effective circuit model, the tunnel splitting delta(U, g) between the two current directions,
(c) mean field.  Same data as make_phase_figs.py."""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy.interpolate import PchipInterpolator
from make_phase_figs import load_full, load_eff, load_mf, Ic_of, grid_map, D, F, OMEGA, C

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'legend.fontsize': 7, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.4, 'legend.frameon': False, 'figure.dpi': 150})


def main():
    full, eff, mf = load_full(), load_eff(), load_mf()
    Icf = Ic_of(eff)
    gf = np.linspace(0.02, 0.40, 191); Uf = np.linspace(0.5, 8.0, 301)
    Us = sorted(U for U in full if len(full[U]['g']) >= 30 and full[U]['g'].max() >= 0.39)
    rows = [np.interp(gf, full[U]['g'], np.abs(full[U]['I'])) / Icf(U) for U in Us]
    Zd = PchipInterpolator(np.array(Us), np.array(rows), axis=0)(np.clip(Uf, min(Us), max(Us)))
    Zd[(Uf < min(Us)) | (Uf > max(Us))] = np.nan
    Ze = grid_map(eff['Us'], eff['gs'], [np.abs(r) / Icf(U) for U, r in zip(eff['Us'], eff['I'])], Uf, gf)
    # seed-free: tunnel splitting delta / hbar omega of the effective model (log), only where the loop is a pi loop
    Zdel = grid_map(eff['Us'], eff['gs'], [np.log10(np.maximum(np.array(r), 1e-9)) for r in eff['delta']], Uf, gf)
    sec = np.array([[1.0 if s == 'odd' else 0.0 for s in row] for row in eff['sector']])
    Zsec = grid_map(eff['Us'], eff['gs'], sec, Uf, gf)
    gm = np.array(mf['gs']); Im = [np.abs(np.array(r)) for r in mf['I']]
    for r in Im: r[0] = r[1]
    Zm = grid_map(mf['Us'], gm, [r / Icf(U) for U, r in zip(mf['Us'], Im)], Uf, gf)
    E2 = np.array([max(eff['harm'][str(U)]['odd'][2], 0) for U in eff['Us']])
    Ucl = np.array(eff['Us']); gcl = np.sqrt(OMEGA / (8 * np.maximum(E2, 1e-9)))

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True, constrained_layout=True)
    ims = []
    for k, (ax, ttl) in enumerate(zip(axs, ('(a) full DMRG', '(b) circuit model: $\\delta$', '(c) mean field'))):
        if k == 1:
            Zp = np.where(Zsec > 0.5, Zdel, np.nan)
            im = ax.pcolormesh(Uf, gf, Zp.T, cmap='Blues_r', vmin=-8, vmax=0, shading='gouraud', rasterized=True)
            ax.contour(Uf, gf, np.nan_to_num(Zp, nan=0.0).T, levels=[-6, -3, -1], colors='w', linewidths=0.7)
            ax.contourf(Uf, gf, Zsec.T, levels=[-0.5, 0.5], colors=['#f2f2f2'])
            ax.text(2.5, 0.05, '0 loop (singlet)', fontsize=6.5, color='0.3')
        else:
            im = ax.pcolormesh(Uf, gf, (Zd if k == 0 else Zm).T, cmap='Blues', norm=Normalize(0, 1), shading='gouraud', rasterized=True)
        ims.append(im)
        ax.contour(Uf, gf, Ze.T, levels=[0.5], colors=C['eff'], linewidths=1.1, linestyles='dashed')
        ax.plot(Ucl[E2 > 0], gcl[E2 > 0], color='#7b3294', lw=0.9, ls=(0, (4, 2, 1, 2)))
        gcd = [(U, full[U]['g'][np.argmax(np.abs(full[U]['I']) / Icf(U) > 0.5)]) for U in Us if np.max(np.abs(full[U]['I']) / Icf(U)) > 0.5]
        if k != 1 and gcd:
            ax.errorbar([u for u, _ in gcd], [g - 0.005 for _, g in gcd], yerr=0.005, fmt='o', color=C['dmrg'], ms=3, mfc='w', mew=0.8, elinewidth=0.7, capsize=0, zorder=5)
        ax.set_xlabel(r'$U/\Delta$'); ax.set_title(ttl, fontsize=8, loc='left')
        ax.set_xlim(0.5, 8); ax.set_ylim(0.02, 0.40)
    axs[0].set_ylabel(r'$g$')
    secax = axs[-1].secondary_yaxis('right', functions=(lambda g: 4 * g ** 2 / np.pi, lambda z: np.sqrt(np.pi * np.maximum(z, 0)) / 2))
    secax.set_ylabel(r'$Z/R_Q$'); secax.set_yticks([0.01, 0.05, 0.1, 0.2])
    cb = fig.colorbar(ims[0], ax=axs[[0, 2]], shrink=0.85, pad=0.12, aspect=25); cb.set_label(r'$|\langle I\rangle|/I_c(U)$')
    cb2 = fig.colorbar(ims[1], ax=axs[1], shrink=0.85, pad=0.03, aspect=25); cb2.set_label(r'$\log_{10}\delta/\hbar\omega$')
    for U in Us:
        axs[0].plot(U, 0.025, marker='^', color='k', ms=3, clip_on=False)
    fig.savefig(os.path.join(F, 'fig_phase_main.pdf')); fig.savefig(os.path.join(F, 'fig_phase_main.png'))


if __name__ == '__main__':
    main()
