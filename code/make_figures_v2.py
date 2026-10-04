"""Recomputed figures: (U, g) phase diagram with seed-independent observables, and the cat-state figure
with the U = 6t full-DMRG check.  Reuses the style/helpers of make_figures.py."""
import json, os
import numpy as np
from make_figures import *          # noqa: F401,F403  (colors, tag, photon_xdist, D, F, classical_theta)
from make_figures import C1, C2, C3, C4, INK, MUTED, D, F, tag, photon_xdist, classical_theta
from matplotlib.colors import LogNorm
from exact_phase_diagram import cos_fit


def fig_main_v2():
    ex = json.load(open(D + 'phase_diagram_v2.json'))
    mf = json.load(open(D + 'mf_hf_grid_fine.json')); mf2 = json.load(open(D + 'mf_hf_grid.json'))
    om = ex['omega']; gs = np.array(ex['gs']); fe = ex['fine']; Ue = np.array(fe['Us'])
    I3 = np.abs(np.array(fe['I_seed3'])); I5 = np.abs(np.array(fe['I_seed5'])); dl = np.array(fe['delta'])
    Ic = 2 * np.abs(np.array(fe['E2']))[:, None]
    gcl = np.array(fe['g_class'])
    Um = np.array(mf['Us'][:len(mf['I'])]); Im = np.abs(np.array(mf['I'])); gm = np.array(mf['gs'])
    fig = plt.figure(figsize=(7.2, 2.7), constrained_layout=True)
    gsp = fig.add_gridspec(2, 4, width_ratios=[1.05, 1, 1, 1])
    a, b = fig.add_subplot(gsp[0, 0]), fig.add_subplot(gsp[1, 0])
    # ---- (a) U = 2t cut
    k2 = list(fe['Us']).index(2.0)
    a.plot(gs, I3[k2], color=C3, lw=1.3, label=r'exact, seed $10^{-3}$')
    a.plot(gs, I5[k2], color=C3, lw=1.0, ls='--', label=r'exact, seed $10^{-5}$')
    km = list(mf2['Us']).index(2.0)
    a.plot(mf2['gs'], np.abs(mf2['I'][km]), color=C2, label='mean field')
    b.plot(gs, fe['n'][k2], color=C3); b.plot(mf2['gs'], mf2['n'][km], color=C2)
    sc = json.load(open(D + 'dmrg_udot_U2_r10.json'))['rows']
    gsc = sum([json.load(open(D + f))['rows'] for f in ('gscan_udot_U2_seed1e-3_part1.json', 'gscan_udot_U2_seed1e-3_part2.json') if os.path.exists(D + f)], [])
    pts = [(r['g'], abs(r['I_vir'])) for r in gsc if r['n'] < 1.0] + \
          [(r['g'], om * r['X01'] / (2 * r['g'])) for r in sc if (r['E1'] - r['E0']) < 0.05 * om]
    a.plot(*zip(*sorted(pts)), 'o', ms=3.5, color=C1, label='full DMRG', zorder=5)
    b.plot([r['g'] for r in sc + gsc], [r.get('n0', r.get('n')) for r in sc + gsc], 'o', ms=3.0, color=C1, zorder=5)
    a.set_ylabel(r'$|\langle I\rangle|$ $(et/\hbar)$', fontsize=7); a.set_xlim(0, 0.4); a.tick_params(labelbottom=False)
    a.legend(fontsize=5.3, loc='upper right'); a.set_title(r'$U=2t$', fontsize=8); tag(a, '(a)')
    b.set_xlabel('$g$'); b.set_ylabel(r'$\langle a^\dagger a\rangle$', fontsize=7); b.set_xlim(0, 0.4)
    # ---- maps
    Irms = np.array(fe['Irms']); nn = np.array(fe['n']); nmf = np.array(mf['n'])
    odd = np.array([[sx == 'odd' for sx in row] for row in fe['sector']])
    mask = lambda A: np.where(odd, A, np.nan)        # criteria are defined in the doublet (pi) sector only
    Xrms = np.array(fe['Xrms'])                      # flux polarization of the cat by the seed, <X>/sqrt<X^2>, with <X> = 2g<I>/omega
    pol3, pol5 = mask(2 * gs[None, :] * I3 / om / Xrms), mask(2 * gs[None, :] * I5 / om / Xrms)
    nn = mask(nn); nmf = np.where(Um[:, None] >= 0.9, nmf, np.nan)
    axs = [fig.add_subplot(gsp[:, c]) for c in (1, 2, 3)]
    vmax = max(I3.max(), Im.max())
    pc0 = axs[0].pcolormesh(gs, Ue, I3, cmap='Blues', vmin=0, vmax=vmax, shading='nearest', rasterized=True)
    axs[0].contour(gs, Ue, pol3, levels=[0.5], colors=INK, linewidths=0.8)
    axs[0].contour(gs, Ue, pol5, levels=[0.5], colors=INK, linewidths=0.8, linestyles=':')
    axs[0].set_title(r'$|\langle I\rangle|$, seed $10^{-3}$', fontsize=7.5, loc='right')
    pc1 = axs[1].pcolormesh(gs, Ue, np.maximum(dl, 1e-8), cmap='Blues_r', norm=LogNorm(1e-8, 1), shading='nearest', rasterized=True)
    axs[1].contour(gs, Ue, dl, levels=[1e-6, 1e-3, 1e-1], colors='white', linewidths=0.6, linestyles='-')
    axs[1].contour(gs, Ue, nn, levels=[1.0], colors=INK, linewidths=0.8)
    axs[1].set_title(r'tunnel splitting $\delta/\hbar\omega$', fontsize=7.5, loc='right')
    pc2 = axs[2].pcolormesh(gm, Um, Im, cmap='Blues', vmin=0, vmax=vmax, shading='nearest', rasterized=True)
    axs[2].contour(gm, Um, nmf, levels=[1.0], colors=C2, linewidths=0.9, linestyles='--')
    axs[2].contour(gs, Ue, nn, levels=[1.0], colors=INK, linewidths=0.8)
    axs[2].set_title('mean field', fontsize=7.5, loc='right')
    Uc = 0.91
    for ax, lab in zip(axs, ('(b)', '(c)', '(d)')):
        ok = Ue >= Uc
        ax.plot(gcl[ok], Ue[ok], color=C4, ls='--', lw=0.9)
        ax.axhline(Uc, color=MUTED, lw=0.5, ls=':')
        ax.set_xlabel('$g$'); ax.set_xlim(0, 0.4); ax.set_ylim(0, 10); tag(ax, lab)
        ax.text(0.2, 0.25, 'singlet (0-junction): no current', fontsize=5.3, color=MUTED, ha='center')
    axs[0].set_ylabel(r'$U/t$')
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)
    for U in ex['Us']:
        axs[0].plot([0.4], [U], '<', ms=2.3, color=MUTED, clip_on=False)
    cb = fig.colorbar(pc0, ax=axs[0], pad=0.02, aspect=28, location='bottom', shrink=0.9); cb.set_label(r'$|\langle I\rangle|$ $(et/\hbar)$', fontsize=6.5)
    cb1 = fig.colorbar(pc1, ax=axs[1], pad=0.02, aspect=28, location='bottom', shrink=0.9); cb1.set_label(r'$\delta/\hbar\omega$', fontsize=6.5)
    cb2 = fig.colorbar(pc2, ax=axs[2], pad=0.02, aspect=28, location='bottom', shrink=0.9); cb2.set_label(r'$|\langle I\rangle|_{\rm MF}$ $(et/\hbar)$', fontsize=6.5)
    for c in (cb, cb1, cb2):
        c.ax.tick_params(labelsize=6)
    fig.savefig(F + 'fig_main_phase_diagram.pdf'); fig.savefig(F + 'fig_main_phase_diagram.png')
    plt.close(fig)


def fig_udot_v2():
    bo = json.load(open(D + 'bo_udot_U2.json'))
    c = np.load(D + 'harmonics_udot_U2.npy')
    EJ = bo['EJ']; betas = np.array(bo['betas'])
    dm = json.load(open(D + 'dmrg_udot_U2_r10.json'))['rows']
    f6 = D + 'dmrg_udot_U6_cat.json'
    d6 = json.load(open(f6)) if os.path.exists(f6) else None
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
    ratios = ['2', '5', '10', '40']
    shades = ['#bcd6f5', '#7fb0ec', C1, '#163f75']
    a = axs[0]
    a.plot(betas, classical_theta(c, betas, EJ, EJ / 10) / (np.pi / 2), color=C4, ls='--', lw=1, label='classical')
    for r, col in zip(ratios, shades):
        v = np.array(bo['runs'][r])
        a.plot(betas, v[:, 4] * v[:, 0] / (np.pi / 2), color=col, lw=1.2, label=r'$E_J/\hbar\omega=%s$' % r)
    a.plot([r['beta'] for r in dm], [r['X01'] * r['g'] / (np.pi / 2) for r in dm], 'o', ms=3.5, color=C2,
           label=r'DMRG, $U=2t$ ($E_J/\hbar\omega=10$)')
    if d6:
        # BO line for the U=6 junction at the same omega, and the full-DMRG points
        from phase_diagram_v2 import load_curves, coeffs
        c6 = coeffs(load_curves())['odd'][6.0].astype(complex); om6 = d6['omega']; EJ6 = d6['EJ']
        from bo import solve
        bb = np.linspace(0.05, 4, 60)
        res = [solve(c6, np.sqrt(b * om6 / (8 * EJ6)), om6, 0.0, Nphot=120) for b in bb]
        a.plot(bb, [o['X01'] * np.sqrt(b * om6 / (8 * EJ6)) / (np.pi / 2) for o, b in zip(res, bb)], color='#8c6bb1', lw=1.0, ls='-.')
        a.plot([r['beta'] for r in d6['rows']], [r['X01'] * r['g'] / (np.pi / 2) for r in d6['rows']], 's', ms=3.5, color='#8c6bb1',
               label=r'DMRG, $U=6t$ ($E_J/\hbar\omega=3.2$)')
        axs[1].semilogy(bb, [(o['E'][1] - o['E'][0]) / om6 for o in res], color='#8c6bb1', lw=1.0, ls='-.')
        axs[1].semilogy([r['beta'] for r in d6['rows']], [(r['E1'] - r['E0']) / om6 for r in d6['rows']], 's', ms=3.5, color='#8c6bb1')
    a.set_xlabel(r'$\beta$'); a.set_ylabel(r'$\Phi^\star/(\Phi_0/2)$'); a.legend(fontsize=5.3, loc='upper left')
    tag(a, '(a)')
    a = axs[1]
    for r, col in zip(ratios, shades):
        v = np.array(bo['runs'][r]); om = EJ / float(r)
        a.semilogy(betas, np.maximum(v[:, 2] / om, 3e-9), color=col, lw=1.2)
    res_ = 4e-2
    ok = [r for r in dm if (r['E1'] - r['E0']) / (EJ / 10) > 5 * res_]
    lim = [r for r in dm if (r['E1'] - r['E0']) / (EJ / 10) <= 5 * res_]
    a.semilogy([r['beta'] for r in ok], [(r['E1'] - r['E0']) / (EJ / 10) for r in ok], 'o', ms=3.5, color=C2)
    a.semilogy([r['beta'] for r in lim], [res_] * len(lim), 'v', ms=4, mfc='none', color=C2)
    a.text(1.75, 1.2e-2, 'DMRG: below\nresolution', fontsize=5.5, color=C2)
    a.axhspan(1e-9, 1e-8, color=MUTED, alpha=0.08, lw=0)
    a.text(0.1, 2e-9, 'BO numerical floor', fontsize=6, color=MUTED)
    a.set_ylim(1e-9, 2); a.set_xlabel(r'$\beta$'); a.set_ylabel(r'$\delta/\hbar\omega$'); tag(a, '(b)')
    a = axs[2]
    cols = ['#9ec5f0', '#5a9be3', C1, '#1a4f91', '#0d2a52']
    for k, r in enumerate(dm):
        x, p = photon_xdist(r['rho_ph'])
        a.plot(r['g'] * x / (np.pi / 2), p * (np.pi / 2) / r['g'], color=cols[k % 5], lw=1.2, label=r'$\beta=%.1f$' % r['beta'])
    a.set_xlim(-1.3, 1.3); a.set_xlabel(r'$\Phi/(\Phi_0/2)$'); a.set_ylabel(r'$P(\Phi)$ (DMRG, $U=2t$)')
    a.legend(fontsize=6); tag(a, '(c)')
    fig.savefig(F + 'fig_udot_cavity.pdf'); fig.savefig(F + 'fig_udot_cavity.png')
    plt.close(fig)


if __name__ == '__main__':
    import sys
    for name in (sys.argv[1:] or ['main_v2', 'udot_v2']):
        globals()['fig_' + name](); print('done', name)
