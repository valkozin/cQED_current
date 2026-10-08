"""Phase-diagram figure of Eq. (1) at a given hbar*omega, with the current from the full electron+photon DMRG,
evaluated as the expectation value of the microscopic fermionic current operator on the junction bond
I = i t_J sum_s <e^{i phi_cl} D c^dag_1s c_0s - h.c.>  (code/run_dmrg_point.py, data/om<omega>/*.json).
Reference lines/maps: adiabatic theory and mean field at the same omega (data/phase_diagram_om<omega>.json).

usage: python make_figures_om.py [omega seed Ucut gmax]   -> ../figs/fig_phase_diagram_om<omega>.{pdf,png}
       defaults 0.01 1e-3 2 1.2;  e.g. python make_figures_om.py 0.003 1e-2 1 0.8
"""
import glob, json, sys
import numpy as np
from matplotlib.colors import LogNorm
from make_figures import plt, C1, C2, C3, C4, INK, MUTED, D, F, tag
import phase_diagram_v2 as pdv2

args = sys.argv[1:] + ['0.01', '1e-3', '2', '1.2'][len(sys.argv[1:]):]
OM, SEED, UCUT, GMAX = float(args[0]), float(args[1]), float(args[2]), float(args[3])
TAG = '%g' % OM


def sci(x):
    return r'10^{%d}' % round(np.log10(x))


def bo_cut(U, gs, seeds):
    """adiabatic current and photon number along g at fixed U (doublet sector), for several seeds"""
    pdv2.omega = OM
    c = pdv2.coeffs(pdv2.load_curves())['odd'][U].astype(complex)
    I = {s: [] for s in seeds}; n = []
    for g in gs:
        for s in seeds:
            o = pdv2.solve(c, g, OM, s, Nphot=140)
            I[s].append(o['I'][0])
        n.append(pdv2.solve(c, g, OM, seeds[0], Nphot=140)['n'][0])
    return {s: np.abs(I[s]) for s in seeds}, np.array(n)


def load_dmrg():
    rows = []
    for f in sorted(glob.glob(D + 'om%s/dmrg_U*_g*.json' % TAG)):
        d = json.load(open(f))
        if not d['stages']:
            continue
        s = d['stages'][-1]
        r = dict(U=d['U'], g=d['g'], chi=s['chi'], I=abs(s['I_bond']), I_vir=abs(s['I_vir']), n=s['n'],
                 dI_chi=(abs(s['I_bond'] - d['stages'][-2]['I_bond']) if len(d['stages']) > 1 else np.nan))
        if 'excited' in d:
            r['delta'] = (d['excited']['E1'] - s['E0']) / OM
        rows.append(r)
    return rows


def fig():
    ex = json.load(open(D + 'phase_diagram_om%s.json' % TAG))
    gs = np.array(ex['gs']); fe = ex['fine']; Ue = np.array(fe['Us']); mf = ex['mf']
    I3, I5 = np.abs(np.array(fe['I_seed3'])), np.abs(np.array(fe['I_seed5']))
    dl, gcl = np.array(fe['delta']), np.array(fe['g_class'])
    odd = np.array([[sx == 'odd' for sx in row] for row in fe['sector']])
    dl = np.where(odd, dl, np.nan)                   # tunnel splitting of the current states: doublet (pi) sector only
    Um, Im, nm = np.array(mf['Us']), np.abs(np.array(mf['I'])), np.array(mf['n'])
    rows = load_dmrg()
    cut = sorted([r for r in rows if r['U'] == UCUT], key=lambda r: r['g'])

    fig = plt.figure(figsize=(7.2, 2.8), constrained_layout=True)
    gsp = fig.add_gridspec(2, 4, width_ratios=[1.1, 1, 1, 1])
    a, b = fig.add_subplot(gsp[0, 0]), fig.add_subplot(gsp[1, 0])
    # ---- (a) cut at U = UCUT
    km = list(Um).index(UCUT)
    gl = np.linspace(0.005, GMAX, 120)
    seeds = [SEED] + [s for s in (1e-3, 1e-5) if s != SEED]
    Ib, nb = bo_cut(UCUT, gl, seeds)
    for s, ls, lw in zip(seeds, ('-', ':', '--'), (1.3, 1.0, 1.0)):
        a.semilogy(gl, Ib[s], color=C3, lw=lw, ls=ls, label=r'adiabatic, seed $%s$' % sci(s))
    a.semilogy(gs[1:], np.maximum(Im[km][1:], 1e-9), color=C2, label='mean field')
    if cut:
        a.semilogy([r['g'] for r in cut], [r['I'] for r in cut], 'o', ms=3.3, mfc='none', mew=0.8, color=C1, zorder=5,
                   label=r'full DMRG (fermionic $\hat I$), seed $%s$' % sci(SEED))
        b.plot([r['g'] for r in cut], [r['n'] for r in cut], 'o', ms=3.0, color=C1, zorder=5)
    a.set_ylim(1e-8, 5e-2); a.set_xlim(0, GMAX); a.tick_params(labelbottom=False)
    a.set_ylabel(r'$|\langle I\rangle|$ $(et/\hbar)$', fontsize=7)
    a.set_title(r'$\hbar\omega=%g\,t$, $U=%gt$' % (OM, UCUT), fontsize=7.5); tag(a, '(a)')
    b.plot(gl, nb, color=C3); b.plot(gs, nm[km], color=C2)
    b.set_xlabel('$g$'); b.set_ylabel(r'$\langle a^\dagger a\rangle$', fontsize=7); b.set_xlim(0, GMAX)
    # ---- (b) full-DMRG map of the fermionic current (coarse grid)
    norm = LogNorm(1e-2 * SEED, 3e-2)        # floor: far below the linear response to the seed
    axs = [fig.add_subplot(gsp[:, c]) for c in (1, 2, 3)]
    if rows:
        Ug, gg = sorted(set(r['U'] for r in rows)), sorted(set(round(r['g'], 3) for r in rows if abs(r['g'] * 10 - round(r['g'] * 10)) < 1e-6))
        M = np.full((len(Ug), len(gg)), np.nan)
        for r in rows:
            if round(r['g'], 3) in gg:
                M[Ug.index(r['U']), gg.index(round(r['g'], 3))] = r['I']
        pc0 = axs[0].pcolormesh(gg, Ug, M, cmap='Blues', norm=norm, shading='nearest', rasterized=True)
        for U in Ug:
            axs[0].plot([GMAX], [U], '<', ms=2.3, color=MUTED, clip_on=False)
    else:
        pc0 = axs[0].pcolormesh(gs, Ue, I3, cmap='Blues', norm=norm, shading='nearest', rasterized=True)
    axs[0].set_title(r'full DMRG $|\langle I\rangle|$, seed $%s$' % sci(SEED), fontsize=7.0, loc='right')
    # ---- (c) tunnel splitting (adiabatic theory)
    pc1 = axs[1].pcolormesh(gs, Ue, dl, cmap='Blues_r', norm=LogNorm(1e-3, 1), shading='nearest', rasterized=True)
    cs = axs[1].contour(gs, Ue, dl, levels=[1e-2, 1e-1, 0.5], colors='white', linewidths=0.6)
    axs[1].clabel(cs, fontsize=5, fmt=lambda v: '%g' % v)
    axs[1].set_title(r'tunnel splitting $\delta/\hbar\omega$', fontsize=7.0, loc='right')
    # ---- (d) mean field, same colour scale as (b)
    pc2 = axs[2].pcolormesh(gs, Um, np.maximum(Im, 1e-9), cmap='Blues', norm=norm, shading='nearest', rasterized=True)
    nmf = np.where(Um[:, None] >= 0.9, nm, np.nan)
    axs[2].contour(gs, Um, nmf, levels=[1.0], colors=C2, linewidths=0.9, linestyles='--')
    axs[2].set_title('mean field', fontsize=7.0, loc='right')
    Uc = 0.91
    for ax, lab in zip(axs, ('(b)', '(c)', '(d)')):
        ok = Ue >= Uc
        ax.plot(gcl[ok], Ue[ok], color=C4, ls='--', lw=0.9)
        ax.axhline(Uc, color=MUTED, lw=0.5, ls=':')
        ax.set_xlabel('$g$'); ax.set_xlim(0, GMAX); ax.set_ylim(0, 10); tag(ax, lab)
        ax.text(GMAX / 2, 0.3, 'singlet (0-junction)', fontsize=5.3, color=MUTED, ha='center')
    axs[0].set_ylabel(r'$U/t$')
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)
    for pc, ax, lab in ((pc0, axs[0], r'$|\langle I\rangle|$ $(et/\hbar)$'), (pc1, axs[1], r'$\delta/\hbar\omega$'),
                        (pc2, axs[2], r'$|\langle I\rangle|_{\rm MF}$ $(et/\hbar)$')):
        cb = fig.colorbar(pc, ax=ax, pad=0.02, aspect=28, location='bottom', shrink=0.9)
        cb.set_label(lab, fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.legend(*a.get_legend_handles_labels(), loc='outside upper center', ncol=5, fontsize=5.6, frameon=False)
    fig.savefig(F + 'fig_phase_diagram_om%s.pdf' % TAG); fig.savefig(F + 'fig_phase_diagram_om%s.png' % TAG)
    plt.close(fig)
    return rows


if __name__ == '__main__':
    rows = fig()
    print('DMRG points:', len(rows))
    for r in sorted(rows, key=lambda r: (r['U'], r['g'])):
        print('U=%.1f g=%.2f chi=%d |I|=%.3e |I_vir|=%.3e dI_chi=%.1e n=%.3f %s' % (
            r['U'], r['g'], r['chi'], r['I'], r['I_vir'], r['dI_chi'], r['n'],
            'delta/om=%.3f' % r['delta'] if 'delta' in r else ''))
