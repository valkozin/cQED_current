"""Figures for the paper (PRL single/double column)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bo import solve, V_of, dV_of
from scipy.linalg import eigh

C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'   # DMRG, MF, BO, classical/extra
INK, MUTED = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'lines.linewidth': 1.5, 'axes.labelcolor': INK, 'text.color': INK, 'legend.frameon': False,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300,
                     'mathtext.fontset': 'cm', 'font.family': 'serif'})
D = '../data/'
F = '../figs/'
os.makedirs(F, exist_ok=True)


def tag(ax, s):
    ax.text(-0.22, 1.02, s, transform=ax.transAxes, fontweight='bold', va='bottom')


def photon_xdist(rho, npts=None):
    """P(x) of X=a+a^dag from photon reduced density matrix (DVR in X eigenbasis)."""
    rho = np.array(rho[0]) + 1j * np.array(rho[1])
    N = rho.shape[0]
    n = np.arange(1, N)
    X = np.diag(np.sqrt(n), 1) + np.diag(np.sqrt(n), -1)
    x, U = eigh(X)
    p = np.real(np.einsum('ik,ij,jk->k', U.conj(), rho, U))
    w = np.gradient(x)
    return x, p / w


# ---------------------------------------------------------------- Fig. notebook model
def fig_notebook():
    mf = np.load(D + 'mf_notebook_port.npy').real
    bo = np.load(D + 'bo_notebook_model.npy')
    dm = json.load(open(D + 'dmrg_notebook_model.json')) if os.path.exists(D + 'dmrg_notebook_model.json') else []
    om = 0.02
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 2.9), constrained_layout=True)
    a = axs[0, 0]
    E00 = mf[0, 2]
    a.plot(mf[:, 0], (mf[:, 2] - E00) / om, color=C2, label='mean field')
    a.plot(bo[:, 0], (bo[:, 1] - E00) / om, color=C3, label='adiabatic (BO)')
    if dm:
        a.plot([r['g'] for r in dm], [(r['E0'] - E00) / om for r in dm], 'o', ms=3.5, color=C1, label='DMRG')
    a.set_xlabel('$g$'); a.set_ylabel(r'$(E_0-E_0^{g=0})/\hbar\omega$')
    tag(a, '(a)')
    a = axs[0, 1]
    a.plot(mf[:, 0], mf[:, 4], color=C2)
    a.plot(bo[:, 0], bo[:, 6], color=C3)
    a.plot(bo[:, 0], np.sqrt(bo[:, 3]), color=C3, ls='--', lw=1)
    if dm:
        a.plot([r['g'] for r in dm if r['g'] > 0], [om * r['X0'] / (2 * r['g']) for r in dm if r['g'] > 0], 'o', ms=3.5, color=C1)
    a.text(0.03, 0.0175, r'$\sqrt{\langle I^2\rangle}$ (BO)', color=MUTED, fontsize=6.5)
    a.set_xlabel('$g$'); a.set_ylabel(r'$\langle I\rangle$ $(e t/\hbar)$'); tag(a, '(b)')
    a = axs[1, 0]
    a.plot(bo[:, 0], bo[:, 2] / om, color=C3)
    if dm:
        a.plot([r['g'] for r in dm], [(r['E1'] - r['E0']) / om for r in dm], 'o', ms=3.5, color=C1)
    a.set_ylim(0, 1.1); a.set_xlabel('$g$'); a.set_ylabel(r'$(E_1-E_0)/\hbar\omega$'); tag(a, '(c)')
    a = axs[1, 1]
    a.plot(mf[:, 0], mf[:, 3], color=C2)
    a.plot(bo[:, 0], bo[:, 5], color=C3)
    if dm:
        a.plot([r['g'] for r in dm], [r['n0'] for r in dm], 'o', ms=3.5, color=C1)
    a.set_xlabel('$g$'); a.set_ylabel(r'$\langle a^\dagger a\rangle$'); tag(a, '(d)')
    a.legend(*axs[0, 0].get_legend_handles_labels(), fontsize=6.5, loc='upper left')
    fig.savefig(F + 'fig_notebook_model.pdf'); fig.savefig(F + 'fig_notebook_model.png')
    plt.close(fig)


# ---------------------------------------------------------------- Fig. half-flux biased weak link
def fig_weaklink(ratio=10):
    bo = json.load(open(D + 'bo_weaklink.json'))
    c = np.array(bo['c'][0]) + 1j * np.array(bo['c'][1])
    EJ = bo['EJ']
    # true curvature-based beta: beta_true = beta_nom * |E''(pi/2)| / (4 EJ)
    m = np.arange(len(c))
    Epp = np.real(np.sum(-(m ** 2) * c * np.exp(1j * m * np.pi / 2)))
    scale = abs(Epp) / (4 * EJ)
    Ic = 2 * EJ  # critical (pair) current scale in units e t/hbar (electron phase): max|dE/dphi| ~ 2 EJ
    betas = np.array(bo['betas']) * scale
    om = EJ / ratio
    tilt = np.array(bo['runs']['tilt_%d' % ratio]); sym = np.array(bo['runs']['sym_%d' % ratio])
    fn = D + 'dmrg_weaklink_r%d.json' % ratio
    dm = json.load(open(fn))['rows'] if os.path.exists(fn) else []
    dm = [r for r in dm if r['beta'] > 0]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1), constrained_layout=True)
    a = axs[0]
    a.plot(betas, np.abs(tilt[:, 6]) / Ic, color=C4, lw=1, ls='--', label='classical (Landau)')
    a.plot(betas, np.abs(tilt[:, 3]) / Ic, color=C3, label='BO')
    a.plot(betas, np.sqrt(sym[:, 4]) / Ic, color=C3, lw=1, ls=':', label=r'BO, $\sqrt{\langle I^2\rangle}$, no $h/e$')
    if dm:
        b = np.array([r['beta'] for r in dm]) * scale
        a.plot(b, [abs(r['I_mf']) / Ic for r in dm], 's', ms=3.5, mfc='none', color=C2, label='mean field')
        a.plot(b, [abs(om * r['X0'] / (2 * r['g'])) / Ic for r in dm], 'o', ms=3.5, color=C1, label='DMRG')
    a.set_xlabel(r'$\beta$'); a.set_ylabel(r'$|\langle I\rangle|/I_c$'); a.legend(fontsize=6, loc='lower right')
    a.set_xlim(0, betas.max()); tag(a, '(a)')
    a = axs[1]
    a.semilogy(betas, tilt[:, 2] / om, color=C3, label=r'BO, with $E_1$ tilt')
    a.semilogy(betas, np.maximum(sym[:, 2] / om, 1e-12), color=C3, ls=':', lw=1, label=r'BO, $E_{\rm odd}=0$')
    if dm:
        a.semilogy(b, [(r['E1'] - r['E0']) / om for r in dm], 'o', ms=3.5, color=C1, label='DMRG')
    a.set_ylim(1e-9, 2); a.set_xlabel(r'$\beta$'); a.set_ylabel(r'$(E_1-E_0)/\hbar\omega$')
    a.legend(fontsize=6, loc='lower left'); a.set_xlim(0, betas.max()); tag(a, '(b)')
    a = axs[2]
    cols = ['#9ec5f0', '#5a9be3', C1, '#1a4f91']
    k = 0
    for r in dm:
        if r['beta'] in (0.5, 1.0, 1.5, 3.0):
            x, p = photon_xdist(r['rho_ph'])
            a.plot(r['g'] * x / (np.pi / 2), p * (np.pi / 2) / r['g'], color=cols[k % 4], lw=1.2, label=r'$\beta=%.2f$' % (r['beta'] * scale))
            k += 1
    a.set_xlim(-1.3, 1.3); a.set_xlabel(r'$\Phi/(\Phi_0/2)$'); a.set_ylabel(r'$P(\Phi)$ (DMRG)')
    a.legend(fontsize=6); tag(a, '(c)')
    fig.savefig(F + 'fig_weaklink_halfflux.pdf'); fig.savefig(F + 'fig_weaklink_halfflux.png')
    plt.close(fig)


# ---------------------------------------------------------------- Fig. junction physics
def fig_junctions():
    from bdg_general import sc_ring, bdg_energy
    from udot_harmonics import load, harmonics_from_half
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
    a = axs[0]
    ph = np.linspace(-np.pi, np.pi, 241)
    cnb = np.load(D + 'harmonics_notebook_model.npy')
    E = V_of(cnb, ph); E -= E[120]
    a.plot(ph / np.pi, E / np.abs(E).max(), color=C2, label='Zeeman-polarized dot\n(notebook model)')
    pw = sc_ring(14, 1.0, 1.0, 0.5)
    E = np.array([bdg_energy(pw, x + np.pi / 2) for x in ph]); E -= E[120]
    a.plot(ph / np.pi, E / np.abs(E).max(), color=C3, label=r'weak link + $\Phi_0/2$')
    phi, Eu = load(D + 'udot_phi_tJ0.5.json', 2.0)
    cu = harmonics_from_half(phi, Eu)
    E = V_of(cu, ph); E -= E[120]
    sc = np.abs(E).max()
    a.plot(ph / np.pi, E / sc, color=C1, lw=1, label=r'interacting dot $U=2t$' + '\n(DMRG, no flux)')
    a.plot(np.r_[-phi[::-1], phi] / np.pi, (np.r_[Eu[::-1], Eu] - Eu[0]) / sc, 'o', ms=2.2, color=C1)
    a.axhline(0, color=MUTED, lw=0.4)
    a.set_ylim(-1.05, 1.0)
    a.set_xlabel(r'$\varphi/\pi$  (flux $/\,\Phi_0$)'); a.set_ylabel(r'$[E(\varphi)-E(0)]/\max|\Delta E|$')
    a.legend(fontsize=5.6, loc='upper center', ncol=3, handlelength=1.0, columnspacing=0.6, borderaxespad=0.1); tag(a, '(a)')
    rows = json.load(open(D + 'udot_scan_tJ0.5.json')) + json.load(open(D + 'udot_scan2_tJ0.5.json'))
    Us = np.array(sorted(set(r['U'] for r in rows)))
    get = lambda U, p, s: [r[s] for r in rows if r['U'] == U and abs(r['phi'] - p) < 1e-3][0]
    a = axs[1]
    g0 = np.array([get(U, 0, 'odd') - get(U, 0, 'even') for U in Us])
    g1 = np.array([get(U, np.pi / 2, 'odd') - get(U, np.pi / 2, 'even') for U in Us])
    a.plot(Us, g0, 'o-', ms=3, color=C1, label=r'$\varphi=0$')
    a.plot(Us, g1, 's-', ms=3, color=C4, label=r'$\varphi=\pi/2$ ($\Phi_0/2$)')
    a.axhline(0, color=MUTED, lw=0.4)
    Uc = np.interp(0, -g0, Us)
    a.axvline(Uc, color=MUTED, lw=0.6, ls='--')
    a.text(Uc + 0.25, -0.95, r'$U_c\approx%.1f\,t$' % Uc, fontsize=6.5, color=MUTED)
    a.set_xlabel(r'$U/t$'); a.set_ylabel(r'$E_{\rm odd}-E_{\rm even}$  $(t)$')
    a.legend(fontsize=6, loc='upper right', borderaxespad=0.1); tag(a, '(b)')
    a = axs[2]
    dEe = np.array([get(U, np.pi / 2, 'even') - get(U, 0, 'even') for U in Us]) / 2
    dEo = np.array([get(U, np.pi / 2, 'odd') - get(U, 0, 'odd') for U in Us]) / 2
    a.semilogy(Us, dEe, 'o-', ms=3, color=C3, label=r'singlet: $0$-junction')
    a.semilogy(Us, -dEo, 's-', ms=3, color=C1, label=r'doublet: $\pi$-junction')
    a.axvspan(Uc, Us.max(), color=C1, alpha=0.07, lw=0)
    a.set_xlabel(r'$U/t$'); a.set_ylabel(r'$|E_2|\simeq|E(\pi/2)-E(0)|/2$  $(t)$')
    a.legend(fontsize=6, loc='lower left'); tag(a, '(c)')
    fig.savefig(F + 'fig_junctions.pdf'); fig.savefig(F + 'fig_junctions.png')
    plt.close(fig)
    print('U_c(phi=0) ~ %.2f' % Uc, 'E2 odd:', dict(zip(Us, np.round(-dEo, 5))))


# ---------------------------------------------------------------- Fig. interaction-driven pi ring + cavity
def classical_theta(c, betas, EJ, om):
    Xg = np.linspace(0, np.pi, 20001)  # theta_e = g X
    out = []
    for b in betas:
        g = np.sqrt(b * om / (8 * EJ))
        U = om * (Xg / g) ** 2 / 4 + V_of(c, Xg)
        out.append(Xg[np.argmin(U)])
    return np.array(out)


def fig_udot():
    bo = json.load(open(D + 'bo_udot_U2.json'))
    c = np.load(D + 'harmonics_udot_U2.npy')
    EJ = bo['EJ']; betas = np.array(bo['betas'])
    fn = D + 'dmrg_udot_U2_r10.json'
    dm = json.load(open(fn))['rows'] if os.path.exists(fn) else []
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
    ratios = ['2', '5', '10', '40']
    shades = ['#bcd6f5', '#7fb0ec', C1, '#163f75']
    a = axs[0]
    a.plot(betas, classical_theta(c, betas, EJ, EJ / 10) / (np.pi / 2), color=C4, ls='--', lw=1, label='classical')
    for r, col in zip(ratios, shades):
        v = np.array(bo['runs'][r])
        a.plot(betas, v[:, 4] * v[:, 0] / (np.pi / 2), color=col, lw=1.2, label=r'$E_J/\hbar\omega=%s$' % r)
    if dm:
        a.plot([r['beta'] for r in dm], [r['X01'] * r['g'] / (np.pi / 2) for r in dm], 'o', ms=3.5, color=C2,
               label=r'DMRG, $E_J/\hbar\omega=10$')
    a.set_xlabel(r'$\beta$'); a.set_ylabel(r'$\Phi^\star/(\Phi_0/2)$'); a.legend(fontsize=5.8, loc='upper left')
    tag(a, '(a)')
    a = axs[1]
    for r, col in zip(ratios, shades):
        v = np.array(bo['runs'][r]); om = EJ / float(r)
        a.semilogy(betas, np.maximum(v[:, 2] / om, 3e-9), color=col, lw=1.2)
    if dm:
        res = 4e-2  # DMRG resolution of excited-state energies (~3e-5 t, from chi=256 vs 400 at beta=1.5)
        ok = [r for r in dm if (r['E1'] - r['E0']) / (EJ / 10) > 5 * res]
        lim = [r for r in dm if (r['E1'] - r['E0']) / (EJ / 10) <= 5 * res]
        a.semilogy([r['beta'] for r in ok], [(r['E1'] - r['E0']) / (EJ / 10) for r in ok], 'o', ms=3.5, color=C2)
        a.semilogy([r['beta'] for r in lim], [res] * len(lim), 'v', ms=4, mfc='none', color=C2)
        a.text(1.75, 1.2e-2, 'DMRG: below\nresolution', fontsize=5.5, color=C2)
    a.axhspan(1e-9, 1e-8, color=MUTED, alpha=0.08, lw=0)
    a.text(0.1, 2e-9, 'BO numerical floor', fontsize=6, color=MUTED)
    a.set_ylim(1e-9, 2); a.set_xlabel(r'$\beta$'); a.set_ylabel(r'$\delta/\hbar\omega$'); tag(a, '(b)')
    a = axs[2]
    cols = ['#9ec5f0', '#5a9be3', C1, '#1a4f91', '#0d2a52']
    k = 0
    for r in dm:
        x, p = photon_xdist(r['rho_ph'])
        a.plot(r['g'] * x / (np.pi / 2), p * (np.pi / 2) / r['g'], color=cols[k % 5], lw=1.2,
               label=r'$\beta=%.1f$' % r['beta'])
        k += 1
    a.set_xlim(-1.3, 1.3); a.set_xlabel(r'$\Phi/(\Phi_0/2)$'); a.set_ylabel(r'$P(\Phi)$ (DMRG)')
    a.legend(fontsize=6); tag(a, '(c)')
    fig.savefig(F + 'fig_udot_cavity.pdf'); fig.savefig(F + 'fig_udot_cavity.png')
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1: schematic + phase diagram
def fig_overview():
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    pdg = json.load(open(D + 'bo_phase_diagram_U2.json'))
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.75), constrained_layout=True, gridspec_kw=dict(width_ratios=[0.8, 1.2]))
    a = axs[0]; a.set_aspect('equal'); a.axis('off')
    a.set_xlim(-1.45, 1.35); a.set_ylim(-1.3, 1.3)
    # loop: right arc superconductor, left side is the shared inductor of the LC mode
    th = np.linspace(-np.pi / 2 + 0.25, np.pi / 2 - 0.25, 100)
    a.plot(0.9 * np.cos(th), 0.9 * np.sin(th), color=INK, lw=1.6)
    a.plot([0, -0.6], [0.9, 0.9], color=INK, lw=1.6); a.plot([0, -0.6], [-0.9, -0.9], color=INK, lw=1.6)
    a.plot([0.9 * np.cos(th[-1]), 0], [0.9 * np.sin(th[-1]), 0.9], color=INK, lw=1.6)
    a.plot([0.9 * np.cos(th[0]), 0], [0.9 * np.sin(th[0]), -0.9], color=INK, lw=1.6)
    # coil on the left branch
    y = np.linspace(-0.6, 0.6, 400)
    a.plot(-0.6 + 0.12 * np.sin(2 * np.pi * 5 * (y + 0.6) / 1.2), y, color=C1, lw=1.3)
    a.plot([-0.6, -0.6], [0.6, 0.9], color=INK, lw=1.6); a.plot([-0.6, -0.6], [-0.9, -0.6], color=INK, lw=1.6)
    # capacitor in parallel
    a.plot([-0.6, -1.2], [0.75, 0.75], color=C1, lw=1.1); a.plot([-0.6, -1.2], [-0.75, -0.75], color=C1, lw=1.1)
    a.plot([-1.2, -1.2], [0.75, 0.08], color=C1, lw=1.1); a.plot([-1.2, -1.2], [-0.75, -0.08], color=C1, lw=1.1)
    a.plot([-1.36, -1.04], [0.08, 0.08], color=C1, lw=1.6); a.plot([-1.36, -1.04], [-0.08, -0.08], color=C1, lw=1.6)
    a.text(-1.42, 0.25, '$C$', color=C1, fontsize=7); a.text(-0.42, -0.1, '$L$', color=C1, fontsize=7)
    # junction (dot) on the right
    a.add_patch(Rectangle((0.78, -0.12), 0.24, 0.24, fc='white', ec=INK, lw=1.0, zorder=3))
    a.add_patch(Circle((0.9, 0), 0.06, fc=C2, ec='none', zorder=4))
    a.text(1.08, 0.13, r'$\pi$', fontsize=8); a.text(1.06, -0.3, r'$U$', fontsize=6.5, color=C2)
    a.add_patch(FancyArrowPatch((0.45, 0.5), (0.45, -0.5), connectionstyle='arc3,rad=-0.7',
                                arrowstyle='<->', mutation_scale=6, color=MUTED, lw=0.8))
    a.text(0.05, -0.04, r'$\pm I$', fontsize=6.5, color=MUTED)
    a.text(-1.35, 1.05, r'$\hbar\omega a^\dagger a$', fontsize=6.5, color=C1)
    tag(a, '(a)')
    a = axs[1]
    b = np.array(pdg['betas']); r = np.array(pdg['ratios']); S = np.array(pdg['delta'])
    lev = np.arange(-8, 0.5, 1.0)
    cs = a.contourf(b, r, np.log10(np.clip(S, 1.01e-8, 1)), levels=lev, cmap='Blues_r')
    a.contour(b, r, np.log10(np.clip(S, 1e-9, 1)), levels=[-6, -3, -1], colors='white', linewidths=0.5)
    a.set_yscale('log'); a.axvline(1, color=INK, lw=0.6, ls='--')
    a.set_xlabel(r'$\beta=2\pi LI_c/\Phi_0$'); a.set_ylabel(r'$E_J/\hbar\omega$')
    cb = fig.colorbar(cs, ax=a, pad=0.02, aspect=15); cb.set_label(r'$\log_{10}\delta/\hbar\omega$', fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    a.text(1.3, 0.75, 'QD + superinductor', fontsize=5.5, color=INK)
    a.text(1.4, 20, 'multichannel $\\pi$ JJ', fontsize=5.5, color='white')
    a.annotate('SFS', xy=(3.0, 98), xytext=(3.05, 45), fontsize=5.5, color='white',
               arrowprops=dict(arrowstyle='->', color='white', lw=0.6))
    tag(a, '(b)')
    fig.savefig(F + 'fig_overview.pdf'); fig.savefig(F + 'fig_overview.png')
    plt.close(fig)


# ---------------------------------------------------------------- Fig. main: I(g) and (U,g) phase diagrams
def _istar(c, g, om, Nph=160):
    """magnitude of the spontaneous current |I*| = hbar*omega X*/2g, X* = flux eigenvalue in the lowest doublet
    (rotation-invariant; equals |<I>| of the symmetry-broken state), and photon number; no seed."""
    from bo import solve
    if g == 0:
        return 0.0, 0.0
    o = solve(c, g, om, 0.0, Nphot=Nph)
    return om * o['X01'] / (2 * g), o['n'][0]


def fig_main():
    ex = json.load(open(D + 'exact_phase_diagram.json'))
    mff = D + 'mf_hf_grid_fine.json' if os.path.exists(D + 'mf_hf_grid_fine.json') else D + 'mf_hf_grid.json'
    mf = json.load(open(mff)); mf2 = json.load(open(D + 'mf_hf_grid.json'))
    om = ex['omega']; gs = np.array(ex['gs'])
    Ue = np.array(ex['fine']['Us'])
    Ie = np.abs(np.array(ex['fine']['I']))
    Um = np.array(mf['Us'][:len(mf['I'])]); Im = np.abs(np.array(mf['I'])); gm = np.array(mf['gs'])
    fig = plt.figure(figsize=(7.0, 2.6), constrained_layout=True)
    gsp = fig.add_gridspec(2, 3, width_ratios=[1.0, 1, 1])
    a, b = fig.add_subplot(gsp[0, 0]), fig.add_subplot(gsp[1, 0])
    c = np.load(D + 'harmonics_udot_U2.npy')
    gg = np.linspace(0.005, 0.4, 120)
    from bo import solve
    res = np.array([_istar(c, x, om) for x in gg])
    for seed, ls, lab in ((1e-3, '-', r'exact, seed $10^{-3}$'), (1e-5, '--', r'exact, seed $10^{-5}$')):
        a.plot(gg, [abs(solve(c, x, om, seed, Nphot=160)['I'][0]) for x in gg], color=C3, ls=ls, lw=1.3 if ls == '-' else 1.0, label=lab)
    k = list(mf2['Us']).index(2.0)
    a.plot(mf2['gs'], np.abs(mf2['I'][k]), color=C2, label='mean field')
    b.plot(gg, res[:, 1], color=C3); b.plot(mf2['gs'], mf2['n'][k], color=C2)
    sc = json.load(open(D + 'dmrg_udot_U2_r10.json'))['rows']
    gsc = [r for r in json.load(open(D + 'gscan_udot_U2_seed1e-3_part1.json'))['rows']] + \
          [r for r in json.load(open(D + 'gscan_udot_U2_seed1e-3_part2.json'))['rows']] if os.path.exists(D + 'gscan_udot_U2_seed1e-3_part2.json') else []
    pts = [(r['g'], abs(r['I_vir'])) for r in gsc if r['n'] < 1.0] + \
          [(r['g'], om * r['X01'] / (2 * r['g'])) for r in sc if (r['E1'] - r['E0']) < 0.05 * om]
    a.plot(*zip(*sorted(pts)), 'o', ms=3.5, color=C1, label='full DMRG', zorder=5)
    dm = sc + sum([json.load(open(D + f))['rows'] for f in ('gscan_udot_U2_seed1e-3_part1.json', 'gscan_udot_U2_seed1e-3_part2.json',
                   'gscan_udot_U2_seed1e-5.json') if os.path.exists(D + f)], [])
    b.plot([r['g'] for r in dm], [r.get('n0', r.get('n')) for r in dm], 'o', ms=3.0, color=C1, zorder=5)
    a.set_ylabel(r'$|\langle I\rangle|$ $(et/\hbar)$', fontsize=7); a.set_xlim(0, 0.4); a.tick_params(labelbottom=False)
    a.legend(fontsize=5.5, loc='upper right'); a.set_title(r'$U=2t$', fontsize=8); tag(a, '(a)')
    b.set_xlabel('$g$'); b.set_ylabel(r'$\langle a^\dagger a\rangle$', fontsize=7); b.set_xlim(0, 0.4)
    vmax = max(Ie.max(), Im.max())
    axm = []
    for col, U, I, g, title, lab in ((1, Ue, Ie, gs, 'exact (DMRG + quantum photon)', '(b)'),
                                      (2, Um, Im, gm, 'mean field (HF + product state)', '(c)')):
        ax = fig.add_subplot(gsp[:, col]); axm.append(ax)
        pc = ax.pcolormesh(g, U, I, cmap='Blues', vmin=0, vmax=vmax, shading='nearest', rasterized=True)
        ax.set_xlabel('$g$'); ax.set_ylabel(r'$U/t$'); ax.set_title(title, fontsize=7.5, loc='right'); tag(ax, lab)
        ax.set_xlim(0, 0.4); ax.set_ylim(0, 10)
    axm[0].contour(gm, Um, Im, levels=[2e-3], colors=C2, linewidths=1.0, linestyles='--')
    axm[0].contour(gs, Ue, Ie, levels=[2e-3], colors=INK, linewidths=0.7)
    axm[1].contour(gs, Ue, Ie, levels=[2e-3], colors=INK, linewidths=0.7)
    for U in ex['Us']:
        axm[0].plot([0.4], [U], '<', ms=2.5, color=MUTED, clip_on=False)
    axm[0].text(0.2, 0.3, 'no current (0-junction)', fontsize=5.5, color=MUTED)
    cb = fig.colorbar(pc, ax=axm, pad=0.02, aspect=25); cb.set_label(r'$|\langle I\rangle|$  $(et/\hbar)$')
    fig.savefig(F + 'fig_main_phase_diagram.pdf'); fig.savefig(F + 'fig_main_phase_diagram.png')
    plt.close(fig)


def fig_notebook_omega():
    """Notebook model: is there a transition? omega = 0.02 (original) vs 0.002."""
    c = np.load(D + 'harmonics_notebook_model.npy')
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 3.0), constrained_layout=True, sharex=True)
    for j, (om, mff, dmfs, title) in enumerate(((0.02, 'mf_notebook_port.npy', ['dmrg_notebook_model.json'], r'$\hbar\omega=0.02$ (as in the notebook)'),
                                                 (0.002, 'mf_notebook_om0.002.npy', ['gscan_notebook_om0.002_part1.json', 'gscan_notebook_om0.002_part2.json'], r'$\hbar\omega=0.002$'))):
        a, b = axs[0, j], axs[1, j]
        gg = np.linspace(0.01, 1.2, 120)
        from bo import solve
        res = [solve(c, x, om, 1e-3, Nphot=220) for x in gg]
        a.plot(gg, [abs(o['I'][0]) for o in res], color=C3, label=r'exact, seed $10^{-3}$')
        b.plot(gg, [o['n'][0] for o in res], color=C3)
        m = np.load(D + mff).real
        a.plot(m[:, 0], np.abs(m[:, 4]), color=C2, label='mean field (notebook)')
        b.plot(m[:, 0], m[:, 3], color=C2)
        rows = []
        for f in dmfs:
            if os.path.exists(D + f):
                d = json.load(open(D + f)); rows += d['rows'] if isinstance(d, dict) else d
        b.plot([r['g'] for r in rows], [r.get('n0', r.get('n')) for r in rows], 'o', ms=3.2, color=C1, label='full DMRG')
        a.set_title(title, fontsize=8); a.set_ylabel(r'$|I|$'); b.set_ylabel(r'$\langle a^\dagger a\rangle$'); b.set_xlabel('$g$')
        a.legend(fontsize=6); b.legend(fontsize=6)
    tag(axs[0, 0], '(a)'); tag(axs[0, 1], '(b)')
    fig.savefig(F + 'fig_notebook_omega.pdf'); fig.savefig(F + 'fig_notebook_omega.png')
    plt.close(fig)


if __name__ == '__main__':
    import sys
    for name in (sys.argv[1:] or ['main', 'notebook_omega', 'overview', 'junctions', 'notebook', 'weaklink', 'udot']):
        globals()['fig_' + name]()
        print('done', name)
