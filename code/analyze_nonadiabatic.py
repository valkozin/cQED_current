"""Cavity-driven 0-pi transition: analysis of data/nonadiabatic/grid_om*.json (full DMRG, both parity sectors,
phi_cl = 0, no seed) against two limits:
  * adiabatic (Born-Oppenheimer): E_s^BO(g, omega; U) = ground energy of omega a^dag a + E_s(g X; U) with the
    g = 0 surfaces E_s(phi; U) of data/effective_Nr10.json (doublet: 4 harmonics; singlet: spline where available);
  * anti-adiabatic (Lang-Firsov): the mode averages the Peierls factor, t_J -> t_J exp(-g^2/2), i.e. the ring at g = 0
    with a reduced junction coupling (data/nonadiabatic/lf_g*.json).
Output: data/nonadiabatic/summary.json (U_c(g, omega) from the sign change of E_odd - E_even, linear in U between
grid points), and figs/fig_cavity_0pi.pdf."""
import json, glob, os, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from bo import solve, solve_V
from effective_grid_lib import spline_V

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../figs')
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'legend.fontsize': 6.8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.3, 'legend.frameon': False, 'figure.dpi': 150})


def crossing(Us, dE):
    """U where dE(U) = E_odd - E_even changes sign (linear interpolation); None if no sign change"""
    Us, dE = np.array(Us), np.array(dE)
    o = np.argsort(Us); Us, dE = Us[o], dE[o]
    for i in range(len(Us) - 1):
        if dE[i] > 0 >= dE[i + 1] or dE[i] >= 0 > dE[i + 1]:
            return float(Us[i] - dE[i] * (Us[i + 1] - Us[i]) / (dE[i + 1] - dE[i]))
    return None


def load_grid():
    grid = {}
    for f in sorted(glob.glob(os.path.join(D, 'nonadiabatic', 'grid_om*.json'))):
        rows = json.load(open(f))
        if rows:
            grid[rows[0]['omega']] = rows
    return grid


def load_lf():
    lf = {0.0: None}
    for f in glob.glob(os.path.join(D, 'nonadiabatic', 'lf_g*.json')):
        g = float(os.path.basename(f)[4:-5])
        lf[g] = json.load(open(f))
    return lf


def bo_surfaces():
    """eff: harmonics of both sectors (effective_Nr10.json); fine: raw even-sector points per U node (the singlet
    surface has a cusp at pi/2, so it is represented by a periodic spline through the raw DMRG points: 25 phases
    where available, otherwise the 9 phases of ephi_Nr10_tJ0.5_A.json)"""
    eff = json.load(open(os.path.join(D, 'effective_Nr10.json')))
    fine = {}
    for r in json.load(open(os.path.join(D, 'ephi_Nr10_tJ0.5_A.json'))):
        fine.setdefault(r['U'], []).append((r['phi'], r['even']))
    for f in glob.glob(os.path.join(D, 'ephi_even_fine_*.json')):
        rows = json.load(open(f))
        for U in {r['U'] for r in rows}:
            fine[U] = [(r['phi'], r['even']) for r in rows if r['U'] == U]
    return eff, fine


_harm_interp = {}


def harm_of(eff, U, s):
    """harmonics of E_s(phi; U), interpolated in U (pchip) between the DMRG nodes of effective_Nr10.json"""
    if s not in _harm_interp:
        from scipy.interpolate import PchipInterpolator
        Us = np.array(eff['Us']); C = np.array([eff['harm'][str(u)][s] for u in eff['Us']])
        _harm_interp[s] = PchipInterpolator(Us, C, axis=0)
    return np.array(_harm_interp[s](U), complex)


def E_bo(eff, fine, U, s, g, omega):
    if s == 'even' and U in fine:
        V, dV = spline_V(fine[U])
        if g == 0:
            return float(V(np.array([0.0]))[0])
        return float(solve_V(V, dV, g, omega, 0.0, Nphot=120, nstates=1)['E'][0])
    if g == 0:
        return float(np.sum(harm_of(eff, U, s).real))
    return float(solve(harm_of(eff, U, s), g, omega, 0.0, Nphot=120, nstates=1)['E'][0])


def main():
    grid = load_grid(); lf = load_lf(); eff, fine = bo_surfaces()
    omegas = sorted(grid)
    summary = dict(omegas=omegas, Uc={}, Uc_bo={}, Uc_lf={}, rows={})
    Ueff = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.25]
    gs_all = sorted({r['g'] for om in grid for r in grid[om]})
    # exact
    for om in omegas:
        rows = grid[om]; gs = sorted({r['g'] for r in rows})
        summary['Uc'][str(om)] = {}
        for g in gs:
            rr = [r for r in rows if r['g'] == g]
            summary['Uc'][str(om)][str(g)] = crossing([r['U'] for r in rr], [r['odd'] - r['even'] for r in rr])
        summary['rows'][str(om)] = [{k: r[k] for k in r if not k.startswith('pn_')} for r in rows]
    # BO reference (adiabatic)
    for om in omegas:
        summary['Uc_bo'][str(om)] = {}
        for g in gs_all:
            dE = [E_bo(eff, fine, U, 'odd', g, om) - E_bo(eff, fine, U, 'even', g, om) for U in Ueff]
            summary['Uc_bo'][str(om)][str(g)] = crossing(Ueff, dE)
    # LF reference (anti-adiabatic, omega -> infinity)
    for g, rows in lf.items():
        if rows is None:
            summary['Uc_lf'][str(g)] = summary['Uc_bo'][str(omegas[0])]['0.0']
        else:
            summary['Uc_lf'][str(g)] = crossing([r['U'] for r in rows], [r['odd'] - r['even'] for r in rows])
    json.dump(summary, open(os.path.join(D, 'nonadiabatic', 'summary.json'), 'w'), indent=1)
    for om in omegas:
        print('omega=%.3f: U_c(g) exact:' % om, {g: (None if v is None else round(v, 3)) for g, v in summary['Uc'][str(om)].items()})
        print('             BO:   ', {g: (None if v is None else round(v, 3)) for g, v in summary['Uc_bo'][str(om)].items()})
    print('LF (omega->inf):', {g: (None if v is None else round(v, 3)) for g, v in summary['Uc_lf'].items()})
    make_figure(grid, summary, eff, fine)


def make_figure(grid, summary, eff, fine):
    omegas = sorted(grid)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(omegas)))
    # (a) E_odd - E_even versus g at U = 0.9 for each omega, with BO (dashed) and LF (dotted, omega -> inf)
    ax = axs[0]; U0 = 0.9
    for om, col in zip(omegas, cols):
        rr = sorted([r for r in grid[om] if r['U'] == U0], key=lambda r: r['g'])
        if not rr:
            continue
        ax.plot([r['g'] for r in rr], [r['odd'] - r['even'] for r in rr], 'o-', color=col, ms=3, mfc='w', mew=0.8,
                label=r'$\hbar\omega=%g\Delta$' % om)
        gb = np.linspace(0, 1, 26)
        ax.plot(gb, [E_bo(eff, fine, U0, 'odd', g, om) - E_bo(eff, fine, U0, 'even', g, om) for g in gb], '--', color=col, lw=0.9)
    lfpts = []
    for f in glob.glob(os.path.join(D, 'nonadiabatic', 'lf_g*.json')):
        g = float(os.path.basename(f)[4:-5]); rows = [r for r in json.load(open(f)) if r['U'] == U0]
        if rows:
            lfpts.append((g, rows[0]['odd'] - rows[0]['even']))
    if lfpts:
        lfpts.sort(); ax.plot([p[0] for p in lfpts], [p[1] for p in lfpts], 'k:', lw=1.0, label=r'$t_J e^{-g^2/2}$ ($\omega\to\infty$)')
    ax.axhline(0, color='0.6', lw=0.6); ax.set_xlabel(r'$g$'); ax.set_ylabel(r'$E_{\rm odd}-E_{\rm even}$  $(\Delta)$')
    ax.set_title(r'(a) $U=0.9\Delta$', loc='left', fontsize=8.5); ax.legend(loc='lower left')
    # (b) U_c(g) for each omega: exact (symbols), BO (dashed), LF (dotted)
    ax = axs[1]
    for om, col in zip(omegas, cols):
        pts = [(float(g), v) for g, v in summary['Uc'][str(om)].items() if v is not None]; pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color=col, ms=3, mfc='w', mew=0.8)
        pb = [(float(g), v) for g, v in summary['Uc_bo'][str(om)].items() if v is not None]; pb.sort()
        ax.plot([p[0] for p in pb], [p[1] for p in pb], '--', color=col, lw=0.9)
    pl = [(float(g), v) for g, v in summary['Uc_lf'].items() if v is not None]; pl.sort()
    ax.plot([p[0] for p in pl], [p[1] for p in pl], 'k:', lw=1.0)
    ax.set_xlabel(r'$g$'); ax.set_ylabel(r'$U_c(g)/\Delta$'); ax.set_title('(b) 0\u2013\u03c0 boundary of the loop', loc='left', fontsize=8.5)
    ax.text(0.03, 0.06, r'$\pi$ (doublet) above, $0$ (singlet) below', transform=ax.transAxes, fontsize=6.5)
    # (c) dressing: |<D>| and photon entanglement entropy versus g at U = 0.9 (doublet sector)
    ax = axs[2]
    for om, col in zip(omegas, cols):
        rr = sorted([r for r in grid[om] if r['U'] == U0], key=lambda r: r['g'])
        if not rr:
            continue
        ax.plot([r['g'] for r in rr], [abs(complex(r['ReD_odd'], r['ImD_odd'])) for r in rr], 's-', color=col, ms=2.5, mfc='w', mew=0.8)
        ax.plot([r['g'] for r in rr], [r['S_ph_odd'] for r in rr], '^--', color=col, ms=2.5, mfc='w', mew=0.8)
    gb = np.linspace(0, 1, 50); ax.plot(gb, np.exp(-gb ** 2 / 2), 'k:', lw=0.9)
    ax.set_xlabel(r'$g$'); ax.set_ylabel(r'$|\langle D\rangle|$ (squares), $S_{\rm ph}$ (triangles)')
    ax.set_title(r'(c) dressing, $U=0.9\Delta$', loc='left', fontsize=8.5)
    fig.savefig(os.path.join(F, 'fig_cavity_0pi.pdf')); fig.savefig(os.path.join(F, 'fig_cavity_0pi.png'))


if __name__ == '__main__':
    main()
