"""Full-DMRG (U, g) phase diagram figure from data/pd/U*.json (run_dmrg_phase_diagram.py) plus the symmetric
U=2 runs of dmrg_udot_U2_r10.json.  No approximation beyond DMRG itself: the order parameter is read off the
flux distribution P(Phi) of the ground state (photon reduced density matrix): Phi* = position of its maximum,
I* = hbar*omega*X*/(2g).  Also <a^dag a>.  Output: figs/fig_dmrg_phase_diagram.pdf/png + data/pd/summary.json"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from make_figures import photon_xdist, C1, C2, C3, C4, INK, MUTED

D, F = '../data/', '../figs/'
OM = 7.61e-4


def analyse(row, g, om=OM):
    x, p = photon_xdist(row['rho_ph'])
    # bimodality test: P(Phi) has a dip at zero (cat / condensate) or a maximum (vacuum-like)
    p0 = p[np.argmin(np.abs(x))]
    half = p[x > 0]; xh = x[x > 0]
    xpk = xh[np.argmax(half)] if p0 < 0.9 * p.max() else 0.0
    return dict(g=g, n=row['n'], X2=row.get('X2', row.get('X20')), xpeak=xpk, Phi=g * xpk / (np.pi / 2),
                I=om * xpk / (2 * g), x=x, p=p, E0=row['E0'], chi=row.get('chi_max', None), t=row.get('time', 0))


def load():
    pts = {}
    for f in sorted(glob.glob(D + 'pd/U*.json')):
        d = json.load(open(f))
        for r in d['rows']:
            pts.setdefault(d['U'], {})[round(r['g'], 4)] = analyse(r, r['g'], d['omega'])
    d = json.load(open(D + 'dmrg_udot_U2_r10.json'))
    for r in d['rows']:
        rr = dict(r, rho_ph=r['rho_ph'], n=r['n0'], X2=r['X20'])
        pts.setdefault(2.0, {})[round(r['g'], 4)] = analyse(rr, r['g'], d['omega'])
    return pts


def main():
    pts = load()
    Us = sorted(pts)
    ex = json.load(open(D + 'exact_phase_diagram.json'))
    gs, Ue, Ie = np.array(ex['gs']), np.array(ex['fine']['Us']), np.abs(np.array(ex['fine']['I']))
    mf = json.load(open(D + 'mf_hf_grid_fine.json')); gm, Um, Im = np.array(mf['gs']), np.array(mf['Us'][:len(mf['I'])]), np.abs(np.array(mf['I']))

    fig = plt.figure(figsize=(7.0, 2.7), constrained_layout=True)
    gsp = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])
    ax = fig.add_subplot(gsp[0, 0])
    allI = [v['I'] for U in Us for v in pts[U].values()]
    norm = Normalize(0, max(allI))
    for U in Us:
        for g, v in sorted(pts[U].items()):
            ax.scatter([g], [U], c=[v['I']], cmap='Blues', norm=norm, marker='s', s=70, edgecolors=INK, linewidths=0.4, zorder=3)
    sc = ax.scatter([], [], c=[], cmap='Blues', norm=norm)
    ax.contour(gs, Ue, Ie, levels=[2e-3], colors=C3, linewidths=1.0)
    ax.contour(gm, Um, Im, levels=[2e-3], colors=C2, linewidths=1.0, linestyles='--')
    ax.axhline(0.9, color=MUTED, lw=0.6, ls=':')
    ax.set_xlim(0.02, 0.44); ax.set_ylim(0, 9); ax.set_xlabel('$g$'); ax.set_ylabel('$U/t$')
    ax.set_title('full DMRG: $I^\\star$ from $P(\\Phi)$', fontsize=8)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, aspect=22); cb.set_label(r'$I^\star$  $(et/\hbar)$')
    ax.text(0.03, 0.3, '0-junction', fontsize=6, color=MUTED)
    ax.text(-0.22, 1.02, '(a)', transform=ax.transAxes, fontweight='bold')

    bx = fig.add_subplot(gsp[0, 1])
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(Us)))
    for c, U in zip(cols, Us):
        gg = sorted(pts[U]); bx.plot(gg, [pts[U][g]['n'] for g in gg], 'o-', ms=3, lw=1.1, color=c, label='$U=%g$' % U)
    bx.set_xlabel('$g$'); bx.set_ylabel(r'$\langle a^\dagger a\rangle$'); bx.legend(fontsize=5, ncol=2)
    bx.text(-0.22, 1.02, '(b)', transform=bx.transAxes, fontweight='bold')

    cx = fig.add_subplot(gsp[0, 2])
    U0 = 3.0 if 3.0 in pts else Us[0]
    gg = sorted(pts[U0]); sel = gg[::max(1, len(gg) // 5)]
    cols = plt.cm.Blues(np.linspace(0.35, 1, len(sel)))
    for c, g in zip(cols, sel):
        v = pts[U0][g]; m = np.abs(v['x'] * g / (np.pi / 2)) < 1.3
        cx.plot(v['x'][m] * g / (np.pi / 2), v['p'][m] * (np.pi / 2) / g, color=c, lw=1.2, label='$g=%.2f$' % g)
    cx.set_xlabel(r'$\Phi/(\Phi_0/2)$'); cx.set_ylabel(r'$P(\Phi)$'); cx.legend(fontsize=5); cx.set_title('$U=%g$' % U0, fontsize=8)
    cx.text(-0.22, 1.02, '(c)', transform=cx.transAxes, fontweight='bold')
    fig.savefig(F + 'fig_dmrg_phase_diagram.pdf'); fig.savefig(F + 'fig_dmrg_phase_diagram.png'); plt.close(fig)

    summ = {str(U): [{k: (float(v[k]) if v[k] is not None else None) for k in v if k not in ('x', 'p')} for g, v in sorted(pts[U].items())] for U in Us}
    json.dump(summ, open(D + 'pd/summary.json', 'w'), indent=1)
    for U in Us:
        print('U=%.1f: ' % U + '  '.join('g=%.3f n=%.2f Phi*=%.2f I*=%.4f' % (g, v['n'], v['Phi'], v['I']) for g, v in sorted(pts[U].items())))


if __name__ == '__main__':
    main()
