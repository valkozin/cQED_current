"""Spin-orbit point and gate-detuning estimate (SM figure fig_so.pdf, numbers for the text).
(a) E_sigma(phi) of the two-orbital dot with the spin-dependent phase alpha: odd_up, odd_dn, even (data/so/ephi_*).
(b) Flux distribution of the odd_up / odd_dn ground states with the mode (data/so/cat_*.json): spin-locked wells.
(c) E_odd - E_even versus the gate detuning of the dot level at g = 0 and g = 0.6 (data/gate/*): width of the
    doublet window in eps_d with and without the zero-point flux."""
import json, glob, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../figs')
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'legend.fontsize': 6.8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.3, 'legend.frameon': False, 'figure.dpi': 150})


def fit_series(phi, E, M=3):
    """E(phi) = sum_m a_m cos(m phi) + b_m sin(m phi)"""
    cols = [np.cos(m * phi) for m in range(M + 1)] + [np.sin(m * phi) for m in range(1, M + 1)]
    A = np.array(cols).T
    c, *_ = np.linalg.lstsq(A, E, rcond=None)
    return c[:M + 1], np.concatenate([[0.0], c[M + 1:]])


def load_ephi(alpha):
    rows = []
    for f in glob.glob(os.path.join(D, 'so', 'ephi_U2_a%s_*.json' % alpha)):
        rows += json.load(open(f))
    rows = sorted({r['phi']: r for r in rows}.values(), key=lambda r: r['phi'])
    if alpha == '0' and rows and min(r['phi'] for r in rows) >= 0:   # alpha = 0: E(-phi) = E(phi), mirror the half range
        mirrored = [dict(r, phi=-r['phi']) for r in rows if r['phi'] > 0]
        rows = sorted(mirrored + rows, key=lambda r: r['phi'])
    return rows


def main():
    res = {}
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
    # (a)
    ax = axs[0]
    for alpha, ls in (('0.6', '-'), ('0', '--')):
        rows = load_ephi(alpha)
        if not rows:
            continue
        phi = np.array([r['phi'] for r in rows])
        for sec, col, lab in (('odd_up', '#1f77b4', r'$S^z=+\frac{1}{2}$'), ('odd_dn', '#d62728', r'$S^z=-\frac{1}{2}$'), ('even', '#2ca02c', 'singlet')):
            if sec not in rows[0]:
                continue
            E = np.array([r[sec] for r in rows]); E0 = np.array([r['odd_up'] for r in rows]).min()
            a, b = fit_series(phi, E)
            if sec != 'even':
                ph = np.linspace(-np.pi, np.pi, 400)
                Efit = sum(a[m] * np.cos(m * ph) + b[m] * np.sin(m * ph) for m in range(len(a)))
                ax.plot(ph / np.pi, 1e3 * (Efit - E0), ls, color=col, lw=1.1, label=lab + (r', $\alpha=0.6$' if alpha == '0.6' else r', $\alpha=0$'))
                if alpha == '0.6':
                    ax.plot(phi / np.pi, 1e3 * (E - E0), 'o', color=col, ms=2.5, mfc='w', mew=0.7)
            res['harm_%s_a%s' % (sec, alpha)] = dict(cos=a.tolist(), sin=b.tolist())
            if sec == 'odd_up':
                phi_min = phi[np.argmin(E)]
                res['phimin_%s_a%s' % (sec, alpha)] = float(phi_min)
    ax.set_xlabel(r'$\varphi/\pi$'); ax.set_ylabel(r'$E_\sigma(\varphi)-E_{\rm min}$  $(10^{-3}\Delta)$'); ax.set_ylim(-0.5, 24); ax.legend(loc='upper center', ncol=2, fontsize=6.2, handlelength=1.6, columnspacing=0.8)
    ax.set_title(r'(a) doublet surfaces, two-orbital dot', loc='left', fontsize=8.5)
    # (b)
    ax = axs[1]
    for f in sorted(glob.glob(os.path.join(D, 'so', 'cat_*.json'))):
        rec = json.load(open(f)); g = rec['g']
        for sec, col, lab in (('odd_up', '#1f77b4', r'$\uparrow$'), ('odd_dn', '#d62728', r'$\downarrow$')):
            rho = np.array(rec[sec]['rho_ph'][0]) + 1j * np.array(rec[sec]['rho_ph'][1])
            Nph = rho.shape[0]; n = np.arange(Nph)
            X = np.diag(np.sqrt(n[1:]), 1) + np.diag(np.sqrt(n[1:]), -1)
            x, V = np.linalg.eigh(X)
            P = np.real(np.einsum('ia,ij,ja->a', V.conj(), rho, V))
            ax.plot(g * x / (np.pi / 2), P / np.gradient(g * x / (np.pi / 2)), color=col, label=lab + r' ($\langle\Phi\rangle=%+.2f\,\Phi_0/2$)' % (g * rec[sec]['X'] / (np.pi / 2)))
            res['cat_%s' % sec] = dict(X=rec[sec]['X'], n=rec[sec]['n'], E=rec[sec]['E'], S_ph=rec[sec]['S_ph'])
        res['cat_params'] = dict(U=rec['U'], alpha=rec['alpha'], omega=rec['omega'], g=g)
        if 'odd_up_exc' in rec:
            res['cat_odd_up_exc'] = rec['odd_up_exc']
    ax.set_xlabel(r'$\Phi/(\Phi_0/2)$'); ax.set_ylabel(r'$P(\Phi)$'); ax.legend(loc='upper left')
    ax.set_title(r'(b) ground states with the mode', loc='left', fontsize=8.5)
    # (c)
    ax = axs[2]
    pts = {}
    for f in glob.glob(os.path.join(D, 'gate', 'gate_U*_d*.json')):
        for r in json.load(open(f)):
            d = round(r['eps_up'][0] + r['U'] / 2, 4) if 'eps_up' in r else float(os.path.basename(f).split('_d')[1][:-5])
            pts.setdefault(r['g'], []).append((d, r['odd'] - r['even']))
    for g, col in ((0.0, '0.3'), (0.6, '#1f77b4')):
        if g not in pts:
            continue
        p = sorted(pts[g]); d = np.array([x[0] for x in p]); dE = np.array([x[1] for x in p])
        ax.plot(d, dE, 'o-', color=col, ms=3, mfc='w', mew=0.8, label=r'$g=%g$' % g)
        # crossing
        idx = np.where(np.diff(np.sign(dE)) != 0)[0]
        if len(idx):
            i = idx[0]; dc = d[i] - dE[i] * (d[i + 1] - d[i]) / (dE[i + 1] - dE[i]); res['gate_crossing_g%g' % g] = float(dc)
            ax.axvline(dc, color=col, lw=0.6, ls=':')
    ax.axhline(0, color='0.6', lw=0.6); ax.set_xlabel(r'$\epsilon_d+U/2$  $(\Delta)$'); ax.set_ylabel(r'$E_{\rm odd}-E_{\rm even}$  $(\Delta)$'); ax.legend()
    ax.set_title(r'(c) gate detuning, $U=1.1\Delta$', loc='left', fontsize=8.5)
    fig.savefig(os.path.join(F, 'fig_so.pdf')); fig.savefig(os.path.join(F, 'fig_so.png'))
    json.dump(res, open(os.path.join(D, 'so', 'summary_so_gate.json'), 'w'), indent=1)
    for k, v in res.items():
        print(k, v if not isinstance(v, dict) or 'cos' not in v else {kk: [round(x, 5) for x in vv] for kk, vv in v.items()})


if __name__ == '__main__':
    main()
