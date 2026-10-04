"""Recomputed (U, g) phase diagram of the microscopic model, Eq. (1) of the paper.

Ingredients
* E(phi; U) of the interacting ring in both fermion-parity sectors from DMRG at fixed phase (U nodes in
  data/udot_phi*.json), cosine-fitted (odd/doublet sector: M=6 on [0, pi]; even/singlet sector: the curve has a
  cusp at phi = pi/2, so it is interpolated only on [0, pi/2], where the diamagnetic photon lives).
* Cosine coefficients interpolated in U (PCHIP) onto a fine grid.
* Photon treated exactly (full Fock space, DVR) in the adiabatic Hamiltonian
      H_BO = hbar*omega a^dag a + E(phi_cl + g (a + a^dag); U).
  The sector with the lower ground-state energy is kept.

Observables on the grid (all seed-independent unless stated):
  I_seed3, I_seed5 : <I> for seed fluxes phi_cl = 1e-3, 1e-5 (what a flux offset of ~3e-4 Phi_0 selects)
  n                : <a^dag a> (seedless)
  delta            : (E_1 - E_0)/hbar*omega (seedless): tunnel splitting of the two current states
  Xrms             : sqrt(<X^2>) (seedless)
  Phi              : flux eigenvalue of X projected on the lowest doublet, in units of Phi_0/2
  g_class          : classical threshold beta(U, g) = 1 from E''(0) of the doublet sector
"""
import numpy as np, json, glob, sys
from scipy.interpolate import PchipInterpolator
from bo import solve

omega = 7.61e-4
gs = np.round(np.arange(0, 0.4001, 0.01), 4)
Ufine = np.round(np.arange(0, 10.0001, 0.1), 3)


def load_curves():
    rows = []
    for f in sorted(glob.glob('../data/udot_phi*.json')):
        rows += json.load(open(f))
    curves = {}
    for U in sorted(set(round(r['U'], 6) for r in rows)):
        rr = sorted([r for r in rows if abs(r['U'] - U) < 1e-9], key=lambda r: r['phi'])
        phi = np.array([r['phi'] for r in rr])
        if len(phi) < 9 or phi.max() < 3.1:
            continue
        curves[U] = {s: (phi, np.array([r[s] for r in rr])) for s in ('even', 'odd')}
    return curves


def cos_fit(phi, E, M):
    A = np.cos(np.outer(phi, np.arange(M + 1)))
    c, *_ = np.linalg.lstsq(A, E, rcond=None)
    return c


def coeffs(curves, M=6):
    """odd: M=6 on [0,pi]; even: M=4 on the smooth branch [0, pi/2] (padded with zeros to length M+1)."""
    out = {'odd': {}, 'even': {}}
    for U, cu in curves.items():
        phi, E = cu['odd']; out['odd'][U] = cos_fit(phi, E, M)
        phi, E = cu['even']; sel = phi <= np.pi / 2 + 1e-9
        c = np.zeros(M + 1); c4 = cos_fit(phi[sel], E[sel], min(4, sel.sum() - 1)); c[:len(c4)] = c4
        out['even'][U] = c
    return out


def point(c, g, seeds=(1e-3, 1e-5), Nph=160):
    """BO solution for one sector; returns dict."""
    c = c.astype(complex)
    o0 = solve(c, g, omega, 0.0, Nphot=Nph)
    r = dict(E=o0['E'][0], delta=(o0['E'][1] - o0['E'][0]) / omega, n=o0['n'][0], Xrms=np.sqrt(o0['X2'][0]),
             Phi=o0['X01'] * g / (np.pi / 2), Irms=np.sqrt(o0['I2'][0]))
    for s in seeds:
        o = solve(c, g, omega, s, Nphot=Nph)
        r['I_seed%d' % round(-np.log10(s))] = o['I'][0]
        r['n_seed%d' % round(-np.log10(s))] = o['n'][0]
    return r


def scan(Ulist, C):
    keys = ('E', 'delta', 'n', 'Xrms', 'Phi', 'Irms', 'I_seed3', 'I_seed5', 'n_seed3', 'n_seed5')
    res = {k: [] for k in keys}; res['sector'] = []; res['g_class'] = []; res['EJ'] = []; res['E2'] = []
    for k, U in enumerate(Ulist):
        rows = {kk: [] for kk in keys}; sec = []
        codd, ceven = C['odd'][k], C['even'][k]
        m = np.arange(len(codd)); Epp = -np.sum(m ** 2 * codd)
        res['g_class'].append(float(np.sqrt(omega / (2 * abs(Epp)))) if Epp < 0 else float('nan'))
        res['EJ'].append(abs(Epp) / 4); res['E2'].append(float(codd[2]))
        for g in gs:
            ro, re = point(codd, g), point(ceven, g)
            best, s = (ro, 'odd') if ro['E'] < re['E'] else (re, 'even')
            for kk in keys:
                rows[kk].append(float(best[kk]))
            sec.append(s)
        for kk in keys:
            res[kk].append(rows[kk])
        res['sector'].append(sec)
        print('U=%.2f  g_class=%.3f  max|I_seed3|=%.4f  max n=%.2f  min delta/om=%.1e' %
              (U, res['g_class'][-1], max(np.abs(rows['I_seed3'])), max(rows['n']), min(rows['delta'])), flush=True)
    return res


if __name__ == '__main__':
    curves = load_curves(); Unodes = sorted(curves)
    print('U nodes:', Unodes, flush=True)
    C = coeffs(curves)
    Cn = {s: np.array([C[s][U] for U in Unodes]) for s in C}
    nodes = scan(Unodes, Cn)
    Cf = {s: PchipInterpolator(np.array(Unodes), Cn[s], axis=0)(Ufine) for s in C}
    fine = scan(Ufine, Cf)
    json.dump(dict(omega=omega, gs=gs.tolist(), Us=list(map(float, Unodes)), nodes=nodes,
                   coeff_nodes={s: Cn[s].tolist() for s in Cn},
                   fine=dict(Us=Ufine.tolist(), **fine)), open('../data/phase_diagram_v2.json', 'w'))
