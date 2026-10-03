"""'Exact' (U, g) phase diagram of the microscopic model: photon treated exactly in the adiabatic (BO)
Hamiltonian with E(phi; U) from DMRG of the interacting ring in both fermion-parity sectors.
(Validated against full electron+photon DMRG, see run_dmrg_scan.py / run_dmrg_gscan.py.)"""
import numpy as np, json, glob
from bo import solve

omega, seed = 7.61e-4, 1e-3
gs = np.round(np.arange(0, 0.4001, 0.01), 4)


def load_curves():
    rows = []
    for f in ['../data/udot_phi_tJ0.5.json', '../data/udot_phi9_A.json', '../data/udot_phi9_B.json']:
        try:
            rows += json.load(open(f))
        except FileNotFoundError:
            pass
    curves = {}
    for U in sorted(set(r['U'] for r in rows)):
        rr = sorted([r for r in rows if r['U'] == U], key=lambda r: r['phi'])
        phi = np.array([r['phi'] for r in rr])
        if len(phi) < 9 or phi.max() < 3.1:
            continue
        curves[U] = {s: (phi, np.array([r[s] for r in rr])) for s in ('even', 'odd')}
    return curves


def cos_fit(phi, E, M=6):
    A = np.cos(np.outer(phi, np.arange(M + 1)))
    c, *_ = np.linalg.lstsq(A, E, rcond=None)
    return c.astype(complex)


def fine_grid(curves, Ufine, M=6):
    """cubic interpolation in U of the cosine coefficients of E(phi) in each sector"""
    from scipy.interpolate import PchipInterpolator
    Us = np.array(sorted(curves))
    out = {}
    for s in ('even', 'odd'):
        C = np.array([cos_fit(*curves[U][s], M=M).real for U in Us])
        f = PchipInterpolator(Us, C, axis=0)
        out[s] = f(Ufine)
    return out


def scan(Ulist, coeffs):
    res = dict(I=[], n=[], sector=[], delta=[], Phi=[], I0=[], E2=[])
    for k, U in enumerate(Ulist):
        cfit = {s: coeffs[s][k].astype(complex) for s in coeffs}
        rowI, rown, rows_, rowd, rowP, rowI0 = [], [], [], [], [], []
        for g in gs:
            best = None
            for s, c in cfit.items():
                o = solve(c, g, omega, seed, Nphot=160)
                if best is None or o['E'][0] < best[1]['E'][0]:
                    best = (s, o, c)
            s, o, c = best
            o0 = solve(c, g, omega, 0.0, Nphot=160)
            rowI.append(o['I'][0]); rown.append(o['n'][0]); rows_.append(s)
            rowd.append((o0['E'][1] - o0['E'][0]) / omega); rowP.append(o0['X01'] * g / (np.pi / 2))
            rowI0.append(np.sqrt(o0['I2'][0]))
        for key, v in zip(('I', 'n', 'sector', 'delta', 'Phi', 'I0'), (rowI, rown, rows_, rowd, rowP, rowI0)):
            res[key].append(v)
        res['E2'].append({s: float(cfit[s][2].real) for s in cfit})
    return res


if __name__ == '__main__':
    import sys
    curves = load_curves()
    Unodes = sorted(curves)
    print('DMRG U nodes:', Unodes, flush=True)
    nodes = scan(Unodes, {s: np.array([cos_fit(*curves[U][s]).real for U in Unodes]) for s in ('even', 'odd')})
    Ufine = np.round(np.arange(0, max(Unodes) + 1e-9, 0.1), 3)
    fine = scan(Ufine, fine_grid(curves, Ufine))
    json.dump(dict(omega=omega, seed=seed, gs=gs.tolist(), Us=list(map(float, Unodes)), **nodes,
                   fine=dict(Us=Ufine.tolist(), **fine)), open('../data/exact_phase_diagram.json', 'w'))
    for U, I in zip(Unodes, nodes['I']):
        print('U=%.2f max|I|=%.4f' % (U, max(np.abs(I))))
