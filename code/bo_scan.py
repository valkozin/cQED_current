"""BO + classical (Landau) scans for the weak-link ring with external half flux quantum."""
import numpy as np, json, sys
from bdg_general import sc_ring, harmonics
from bo import solve, V_of, dV_of


def classical(c, g, omega, phi_cl, Xgrid=np.linspace(-60, 60, 24001)):
    U = omega * Xgrid ** 2 / 4 + V_of(c, phi_cl + g * Xgrid)
    k = np.argmin(U)
    return Xgrid[k], -dV_of(c, np.array([phi_cl + g * Xgrid[k]]))[0], U[k]


if __name__ == '__main__':
    p = sc_ring(14, 1.0, 1.0, 0.5)
    c = harmonics(p, M=16)
    EJ = abs(c[2].real)
    csym = c.copy(); csym[1::2] = 0
    betas = np.linspace(0, 4, 81)
    out = dict(EJ=EJ, c=[c.real.tolist(), c.imag.tolist()], betas=betas.tolist(), runs={})
    for ratio in [2, 5, 10, 20, 40]:
        om = EJ / ratio
        for lab, cc in (('tilt', c), ('sym', csym)):
            rows = []
            for b in betas:
                g = np.sqrt(b * om / (8 * EJ))
                o = solve(cc, g, om, np.pi / 2, Nphot=400)
                Xc, Icl, Ucl = classical(cc, g, om, np.pi / 2 + 1e-9)
                rows.append([g, o['E'][0], o['E'][1] - o['E'][0], o['I'][0], o['I2'][0], o['n'][0], Icl])
            out['runs']['%s_%d' % (lab, ratio)] = rows
            print(lab, ratio, 'done', flush=True)
    json.dump(out, open('../data/bo_weaklink.json', 'w'))
