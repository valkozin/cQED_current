"""Effective circuit-QED model: fermions integrated out, H_eff = omega a^dag a + E(phi_cl + g X),
with E(phi) = ground-state energy of the ring at fixed phase from DMRG (both parity sectors; the lower one is taken
point by point).  Photon treated exactly (DVR, Nphot=200).  Same (U, g) grid as the full DMRG.
usage: python effective_grid.py <ephi json(s) comma-separated> <omega> <seed> <g list> <outfile> [U list]"""
import numpy as np, json, sys
from bo import solve
from harm import harmonics

files = sys.argv[1].split(','); omega, seed = float(sys.argv[2]), float(sys.argv[3])
gs = [float(g) for g in sys.argv[4].split(',')]; outfile = sys.argv[5]
rows = sum((json.load(open(f)) for f in files), [])
Us = sorted(set(r['U'] for r in rows))
if len(sys.argv) > 6:
    Us = [u for u in Us if u in [float(x) for x in sys.argv[6].split(',')]]
out = dict(omega=omega, seed=seed, gs=gs, Us=Us, I=[], n=[], X=[], X2=[], sector=[], delta=[], Phi=[], Irms=[], harm={}, Ic=[])
M = 4
for U in Us:
    rr = sorted([r for r in rows if r['U'] == U], key=lambda r: r['phi'])
    phi = np.array([r['phi'] for r in rr])
    c = {s: harmonics(phi, np.array([r[s] for r in rr]), M=M) for s in ('even', 'odd')}
    out['harm'][str(U)] = {s: c[s].tolist() for s in c}
    s0 = min(c, key=lambda s: c[s][0] + c[s][1] + c[s][2] + c[s][3] + c[s][4])   # ground sector at phi=0
    ph = np.linspace(0, 2 * np.pi, 2001)
    m = np.arange(M + 1)
    Ic = max(np.max(np.abs(np.sum(-m[None, :] * c[s][None, :] * np.sin(np.outer(ph, m)), 1))) for s in [s0])
    out['Ic'].append(float(Ic))
    rI, rn, rX, rX2, rs, rd, rP, rIr = [], [], [], [], [], [], [], []
    for g in gs:
        best = None
        for s in ('even', 'odd'):
            o = solve(c[s].astype(complex), g, omega, seed, Nphot=200)
            if best is None or o['E'][0] < best[1]['E'][0]:
                best = (s, o)
        s, o = best
        o0 = solve(c[s].astype(complex), g, omega, 0.0, Nphot=200)
        rI.append(float(o['I'][0])); rn.append(float(o['n'][0])); rX.append(float(o['X'][0])); rX2.append(float(o['X2'][0])); rs.append(s)
        rd.append(float((o0['E'][1] - o0['E'][0]) / omega)); rP.append(float(o0['X01'] * g / (np.pi / 2))); rIr.append(float(np.sqrt(o0['I2'][0])))
    for k, v in zip(('I', 'n', 'X', 'X2', 'sector', 'delta', 'Phi', 'Irms'), (rI, rn, rX, rX2, rs, rd, rP, rIr)):
        out[k].append(v)
    print('U=%.2f sector=%s E2=%+.4e Ic=%.4f max|I|/Ic=%.3f' % (U, s0, c[s0][2], Ic, max(np.abs(rI)) / Ic), flush=True)
    json.dump(out, open(outfile, 'w'))
