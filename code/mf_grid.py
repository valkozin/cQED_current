"""Mean field of the notebook (HF-BdG electrons x photon product state, upward sweep in g with warm start)
for the interacting-dot ring, on a (U, g) grid.  usage: python mf_grid.py <Nr> <tJ> <omega> <seed> <U list> <g list> <outfile>"""
import numpy as np, json, sys
from scipy.linalg import expm
from bdg_general import sc_ring
from hf_udot import hf_energy, mf_point

Nr, tJ, omega, seed = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
Us = [float(u) for u in sys.argv[5].split(',')]
gs = [float(g) for g in sys.argv[6].split(',')]
outfile = sys.argv[7]
Nph = 120
n = np.arange(Nph); N = np.diag(n.astype(float)); X = np.diag(np.sqrt(n[1:]), 1) + np.diag(np.sqrt(n[1:]), -1)
Dops = {g: expm(1j * g * X) for g in gs}
out = dict(Nr=Nr, tJ=tJ, omega=omega, seed=seed, Us=Us, gs=gs, I=[], E=[], n=[], X=[], m=[], alpha=[])
for U in Us:
    p = sc_ring(Nr, 1.0, 1.0, tJ, Udot=U, dot=True)
    cands = [hf_energy(p, U, seed, mag) for mag in (True, False)]
    E0, nu, nd, F = min(cands, key=lambda c: c[0])
    state = (nu, nd, F, 1.0 + 0j)
    rI, rE, rn, rX, rm, ra = [], [], [], [], [], []
    for g in gs:
        r, state = mf_point(p, U, g, omega, seed, state, Dops[g], N, X)
        rI.append(r['I']); rE.append(r['E']); rn.append(r['n']); rX.append(r['X']); rm.append(r['nu'] - r['nd']); ra.append(abs(r['alpha']))
    for k, v in zip(('I', 'E', 'n', 'X', 'm', 'alpha'), (rI, rE, rn, rX, rm, ra)):
        out[k].append(v)
    print('U=%.2f  max|I|=%.4f  m=%.2f' % (U, max(np.abs(rI)), rm[0]), flush=True)
    json.dump(out, open(outfile, 'w'))
