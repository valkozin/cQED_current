"""User's mean-field method for the microscopic U model: HF-BdG electrons x photon product state,
upward g sweep with warm start (as in the notebook), on a (U, g) grid."""
import numpy as np, json
from scipy.linalg import expm
from bdg_general import sc_ring
from hf_udot import hf_energy, mf_point

omega, seed = 7.61e-4, 1e-3
import sys
Us = [round(x, 3) for x in np.arange(0, 10.001, 0.25)] if 'fine' in sys.argv else [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
OUT = '../data/mf_hf_grid_fine.json' if 'fine' in sys.argv else '../data/mf_hf_grid.json'
gs = np.round(np.arange(0, 0.4001, 0.01), 4)
Nph = 100
n = np.arange(Nph); N = np.diag(n.astype(float)); X = np.diag(np.sqrt(n[1:]), 1) + np.diag(np.sqrt(n[1:]), -1)
Dops = {g: expm(1j * g * X) for g in gs}
out = dict(omega=omega, seed=seed, Us=Us, gs=gs.tolist(), I=[], E=[], n=[], m=[])
for U in Us:
    p = sc_ring(14, 1.0, 1.0, 0.5, Udot=U, dot=True)
    cands = [hf_energy(p, U, seed, mag) for mag in (True, False)]
    E0, nu, nd, F = min(cands, key=lambda c: c[0])
    state = (nu, nd, F, 1.0 + 0j)
    rI, rE, rn, rm = [], [], [], []
    for g in gs:
        r, state = mf_point(p, U, g, omega, seed, state, Dops[g], N, X)
        rI.append(r['I']); rE.append(r['E']); rn.append(r['n']); rm.append(r['nu'] - r['nd'])
    out['I'].append(rI); out['E'].append(rE); out['n'].append(rn); out['m'].append(rm)
    print('U=%.2f  max|I|=%.4f  g at |I|>1e-3: %s' % (U, max(np.abs(rI)), gs[np.argmax(np.abs(rI) > 1e-3)] if max(np.abs(rI)) > 1e-3 else '-'), flush=True)
    json.dump(out, open(OUT, 'w'))
