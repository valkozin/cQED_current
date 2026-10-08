"""(U, g) phase diagram of Eq. (1) at a given hbar*omega (default 0.01 t): adiabatic theory with the DMRG E(phi; U)
(exactly as phase_diagram_v2.py) and the HF-BdG x photon mean field (as mf_hf_grid.py), on g in [0, 1.2].
These are the reference lines/maps for the full-DMRG runs of cluster/dmrg_om0.01.sbatch.

usage: python phase_diagram_om.py [omega [tJ]]   -> ../data/phase_diagram_om<omega>[_tJ<tJ>].json
tJ = 0.5 (default) uses data/udot_phi*.json; other tJ use data/tJ<tJ>/udot_phi*.json (U range = DMRG nodes).
"""
import sys, json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import expm
import phase_diagram_v2 as pd
from bdg_general import sc_ring
from hf_udot import hf_energy, mf_point

omega = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
tJ = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
pattern = '../data/udot_phi*.json' if tJ == 0.5 else '../data/tJ%g/udot_phi*.json' % tJ
suffix = '' if tJ == 0.5 else '_tJ%g' % tJ
seed = 1e-3
pd.omega = omega
pd.gs = np.round(np.arange(0, 1.2001, 0.02), 4)
pd.Ufine = None   # set from the DMRG nodes below
_point = pd.point
pd.point = lambda c, g, seeds=(1e-3, 1e-5), Nph=80: _point(c, g, seeds, Nph)

curves = pd.load_curves(pattern); Unodes = sorted(curves)
pd.Ufine = np.round(np.arange(0, 10.0001, 0.1), 3) if tJ == 0.5 else np.round(np.arange(Unodes[0], Unodes[-1] + 1e-9, 0.1), 3)
C = pd.coeffs(curves)
Cn = {s: np.array([C[s][U] for U in Unodes]) for s in C}
Cf = {s: PchipInterpolator(np.array(Unodes), Cn[s], axis=0)(pd.Ufine) for s in C}
fine = pd.scan(pd.Ufine, Cf)

# mean field: HF-BdG electrons x photon product state, upward g sweep with warm start
Nph = 60
n = np.arange(Nph); N = np.diag(n.astype(float)); X = np.diag(np.sqrt(n[1:]), 1) + np.diag(np.sqrt(n[1:]), -1)
Dops = {g: expm(1j * g * X) for g in pd.gs}
Um = [round(x, 3) for x in np.arange(0, 10.001, 0.25)] if tJ == 0.5 else [round(x, 3) for x in np.arange(Unodes[0], Unodes[-1] + 1e-9, 0.25)]
mf = dict(Us=Um, gs=pd.gs.tolist(), I=[], n=[])
for U in Um:
    p = sc_ring(14, 1.0, 1.0, tJ, Udot=U, dot=True)
    E0, nu, nd, F = min([hf_energy(p, U, seed, mag) for mag in (True, False)], key=lambda c: c[0])
    state = (nu, nd, F, 1.0 + 0j); rI, rn = [], []
    for g in pd.gs:
        r, state = mf_point(p, U, g, omega, seed, state, Dops[g], N, X)
        rI.append(float(r['I'])); rn.append(float(r['n']))
    mf['I'].append(rI); mf['n'].append(rn)
    print('MF U=%.2f max|I|=%.2e max n=%.3f' % (U, max(np.abs(rI)), max(rn)), flush=True)

json.dump(dict(omega=omega, tJ=tJ, seed=seed, gs=pd.gs.tolist(), Us=list(map(float, Unodes)),
               fine=dict(Us=pd.Ufine.tolist(), **fine), mf=mf),
          open('../data/phase_diagram_om%g%s.json' % (omega, suffix), 'w'))
