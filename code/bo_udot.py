"""BO predictions for the interacting-dot pi ring (E(phi) from DMRG), beta and EJ/omega scans.
usage: python bo_udot.py [harmonics npy] [outfile]   (default: L=10 harmonics, data/bo_udot_U2_Nr10.json)
beta = 8 g^2 E_J / hbar omega with E_J = E_2 (the definition used throughout the paper)."""
import numpy as np, json, sys
from bo import solve
hfile = sys.argv[1] if len(sys.argv) > 1 else '../data/harmonics_udot_U2_Nr10.npy'
outfile = sys.argv[2] if len(sys.argv) > 2 else '../data/bo_udot_U2_Nr10.json'
c = np.load(hfile)
m = np.arange(len(c))
EJ = c[2].real
betas = np.linspace(0.02, 4, 100)
out = dict(EJ=EJ, betas=betas.tolist(), runs={})
for ratio in [1, 2, 5, 10, 20, 40]:
    om = EJ / ratio
    rows = []
    for b in betas:
        g = np.sqrt(b * om / (8 * EJ))
        o = solve(c, g, om, 0.0, Nphot=500 if ratio >= 20 else 250)
        rows.append([g, o['E'][0], o['E'][1] - o['E'][0], o['E'][2] - o['E'][0], o['X01'], o['I2'][0], o['n'][0]])
    out['runs'][str(ratio)] = rows
    print(ratio, 'done', flush=True)
json.dump(out, open(outfile, 'w'))
