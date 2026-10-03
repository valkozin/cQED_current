"""BO predictions for the interacting-dot pi ring (E(phi) from DMRG), beta and EJ/omega scans."""
import numpy as np, json
from bo import solve
c = np.load('../data/harmonics_udot_U2.npy')
m = np.arange(len(c))
EJ = abs(np.sum(-(m ** 2) * c.real)) / 4
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
json.dump(out, open('../data/bo_udot_U2.json', 'w'))
