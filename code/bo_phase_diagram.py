"""Tunnel splitting delta/hbar*omega on a (beta, EJ/hbar*omega) grid; adiabatic theory with the DMRG E(phi) at U=2.
usage: python bo_phase_diagram.py [harmonics npy] [outfile]   (default: L=10 harmonics, data/bo_phase_diagram_U2_Nr10.json)
beta = 8 g^2 E_J / hbar omega with E_J = E_2."""
import numpy as np, json, sys
from bo import solve
hfile = sys.argv[1] if len(sys.argv) > 1 else '../data/harmonics_udot_U2_Nr10.npy'
outfile = sys.argv[2] if len(sys.argv) > 2 else '../data/bo_phase_diagram_U2_Nr10.json'
c = np.load(hfile)
m = np.arange(len(c)); EJ = c[2].real
betas = np.linspace(0.05, 4, 40)
ratios = np.logspace(np.log10(0.5), 2, 28)
S = np.zeros((len(ratios), len(betas))); Xs = np.zeros_like(S)
for i, r in enumerate(ratios):
    om = EJ / r
    Nph = int(min(700, max(120, 8 * r + 80)))
    for j, b in enumerate(betas):
        g = np.sqrt(b * om / (8 * EJ))
        o = solve(c, g, om, 0.0, Nphot=Nph)
        S[i, j] = (o['E'][1] - o['E'][0]) / om
        Xs[i, j] = o['X01'] * g / (np.pi / 2)
    print(i, r, flush=True)
json.dump(dict(betas=betas.tolist(), ratios=ratios.tolist(), delta=S.tolist(), Xstar=Xs.tolist()),
          open(outfile, 'w'))
