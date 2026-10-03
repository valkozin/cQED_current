"""Full DMRG (electrons + quantum photon) for the exact notebook model; ground + 1st excited state."""
import numpy as np, time, json, sys
from dmrg_ring import *
from bench_params import nb_params
glist = [float(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else \
    [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
chi, Nph = 320, 30
out = []
psi_prev = None
for g in glist:
    m = RingCavityModel(nb_params(g, 1e-3, Nph=Nph))
    t0 = time.time()
    E0, psi0 = run_dmrg(m, ['up'] + ['up', 'down'] * 5, chi=chi, sweeps=12, psi0=psi_prev)
    o0 = observables(m, psi0)
    E1, psi1 = run_dmrg(m, ['up'] + ['up', 'down'] * 5, chi=chi, sweeps=12, orthogonal_to=[psi0])
    o1 = observables(m, psi1)
    psi_prev = psi0
    tmp = psi1.copy(); tmp.apply_local_op(0, 'X')
    X01 = abs(psi0.overlap(tmp))
    rec = dict(g=g, E0=E0, E1=E1, X01=X01, I0=o0['I'], I1=o1['I'], n0=o0['n'], X0=o0['X'], X20=o0['X2'],
               D0=[o0['ReD'].real, o0['ReD'].imag], pn0=o0['pn'].tolist(), S_ph=o0['S_ph'],
               rho_ph=[o0['rho_ph'].real.tolist(), o0['rho_ph'].imag.tolist()], time=time.time() - t0)
    out.append(rec)
    print('g=%.2f E0=%.8f E1-E0=%.3e I0=%.3e I1=%.3e n=%.3f X=%.3e X2=%.3f S_ph=%.3f t=%.0fs' %
          (g, E0, E1 - E0, o0['I'], o1['I'], o0['n'], o0['X'], o0['X2'], o0['S_ph'], rec['time']), flush=True)
    json.dump(out, open('../data/dmrg_notebook_model.json', 'w'))
