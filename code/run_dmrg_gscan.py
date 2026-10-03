"""DMRG ground state of the full electron+photon model vs coupling g, with a tiny symmetry-breaking seed flux.

usage: python run_dmrg_gscan.py <model> <omega> <seed phi_cl> <g list> <chi> <Nph> <outfile>
model = 'notebook'  : exact Hamiltonian of the Mathematica notebook (Zeeman-polarized dot, 1+10 sites, t=2, Delta=0.6)
model = 'udot:U'    : SC ring (L=14, t=1, Delta=1) with an interacting dot (eps=-U/2, tJ=0.5); no external flux
Spontaneous symmetry breaking <=> <I> stays finite and seed-independent as seed -> 0 (here: compare seeds).
The current is evaluated from the exact identity <I> = hbar*omega <X> / (2 g).
"""
import numpy as np, time, json, sys
from dmrg_ring import RingCavityModel, run_dmrg, observables
from bdg_general import sc_ring
from bench_params import nb_params

model, omega, seed = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
glist = [float(x) for x in sys.argv[4].split(',')]
chi, Nph, outfile = int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
out = dict(model=model, omega=omega, seed=seed, chi=chi, Nph=Nph, rows=[])
psi_prev = None
for g in glist:
    if model == 'notebook':
        p = nb_params(g, seed, Nph=Nph); p['omega'] = omega
        init = ['up'] + ['up', 'down'] * 5
    else:
        U = float(model.split(':')[1])
        p = dict(sc_ring(14, 1.0, 1.0, 0.5, Udot=U, dot=True), omega=omega, g=g, phi_cl=seed, Nph=Nph)
        init = ['up'] + ['up', 'down'] * 6 + ['empty']
    m = RingCavityModel(p)
    t0 = time.time()
    E0, psi = run_dmrg(m, init, chi=chi, sweeps=int(__import__('os').environ.get('SWEEPS', 24)), psi0=psi_prev)
    o = observables(m, psi)
    psi_prev = psi
    Ivir = omega * o['X'] / (2 * g) if g > 0 else o['I']
    rec = dict(g=g, E0=E0, X=o['X'], X2=o['X2'], n=o['n'], I_vir=Ivir, I_bond=o['I'], pn=o['pn'].tolist(),
               rho_ph=[o['rho_ph'].real.tolist(), o['rho_ph'].imag.tolist()], time=time.time() - t0)
    out['rows'].append(rec)
    print('g=%.4f E0=%.9f X=%.4f n=%.3f I_vir=%.4e I_bond=%.4e flux=gX/pi=%.4f pmax=%.1e t=%.0fs' %
          (g, E0, o['X'], o['n'], Ivir, o['I'], g * o['X'] / np.pi, o['pn'][-1], rec['time']), flush=True)
    json.dump(out, open(outfile, 'w'))
