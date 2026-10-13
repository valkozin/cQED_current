"""Full electron+photon DMRG at one (U, g) point of the interacting-dot ring, Eq. (1) of the paper,
with the current evaluated from the microscopic fermionic operator on the junction bond,

    I = -dH/dphi_cl = i t_J sum_s [ e^{i phi_cl} D c^dag_{1 s} c_{0 s} - h.c. ],   D = exp(i g (a + a^dag)),

i.e. the expectation value of a three-site (photon x two fermion sites) operator in the DMRG ground state.
The virial value hbar*omega <X> / (2g), which must coincide with it in an exact eigenstate, is stored as a check.

The bond dimension is raised along a ladder (default 256, 512, 768; each stage warm-starts from the previous
one without re-truncation); observables are stored after each stage
so that the convergence of the current in chi can be read off the output.

With --alpha A the photon starts in the coherent state |A> (A real, <X> = 2A) instead of the vacuum: in the
current-carrying regime the seeded ground state is one well of a double well whose splitting (and seed bias,
~2 I* phi_cl) are below the DMRG energy resolution, and a vacuum start converges to an unpolarized superposition.
The energy of the run is stored, so vacuum and polarized starts can be compared.

usage: python run_dmrg_point.py U g omega seed outfile [--chis 256,512,768] [--Nph 20] [--sweeps 30] [--excited] [--alpha A] [--tJ 0.5]
"""
import argparse, json, time, warnings
import numpy as np
from math import factorial
from tenpy.networks.mps import MPS
from dmrg_ring import RingCavityModel, run_dmrg, observables
from bdg_general import sc_ring

warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('U', type=float); ap.add_argument('g', type=float)
ap.add_argument('omega', type=float); ap.add_argument('seed', type=float); ap.add_argument('outfile')
ap.add_argument('--chis', default='256,512,768'); ap.add_argument('--Nph', type=int, default=20)
ap.add_argument('--sweeps', type=int, default=30); ap.add_argument('--excited', action='store_true')
ap.add_argument('--alpha', type=float, default=0.0); ap.add_argument('--tJ', type=float, default=0.5)
a = ap.parse_args()

p = dict(sc_ring(14, 1.0, 1.0, a.tJ, Udot=a.U, dot=True), omega=a.omega, g=a.g, phi_cl=a.seed, Nph=a.Nph)
m = RingCavityModel(p)
init = ['up'] + ['up', 'down'] * 6 + ['empty']            # odd fermion parity, S^z = 1/2 (doublet / pi sector)
out = dict(U=a.U, g=a.g, omega=a.omega, seed=a.seed, Nph=a.Nph, L=14, t=1.0, Delta=1.0, tJ=a.tJ, alpha=a.alpha, stages=[])
psi = None
if a.alpha:
    k = np.arange(a.Nph)
    coh = np.array([a.alpha ** j / np.sqrt(float(factorial(j))) for j in k]); coh /= np.linalg.norm(coh)
    st = [coh] + [None] * 14
    for r, s_ in enumerate(init):
        st[m.pos[r]] = s_
    psi = MPS.from_product_state(m.lat.mps_sites(), st, bc='finite')
for chi in [int(c) for c in a.chis.split(',')]:
    t0 = time.time()
    E0, psi = run_dmrg(m, init, chi=chi, sweeps=a.sweeps, psi0=psi,
                       chi_list=None if psi is None else {0: chi})   # keep the previous stage / polarized start
    o = observables(m, psi)
    rec = dict(chi=chi, chi_used=int(o['chi_max']), E0=float(E0), n=o['n'], X=o['X'], X2=o['X2'],
               I_bond=float(o['I']), I_vir=float(a.omega * o['X'] / (2 * a.g)) if a.g > 0 else float(o['I']),
               ReD=float(np.real(o['ReD'])), ImD=float(np.imag(o['ReD'])), p_tail=float(o['pn'][-1]),
               pn=o['pn'].tolist(), time=time.time() - t0)
    out['stages'].append(rec)
    print('U=%.3f g=%.3f chi=%d E0=%.10f n=%.4f I_bond=%.5e I_vir=%.5e p_tail=%.1e t=%.0fs' %
          (a.U, a.g, chi, E0, o['n'], rec['I_bond'], rec['I_vir'], rec['p_tail'], rec['time']), flush=True)
    json.dump(out, open(a.outfile, 'w'))
if a.excited:                                              # first excited state in the same sector -> tunnel splitting
    t0 = time.time()
    E1, psi1 = run_dmrg(m, init, chi=out['stages'][-1]['chi'], sweeps=a.sweeps, orthogonal_to=[psi])
    o1 = observables(m, psi1)
    out['excited'] = dict(E1=float(E1), n=o1['n'], X=o1['X'], X2=o1['X2'], I_bond=float(o1['I']), time=time.time() - t0)
    print('  excited: E1-E0=%.4e (= %.3f hbar*omega)  t=%.0fs' % (E1 - out['stages'][-1]['E0'],
          (E1 - out['stages'][-1]['E0']) / a.omega, time.time() - t0), flush=True)
    json.dump(out, open(a.outfile, 'w'))
