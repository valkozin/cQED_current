"""Full electron+photon DMRG (no adiabatic or product-state approximation) on a (U, g) grid.

usage: python run_dmrg_phase_diagram.py <U> <sector: odd|even> <g list> <chi> <Nph> <outfile> [phi_cl]
Ring L=14 (t=1, Delta=1), Anderson dot eps=-U/2, tJ=0.5, hbar*omega = 7.61e-4 as in the paper; no external flux
(phi_cl = 1e-9 keeps H generic but symmetric to machine precision).  Ground state only, warm-started along g.
Order parameter: the flux distribution P(Phi) from the photon reduced density matrix (bimodal <=> condensate),
plus <a^dag a> and <X^2>.  rho_ph is stored so P(Phi) can be reconstructed (make_figures.photon_xdist).
"""
import numpy as np, time, json, sys, os
from dmrg_ring import RingCavityModel, run_dmrg, observables
from bdg_general import sc_ring

U, sector = float(sys.argv[1]), sys.argv[2]
glist = [float(x) for x in sys.argv[3].split(',')]
chi, Nph, outfile = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
phi_cl = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-9
omega = 7.61e-4
init = ['up'] + ['up', 'down'] * 6 + ['empty'] if sector == 'odd' else ['up', 'down'] * 7
out = dict(model='udot', U=U, sector=sector, omega=omega, phi_cl=phi_cl, chi=chi, Nph=Nph, rows=[])
if os.path.exists(outfile):                       # resume
    out = json.load(open(outfile)); done = {r['g'] for r in out['rows']}
    glist = [g for g in glist if g not in done]
psi_prev = None
for g in glist:
    p = dict(sc_ring(14, 1.0, 1.0, 0.5, Udot=U, dot=True), omega=omega, g=g, phi_cl=phi_cl, Nph=Nph)
    m = RingCavityModel(p)
    t0 = time.time()
    E0, psi = run_dmrg(m, init, chi=chi, sweeps=int(os.environ.get('SWEEPS', 20)), psi0=psi_prev)
    o = observables(m, psi)
    psi_prev = psi
    rec = dict(g=g, E0=E0, X=o['X'], X2=o['X2'], n=o['n'], I_bond=o['I'], pn=o['pn'].tolist(), S_ph=o['S_ph'],
               chi_max=o['chi_max'], rho_ph=[o['rho_ph'].real.tolist(), o['rho_ph'].imag.tolist()],
               time=time.time() - t0)
    out['rows'].append(rec)
    print('U=%.2f g=%.3f E0=%.9f n=%.3f sqrt<X2>=%.3f X=%.2e pmax=%.1e chi=%d t=%.0fs' %
          (U, g, E0, o['n'], np.sqrt(o['X2']), o['X'], o['pn'][-1], o['chi_max'], rec['time']), flush=True)
    json.dump(out, open(outfile, 'w'))
