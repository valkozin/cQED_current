"""DMRG scan (electrons + quantum photon) vs the dimensionless screening parameter beta.

usage: python run_dmrg_scan.py <model> <EJ/omega> <beta list> <chi> <Nph> <outfile>
model = 'weaklink'  : all-SC ring Nr=14, ordinary 0-junction (tJ=0.5) + external flux Phi0/2 (phi_cl=pi/2)
model = 'udot:U'    : SC ring with interacting dot (Hubbard U, eps=-U/2), no external flux (phi_cl=0)
"""
import numpy as np, time, json, sys
from dmrg_ring import RingCavityModel, run_dmrg, observables
from bdg_general import sc_ring, harmonics
from mf_general import solve_mf

model, ratio, betas, chi, Nph, outfile = sys.argv[1], float(sys.argv[2]), \
    [float(b) for b in sys.argv[3].split(',')], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
EJ_in = float(sys.argv[7]) if len(sys.argv) > 7 else None
if model == 'weaklink':
    base = sc_ring(14, 1.0, 1.0, 0.5)
    EJ = abs(harmonics(base, M=8)[2].real)
    phi_cl, init = np.pi / 2, ['up', 'down'] * 7
else:
    U = float(model.split(':')[1])
    base = sc_ring(14, 1.0, 1.0, float(model.split(':')[2]) if model.count(':') > 1 else 0.5, Udot=U, dot=True)
    EJ = EJ_in
    phi_cl, init = 1e-9, ['up'] + ['up', 'down'] * 6 + ['empty']   # odd parity, Sz=+1/2 (doublet sector)
omega = EJ / ratio
out = dict(model=model, EJ=EJ, omega=omega, ratio=ratio, chi=chi, Nph=Nph, rows=[])
psi_prev = None
for b in betas:
    g = np.sqrt(b * omega / (8 * EJ))
    p = dict(base, omega=omega, g=g, phi_cl=phi_cl, Nph=Nph)
    m = RingCavityModel(p)
    t0 = time.time()
    E0, psi0 = run_dmrg(m, init, chi=chi, sweeps=20, psi0=psi_prev)
    o0 = observables(m, psi0)
    E1, psi1 = run_dmrg(m, init, chi=chi, sweeps=20, orthogonal_to=[psi0])
    o1 = observables(m, psi1)
    psi_prev = psi0
    tmp = psi1.copy(); tmp.apply_local_op(0, 'X')
    X01 = psi0.overlap(tmp)
    rec = dict(beta=b, g=g, E0=E0, E1=E1, X01=abs(X01), I0=o0['I'], I1=o1['I'], n0=o0['n'], n1=o1['n'], X0=o0['X'], X20=o0['X2'],
               pn0=o0['pn'].tolist(), S_ph=o0['S_ph'], rho_ph=[o0['rho_ph'].real.tolist(), o0['rho_ph'].imag.tolist()],
               rho_ph1=[o1['rho_ph'].real.tolist(), o1['rho_ph'].imag.tolist()], time=time.time() - t0)
    if model == 'weaklink':
        mf = solve_mf(base, g, omega, phi_cl, Nphot=120)
        rec.update(E_mf=mf['E'], I_mf=mf['I'], n_mf=mf['n'])
    out['rows'].append(rec)
    print('beta=%.2f g=%.4f E0=%.9f E1-E0=%.3e I0=%.4e I1=%.4e |X01|=%.3f n=%.3f X=%.3e S_ph=%.3f pmax=%.1e t=%.0fs' %
          (b, g, E0, E1 - E0, o0['I'], o1['I'], abs(X01), o0['n'], o0['X'], o0['S_ph'], o0['pn'][-1], rec['time']), flush=True)
    json.dump(out, open(outfile, 'w'))
