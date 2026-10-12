"""Full DMRG (electrons + photon, no approximation) of the interacting-dot SC ring coupled to the LC mode,
scanned in g at fixed U with warm start, in a given fermion-parity sector.

usage: python run_full_grid.py <Nr> <U> <tJ> <omega> <seed> <sector even|odd> <g list> <chi> <Nph> <outfile> [sweeps]

Records for every g: E0, <X>, <X^2>, <n>, the gauge-invariant junction current
  I = -dH/dphi_cl = i tJ sum_s (e^{i phi_cl} D c^dag_{1s} c_{0s} - h.c.)   (D = e^{i g (a+a^dag)})
('I_bond'), the virial value omega <X> / (2 g) ('I_vir'; exact sum rule <[H,P]>=0), the Fock distribution,
DMRG statistics (max chi, truncation error, number of sweeps, last energy change) and the wall time.
"""
import numpy as np, json, sys, time, os
from dmrg_ring import RingCavityModel, run_dmrg, observables, product_state
from bdg_general import sc_ring
from tenpy.algorithms import dmrg
from ring_ephi import init_states

Nr, U, tJ, omega, seed = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
sector = sys.argv[6]
glist = [float(x) for x in sys.argv[7].split(',')]
chi, Nph, outfile = int(sys.argv[8]), int(sys.argv[9]), sys.argv[10]
sweeps = int(sys.argv[11]) if len(sys.argv) > 11 else 30
Delta = float(os.environ.get('DELTA', 1.0))

out = dict(Nr=Nr, U=U, tJ=tJ, Delta=Delta, omega=omega, seed=seed, sector=sector, chi=chi, Nph=Nph, rows=[])
if os.path.exists(outfile):  # resume
    out = json.load(open(outfile))
    done = {round(r['g'], 6) for r in out['rows']}
    glist = [g for g in glist if round(g, 6) not in done]
    print('resuming, %d points left' % len(glist), flush=True)
init = init_states(Nr)[sector]
psi_prev = None
for g in glist:
    p = sc_ring(Nr, 1.0, Delta, tJ, Udot=U, dot=True)
    p.update(omega=omega, g=g, phi_cl=seed, Nph=Nph)
    m = RingCavityModel(p)
    t0 = time.time()
    psi = psi_prev.copy() if psi_prev is not None else product_state(m, init)
    opts = {'mixer': True, 'max_E_err': 1e-11, 'max_trunc_err': 1.0, 'max_sweeps': sweeps, 'min_sweeps': 8,
            'trunc_params': {'chi_max': chi, 'svd_min': 1e-11}, 'chi_list': {0: 64, 4: chi},
            'mixer_params': {'amplitude': 1e-5, 'decay': 1.5, 'disable_after': 10}}
    info = dmrg.run(psi, m, opts)
    E0 = info['E']
    o = observables(m, psi)
    psi_prev = psi
    Ivir = omega * o['X'] / (2 * g) if g > 0 else float('nan')
    # bond expectation of the dressed hopping (gauge-invariant):  B = <e^{i phi_cl} D c^dag_1 c_0> summed over spin
    B = sum(psi.expectation_value_term([('D', 0), ('Cd' + s, m.pos[1]), ('C' + s, m.pos[0])]) for s in 'ud') * np.exp(1j * seed)
    st = info['sweep_statistics']
    rec = dict(g=g, E0=E0, X=o['X'], X2=o['X2'], n=o['n'], I_bond=o['I'], I_vir=Ivir, B=[B.real, B.imag],
               pn=o['pn'].tolist(), S_ph=o['S_ph'], chi_max=int(o['chi_max']),
               trunc_err=float(max(st['max_trunc_err'][-2:])), sweeps=int(st['sweep'][-1]),
               dE_last=float(abs(st['E'][-1] - st['E'][-2])) if len(st['E']) > 1 else float('nan'),
               nd=float(psi.expectation_value('Ntot', [m.pos[0]])[0].real), time=time.time() - t0)
    out['rows'].append(rec)
    print('U=%.3f g=%.4f E0=%.10f X=%.4f X2=%.3f n=%.3f I_bond=%.4e I_vir=%.4e dev=%.1e chi=%d terr=%.1e sw=%d dE=%.1e pmax=%.1e t=%.0fs' %
          (U, g, E0, o['X'], o['X2'], o['n'], o['I'], Ivir, abs(o['I'] - Ivir), o['chi_max'], rec['trunc_err'], rec['sweeps'],
           rec['dE_last'], o['pn'][-1], rec['time']), flush=True)
    json.dump(out, open(outfile, 'w'))
