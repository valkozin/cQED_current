"""Diagnostics of one full-DMRG point: energy variance, entanglement with the photon, sum rule, Hellmann-Feynman.
usage: python diag_point.py <Nr> <U> <omega> <seed> <g> <chi> <Nph> <sweeps> [psi0 from g_prev? no]"""
import numpy as np, sys, time
from dmrg_ring import RingCavityModel, observables, product_state
from bdg_general import sc_ring
from tenpy.algorithms import dmrg
from ring_ephi import init_states
Nr, U, omega, seed, g, chi, Nph, sweeps = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
mixer_off = int(sys.argv[9]) if len(sys.argv) > 9 else 10
p = sc_ring(Nr, 1.0, 1.0, 0.5, Udot=U, dot=True); p.update(omega=omega, g=g, phi_cl=seed, Nph=Nph)
m = RingCavityModel(p)
psi = product_state(m, init_states(Nr)['odd'])
opts = {'mixer': True, 'max_E_err': 1e-13, 'max_trunc_err': 1.0, 'max_sweeps': sweeps, 'min_sweeps': sweeps,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-11}, 'chi_list': {0: 64, 4: chi},
        'mixer_params': {'amplitude': 1e-5, 'decay': 1.5, 'disable_after': mixer_off}}
eng = dmrg.TwoSiteDMRGEngine(psi, m, opts)
t0 = time.time()
for k in range(sweeps // 5):
    eng.options['max_sweeps'] = 5 * (k + 1)
    E, psi = eng.run()
    o = observables(m, psi)
    Ivir = omega * o['X'] / (2 * g)
    var = float(np.real(m.H_MPO.variance(psi)))
    S1 = psi.entanglement_entropy(bonds=[1])[0]
    print('sw=%2d E=%.11f var=%.2e S_ph=%.4f X=%.4f X2=%.3f n=%.3f I_bond=%.4e I_vir=%.4e dev=%.1e chi=%d t=%.0fs' %
          (5 * (k + 1), E, var, S1, o['X'], o['X2'], o['n'], o['I'], Ivir, abs(o['I'] - Ivir), max(psi.chi), time.time() - t0), flush=True)
