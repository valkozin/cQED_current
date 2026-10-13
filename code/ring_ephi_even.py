"""Even-sector E(phi) only, on a fine phi grid (the singlet sector is strongly anharmonic near U_c).
usage: python ring_ephi_even.py <Nr> <U list> <tJ> <phi list> <chi> <outfile>"""
import numpy as np, json, sys, time
from dmrg_ring import RingCavityModel, run_dmrg
from bdg_general import sc_ring
from ring_ephi import init_states
Nr = int(sys.argv[1]); Us = [float(u) for u in sys.argv[2].split(',')]; tJ = float(sys.argv[3])
phis = [float(x) for x in sys.argv[4].split(',')]; chi, outfile = int(sys.argv[5]), sys.argv[6]
init = init_states(Nr)['even']; out = []
for U in Us:
    psi_prev = None
    for phi in phis:
        p = sc_ring(Nr, 1.0, 1.0, tJ, Udot=U, dot=True); p.update(omega=1.0, g=0.0, phi_cl=phi, Nph=2)
        m = RingCavityModel(p); t0 = time.time()
        E, psi = run_dmrg(m, init, chi=chi, sweeps=16, psi0=psi_prev); psi_prev = psi
        # also a fresh start to detect level crossings inside the sector (singlet/triplet)
        E2, psi2 = run_dmrg(m, init, chi=chi, sweeps=16)
        if E2 < E - 1e-9:
            E, psi = E2, psi2; psi_prev = psi2
        out.append(dict(Nr=Nr, U=U, phi=phi, tJ=tJ, even=E, E_warm=min(E, E2), E_fresh=E2, chi=chi))
        print('U=%.2f phi=%.4f E_even=%.9f (fresh %.9f) t=%.0fs' % (U, phi, E, E2, time.time() - t0), flush=True)
        json.dump(out, open(outfile, 'w'))
