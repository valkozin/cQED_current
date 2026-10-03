"""DMRG of the interacting-dot ring without photon: E(phi) in even (singlet) and odd (doublet) sectors."""
import numpy as np, json, sys, time
from dmrg_ring import RingCavityModel, run_dmrg
from bdg_general import sc_ring
Us = [float(u) for u in sys.argv[1].split(',')]
phis = [float(x) for x in sys.argv[2].split(',')]
tJ, chi, outfile = float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
init = {'even': ['up', 'down'] * 7, 'odd': ['up'] + ['up', 'down'] * 6 + ['empty']}
out = []
for U in Us:
    for phi in phis:
        p = sc_ring(14, 1.0, 1.0, tJ, Udot=U, dot=True)
        p.update(omega=1.0, g=0.0, phi_cl=phi, Nph=2)
        m = RingCavityModel(p)
        rec = dict(U=U, phi=phi, tJ=tJ)
        for sec in ('even', 'odd'):
            t0 = time.time()
            E, psi = run_dmrg(m, init[sec], chi=chi, sweeps=14)
            rec[sec] = E
            rec['nd_' + sec] = float(psi.expectation_value('Ntot', [m.pos[0]])[0].real)
            rec['nn_' + sec] = float(psi.expectation_value('NuNd', [m.pos[0]])[0].real)
        out.append(rec)
        print('U=%.2f phi=%.4f E_even=%.8f E_odd=%.8f  odd-even=%+.5f  nd=%.3f/%.3f  t=%.0fs' %
              (U, phi, rec['even'], rec['odd'], rec['odd'] - rec['even'], rec['nd_even'], rec['nd_odd'], time.time() - t0), flush=True)
        json.dump(out, open(outfile, 'w'))
