"""E(phi) of the interacting-dot ring (no photon) in the even and odd parity sectors, for any ring size.
usage: python ring_ephi.py <Nr> <U list> <tJ> <phi list> <chi> <outfile> [Delta]"""
import numpy as np, json, sys, time
from dmrg_ring import RingCavityModel, run_dmrg
from bdg_general import sc_ring

def init_states(Nr):
    k = (Nr - 1) // 2
    even = (['up', 'down'] * ((Nr + 1) // 2))[:Nr]
    if Nr % 2:
        even = ['empty'] + ['up', 'down'] * k            # N = Nr-1 even, Sz=0
    odd = ['up'] + ['up', 'down'] * k + (['empty'] if Nr % 2 == 0 else [])
    return {'even': even, 'odd': odd}



if __name__ == '__main__':
    Nr = int(sys.argv[1]); Us = [float(u) for u in sys.argv[2].split(',')]
    tJ = float(sys.argv[3]); phis = [float(x) for x in sys.argv[4].split(',')]
    chi, outfile = int(sys.argv[5]), sys.argv[6]
    Delta = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0


    init = init_states(Nr)
    out = []
    for U in Us:
        psi_prev = {}
        for phi in phis:
            p = sc_ring(Nr, 1.0, Delta, tJ, Udot=U, dot=True)
            p.update(omega=1.0, g=0.0, phi_cl=phi, Nph=2)
            m = RingCavityModel(p)
            rec = dict(Nr=Nr, U=U, phi=phi, tJ=tJ, Delta=Delta, chi=chi)
            t0 = time.time()
            for sec in ('even', 'odd'):
                E, psi = run_dmrg(m, init[sec], chi=chi, sweeps=16, psi0=psi_prev.get(sec))
                psi_prev[sec] = psi
                rec[sec] = E
                rec['nd_' + sec] = float(psi.expectation_value('Ntot', [m.pos[0]])[0].real)
                rec['chi_' + sec] = int(max(psi.chi))
            out.append(rec)
            print('Nr=%d U=%.2f phi=%.4f E_even=%.9f E_odd=%.9f odd-even=%+.6f nd=%.3f/%.3f t=%.0fs' %
                  (Nr, U, phi, rec['even'], rec['odd'], rec['odd'] - rec['even'], rec['nd_even'], rec['nd_odd'], time.time() - t0), flush=True)
            json.dump(out, open(outfile, 'w'))
