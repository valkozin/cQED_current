"""Full DMRG of the ring + photon at finite hbar*omega (non-adiabatic regime, hbar*omega comparable to the
singlet-doublet gap): ground state in each fermion-parity sector at phi_cl = 0 (no seed), as a function of the
coupling g.  Output: E_even, E_odd, photon number, <D> = <e^{ig(a+a^dag)}>, photon entanglement entropy and
dot occupation in each sector.  The crossing E_odd = E_even locates the 0-pi boundary U_c(g, omega) of the loop.

usage: python run_nonadiabatic.py <Nr> <omega> <U list> <g list> <chi> <Nph> <outfile> [tJ] [Delta]
"""
import numpy as np, json, sys, os, time
from dmrg_ring import RingCavityModel, observables, product_state
from bdg_general import sc_ring
from ring_ephi import init_states
from tenpy.algorithms import dmrg

Nr, omega = int(sys.argv[1]), float(sys.argv[2])
Us = [float(u) for u in sys.argv[3].split(',')]
gs = [float(g) for g in sys.argv[4].split(',')]
chi, Nph, outfile = int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
tJ = float(sys.argv[8]) if len(sys.argv) > 8 else 0.5
Delta = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
sweeps = int(os.environ.get('SWEEPS', 30))
init = init_states(Nr)


def run(m, psi, orth=()):
    opts = {'mixer': True, 'max_E_err': 1e-11, 'max_S_err': 1e-3, 'max_trunc_err': 1.0, 'max_sweeps': sweeps,
            'min_sweeps': 6, 'trunc_params': {'chi_max': chi, 'svd_min': 1e-11}, 'chi_list': {0: 64, 2: chi},
            'mixer_params': {'amplitude': 1e-5, 'decay': 1.5, 'disable_after': 6}}
    info = dmrg.run(psi, m, opts, orthogonal_to=list(orth))
    st = info['sweep_statistics']
    return info['E'], psi, dict(sweeps=int(st['sweep'][-1]), trunc_err=float(max(st['max_trunc_err'][-2:])))


def rebind(psi, m):
    psi = psi.copy(); psi.sites = list(m.lat.mps_sites()); return psi


out = []
if os.path.exists(outfile):
    out = json.load(open(outfile))
done = {(r['U'], r['g']) for r in out}
for U in Us:
    psi_prev = {}
    for g in gs:
        if (U, g) in done:
            continue
        p = sc_ring(Nr, 1.0, Delta, tJ, Udot=U, dot=True)
        p.update(omega=omega, g=g, phi_cl=0.0, Nph=Nph)
        if os.environ.get('EPS_SHIFT'):      # gate detuning of the dot level from the particle-hole symmetric point
            p['eps_up'][0] = p['eps_dn'][0] = -U / 2 + float(os.environ['EPS_SHIFT'])
        if os.environ.get('TJ_PH'):          # Lang-Firsov reference: reduced hopping on the dressed bond (0,1) only
            p['tJ'] = float(os.environ['TJ_PH'])
        m = RingCavityModel(p)
        rec = dict(Nr=Nr, U=U, g=g, omega=omega, tJ=tJ, Delta=Delta, chi=chi, Nph=Nph)
        t0 = time.time()
        for sec in ('even', 'odd'):
            psi0 = rebind(psi_prev[sec], m) if sec in psi_prev else product_state(m, init[sec])
            E, psi, st = run(m, psi0)
            psi_prev[sec] = psi
            o = observables(m, psi)
            rec[sec] = E
            rec['n_' + sec] = o['n']; rec['X2_' + sec] = o['X2']; rec['X_' + sec] = o['X']
            rec['ReD_' + sec] = float(np.real(o['ReD'])); rec['ImD_' + sec] = float(np.imag(o['ReD']))
            rec['S_ph_' + sec] = o['S_ph']; rec['pn_last_' + sec] = float(o['pn'][-1])
            rec['nd_' + sec] = float(psi.expectation_value('Ntot', [m.pos[0]])[0].real)
            rec['sweeps_' + sec] = st['sweeps']; rec['terr_' + sec] = st['trunc_err']
        rec['time'] = time.time() - t0
        out.append(rec)
        print('om=%.4f U=%.3f g=%.3f E_even=%.9f E_odd=%.9f odd-even=%+.6f n=%.3f/%.3f S_ph=%.4f/%.4f |D|=%.3f/%.3f pn_last=%.1e sw=%d/%d t=%.0fs' %
              (omega, U, g, rec['even'], rec['odd'], rec['odd'] - rec['even'], rec['n_even'], rec['n_odd'], rec['S_ph_even'], rec['S_ph_odd'],
               abs(complex(rec['ReD_even'], rec['ImD_even'])), abs(complex(rec['ReD_odd'], rec['ImD_odd'])), max(rec['pn_last_even'], rec['pn_last_odd']),
               rec['sweeps_even'], rec['sweeps_odd'], rec['time']), flush=True)
        json.dump(out, open(outfile, 'w'))
