"""Spin-orbit point: two-orbital dot with a spin-dependent phase alpha on one of its two paths (dmrg_ring_so.py).

usage:
  python run_so.py ephi  <U> <alpha> <phi list> <outfile>             E(phi) at g=0 in sectors even, odd_up, odd_dn
  python run_so.py uscan <U list> <alpha> <outfile>                   E_odd_up - E_even at phi = 0, g = 0
  python run_so.py cat   <U> <alpha> <omega> <g> <Nph> <outfile>      photon runs, no seed: odd_up and odd_dn ground
                                                                       states (+ first excited of odd_up), even ground state
  python run_so.py switch <U list> <alpha> <omega> <g list> <Nph> <outfile>   E_odd_up - E_even with the mode
Fixed: Nr=10, t=Delta=1, ta=tb=0.5, eps_a=-U/2, eps_b=eps_a+DEPS (env, default 0.5), Uab=U, chi=CHI (env, 200)."""
import numpy as np, json, sys, os, time
from tenpy.algorithms import dmrg
from dmrg_ring_so import RingCavitySOModel, init_states_so, product_state_so, observables_so
from scipy.special import gammaln

Nr = 10
chi = int(os.environ.get('CHI', 200)); DEPS = float(os.environ.get('DEPS', 0.5))
init = init_states_so(Nr)


def params(U, alpha, omega=1.0, g=0.0, phi=0.0, Nph=4):
    return dict(Nr=Nr, t=1.0, Delta=1.0, ta=0.5, tb=0.5, eps_a=-U / 2, eps_b=-U / 2 + DEPS, U=U, Uab=U,
                alpha=alpha, omega=omega, g=g, phi_cl=phi, Nph=Nph)


def run(m, psi, sweeps=30, orth=()):
    opts = {'mixer': True, 'max_E_err': 1e-11, 'max_S_err': 1e-3, 'max_trunc_err': 1.0, 'max_sweeps': sweeps,
            'min_sweeps': 6, 'trunc_params': {'chi_max': chi, 'svd_min': 1e-11}, 'chi_list': {0: 64, 2: chi},
            'mixer_params': {'amplitude': 1e-5, 'decay': 1.5, 'disable_after': 6}}
    info = dmrg.run(psi, m, opts, orthogonal_to=list(orth))
    st = info['sweep_statistics']
    return info['E'], psi, float(max(st['max_trunc_err'][-2:]))


def coherent(Nph, X0):
    n = np.arange(Nph); a = X0 / 2
    amp = np.exp(-a * a / 2 + n * np.log(abs(a) + 1e-300) - 0.5 * gammaln(n + 1)) * np.sign(a) ** n
    return amp / np.linalg.norm(amp)


def rebind(psi, m):
    psi = psi.copy(); psi.sites = list(m.lat.mps_sites()); return psi


mode = sys.argv[1]
out = []
if mode == 'ephi':
    U, alpha = float(sys.argv[2]), float(sys.argv[3]); phis = [float(x) for x in sys.argv[4].split(',')]; outfile = sys.argv[5]
    prev = {}
    for phi in phis:
        m = RingCavitySOModel(params(U, alpha, phi=phi)); rec = dict(U=U, alpha=alpha, phi=phi); t0 = time.time()
        for sec in ('even', 'odd_up', 'odd_dn'):
            E, psi, terr = run(m, rebind(prev[sec], m) if sec in prev else product_state_so(m, init[sec]))
            prev[sec] = psi; o = observables_so(m, psi)
            rec[sec] = E; rec['na_' + sec] = o['na']; rec['nb_' + sec] = o['nb']; rec['Sza_' + sec] = o['Sz_a']; rec['terr_' + sec] = terr
        out.append(rec); json.dump(out, open(outfile, 'w'))
        print('U=%.2f alpha=%.2f phi=%+.4f E_even=%.8f E_up=%.8f E_dn=%.8f up-even=%+.5f up-dn=%+.2e na/nb(up)=%.2f/%.2f t=%.0fs' %
              (U, alpha, phi, rec['even'], rec['odd_up'], rec['odd_dn'], rec['odd_up'] - rec['even'], rec['odd_up'] - rec['odd_dn'],
               rec['na_odd_up'], rec['nb_odd_up'], time.time() - t0), flush=True)
elif mode == 'uscan':
    Us = [float(x) for x in sys.argv[2].split(',')]; alpha = float(sys.argv[3]); outfile = sys.argv[4]
    for U in Us:
        m = RingCavitySOModel(params(U, alpha)); rec = dict(U=U, alpha=alpha); t0 = time.time()
        for sec in ('even', 'odd_up'):
            E, psi, terr = run(m, product_state_so(m, init[sec])); o = observables_so(m, psi)
            rec[sec] = E; rec['na_' + sec] = o['na']; rec['nb_' + sec] = o['nb']
        out.append(rec); json.dump(out, open(outfile, 'w'))
        print('U=%.2f alpha=%.2f up-even=%+.5f na/nb(up)=%.2f/%.2f na/nb(even)=%.2f/%.2f t=%.0fs' %
              (U, alpha, rec['odd_up'] - rec['even'], rec['na_odd_up'], rec['nb_odd_up'], rec['na_even'], rec['nb_even'], time.time() - t0), flush=True)
elif mode == 'cat':
    U, alpha, omega, g, Nph = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]); outfile = sys.argv[7]
    m = RingCavitySOModel(params(U, alpha, omega=omega, g=g, Nph=Nph)); rec = dict(U=U, alpha=alpha, omega=omega, g=g, Nph=Nph)
    for sec in ('odd_up', 'odd_dn', 'even'):
        t0 = time.time()
        cands = []
        for X0 in (0.0, 4.0, -4.0):                      # vacuum start and the two polarized starts
            st = [coherent(Nph, X0) if X0 else 0] + [None] * (Nr + 1)
            keys = [0, 'b'] + list(range(1, Nr))
            for key, s in zip(keys, init[sec]):
                st[m.pos[key]] = s
            from tenpy.networks.mps import MPS
            psi = MPS.from_product_state(m.lat.mps_sites(), st, bc='finite')
            E, psi, terr = run(m, psi); cands.append((E, psi, X0))
        E, psi, X0 = min(cands, key=lambda c: c[0]); o = observables_so(m, psi)
        rec[sec] = dict(E=E, X=o['X'], X2=o['X2'], n=o['n'], pn=o['pn'].tolist(), S_ph=o['S_ph'], na=o['na'], nb=o['nb'],
                        Sz_a=o['Sz_a'], Sz_b=o['Sz_b'], E_starts={str(c[2]): c[0] for c in cands}, X_starts={str(c[2]): float(observables_so(m, c[1])['X']) for c in cands},
                        rho_ph=[o['rho_ph'].real.tolist(), o['rho_ph'].imag.tolist()], time=time.time() - t0)
        if sec == 'odd_up':
            E1, psi1, _ = run(m, psi.copy(), orth=[psi]); o1 = observables_so(m, psi1)
            rec['odd_up_exc'] = dict(E=E1, X=o1['X'], n=o1['n'])
        print('%s: E=%.8f X=%+.4f n=%.3f S_ph=%.3f Sz_a=%+.3f starts(E)=%s starts(X)=%s t=%.0fs' %
              (sec, E, o['X'], o['n'], o['S_ph'], o['Sz_a'], {k: round(v - E, 8) for k, v in rec[sec]['E_starts'].items()},
               {k: round(v, 3) for k, v in rec[sec]['X_starts'].items()}, time.time() - t0), flush=True)
        json.dump(rec, open(outfile, 'w'))
elif mode == 'switch':
    Us = [float(x) for x in sys.argv[2].split(',')]; alpha, omega = float(sys.argv[3]), float(sys.argv[4])
    gs = [float(x) for x in sys.argv[5].split(',')]; Nph = int(sys.argv[6]); outfile = sys.argv[7]
    for U in Us:
        prev = {}
        for g in gs:
            m = RingCavitySOModel(params(U, alpha, omega=omega, g=g, Nph=Nph)); rec = dict(U=U, alpha=alpha, omega=omega, g=g); t0 = time.time()
            for sec in ('even', 'odd_up'):
                E, psi, terr = run(m, rebind(prev[sec], m) if sec in prev else product_state_so(m, init[sec])); prev[sec] = psi
                o = observables_so(m, psi); rec[sec] = E; rec['n_' + sec] = o['n']; rec['terr_' + sec] = terr
            out.append(rec); json.dump(out, open(outfile, 'w'))
            print('U=%.2f g=%.2f om=%.3f up-even=%+.5f n=%.3f/%.3f t=%.0fs' % (U, g, omega, rec['odd_up'] - rec['even'], rec['n_even'], rec['n_odd_up'], time.time() - t0), flush=True)
