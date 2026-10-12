"""Full DMRG (electrons + photon, no approximation) of the interacting-dot SC ring coupled to the LC mode,
scanned in g at fixed U in a given fermion-parity sector.

usage: python run_full_grid.py <Nr> <U> <tJ> <omega> <seed> <sector even|odd> <g list> <chi> <Nph> <outfile> [max sweeps]

Two independent initial states are used at every g and the lower-energy result is kept:
  A: warm start from the converged state at the previous g (the MPS sites are re-bound to the new model,
     so that the photon displacement operator D = exp(i g X) used in all observables carries the *current* g);
  B: product state with the photon in a coherent state |alpha>, 2 alpha = X_cl, where X_cl is the minimum of the
     classical Landau energy omega X^2/4 + E_2 cos(2 g X) (E_2(U) taken from an earlier run; only a starting guess);
  C: seed annealing: start B converged with a larger seed (1e-2, then 3e-3), then relaxed at the final seed.
     A approaches the polarized state from the vacuum side, C from the polarized side; they bracket the ground state.
Recorded for every g: E0, <X>, <X^2>, <n>, the gauge-invariant junction current
  I = -dH/dphi_cl = i tJ sum_s (e^{i phi_cl} D c^dag_{1s} c_{0s} - h.c.)                       ('I_bond'),
the virial value omega <X>/(2 g) ('I_vir'; exact sum rule <[H,P]>=0 for any eigenstate), the Fock distribution,
the energies/<X> of both starts, DMRG statistics and the wall time.
"""
import numpy as np, json, sys, time, os
from scipy.optimize import minimize_scalar
from scipy.interpolate import PchipInterpolator
from dmrg_ring import RingCavityModel, observables, product_state
from bdg_general import sc_ring
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from ring_ephi import init_states

Nr, U, tJ, omega, seed = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
sector = sys.argv[6]
glist = [float(x) for x in sys.argv[7].split(',')]
chi, Nph0, outfile = int(sys.argv[8]), int(sys.argv[9]), sys.argv[10]
max_sweeps = int(sys.argv[11]) if len(sys.argv) > 11 else 40
Delta = float(os.environ.get('DELTA', 1.0))
starts = os.environ.get('STARTS', 'AC')
anneal = [float(x) for x in os.environ.get('ANNEAL', '1e-2,3e-3').split(',')]   # seeds used before the final one in start C

hg = json.load(open(os.path.join(os.path.dirname(__file__), '../data/harm_guess_Nr14.json')))
Ug = np.array(sorted(float(u) for u in hg)); E2g = PchipInterpolator(Ug, [hg[str(u) if str(u) in hg else repr(u)]['odd'][2] for u in Ug])


def X_classical(g):
    if sector == 'even' or g == 0:
        return 0.0
    E2 = float(E2g(min(U, Ug[-1])))
    F = lambda X: omega * X ** 2 / 4 + E2 * np.cos(2 * g * X)
    r = minimize_scalar(F, bounds=(0, np.pi / (2 * g)), method='bounded')
    return float(r.x) if F(r.x) < F(0.0) - 1e-14 else 0.0


def coherent(Nph, X0):
    n = np.arange(Nph); a = X0 / 2
    from scipy.special import gammaln
    c = np.exp(-a * a / 2 + n * np.log(a + 1e-300) - 0.5 * gammaln(n + 1))
    return c / np.linalg.norm(c)


def rebind(psi, m):
    psi = psi.copy(); psi.sites = list(m.lat.mps_sites()); return psi


def run(m, psi, sweeps, min_sweeps=10):
    opts = {'mixer': True, 'max_E_err': 1e-12, 'max_S_err': 1e-2, 'max_trunc_err': 1.0, 'max_sweeps': sweeps, 'min_sweeps': min_sweeps,
            'trunc_params': {'chi_max': chi, 'svd_min': 1e-11}, 'chi_list': {0: 64, 2: chi},
            'mixer_params': {'amplitude': 1e-5, 'decay': 1.5, 'disable_after': 6}}
    info = dmrg.run(psi, m, opts)
    st = info['sweep_statistics']
    return info['E'], psi, dict(sweeps=int(st['sweep'][-1]), trunc_err=float(max(st['max_trunc_err'][-2:])),
                                dE_last=float(abs(st['E'][-1] - st['E'][-2])) if len(st['E']) > 1 else float('nan'))


out = dict(Nr=Nr, U=U, tJ=tJ, Delta=Delta, omega=omega, seed=seed, sector=sector, chi=chi, Nph=Nph0, starts=starts, rows=[])
if os.path.exists(outfile):  # resume
    out = json.load(open(outfile))
    done = {round(r['g'], 6) for r in out['rows']}
    glist = [g for g in glist if round(g, 6) not in done]
    print('resuming, %d points left' % len(glist), flush=True)
init = init_states(Nr)[sector]
psi_prev = None
for g in glist:
    X0 = X_classical(g)
    Nph = int(np.clip(1.6 * (X0 / 2) ** 2 + 36, Nph0, 90))
    p = sc_ring(Nr, 1.0, Delta, tJ, Udot=U, dot=True)
    p.update(omega=omega, g=g, phi_cl=seed, Nph=Nph)
    m = RingCavityModel(p)
    t0 = time.time()
    cands = {}
    if psi_prev is not None and 'A' in starts and psi_prev.sites[0].dim == Nph:
        cands['A'] = run(m, rebind(psi_prev, m), max_sweeps)
    def prod(X_init):
        ph0 = coherent(Nph, X_init) if X_init > 0.3 else 0
        st = [ph0] + [None] * Nr
        for r, s in enumerate(init):
            st[m.pos[r]] = s
        return MPS.from_product_state(m.lat.mps_sites(), st, bc='finite')
    if 'B' in starts or not cands:
        cands['B'] = run(m, prod(X0), max_sweeps)
    if 'C' in starts and X0 > 0.3:
        psiC = prod(X0)
        for sd in anneal:
            pa = dict(p, phi_cl=sd)
            Ea, psiC, _ = run(RingCavityModel(pa), psiC, 16, min_sweeps=6)
        cands['C'] = run(m, psiC, max_sweeps, min_sweeps=6)
    best = min(cands, key=lambda k: cands[k][0])
    E0, psi, stat = cands[best]
    o = observables(m, psi)
    psi_prev = psi
    Ivir = omega * o['X'] / (2 * g) if g > 0 else float('nan')
    B = sum(psi.expectation_value_term([('D', 0), ('Cd' + s, m.pos[1]), ('C' + s, m.pos[0])]) for s in 'ud') * np.exp(1j * seed)
    rec = dict(g=g, E0=E0, X=o['X'], X2=o['X2'], n=o['n'], I_bond=o['I'], I_vir=Ivir, B=[B.real, B.imag],
               pn=o['pn'].tolist(), S_ph=float(psi.entanglement_entropy(bonds=[1])[0]), chi_max=int(o['chi_max']),
               Nph=Nph, X_cl=X0, start=best, E_starts={k: cands[k][0] for k in cands},
               X_starts={k: float(observables(m, cands[k][1])['X']) for k in cands}, **stat,
               nd=float(psi.expectation_value('Ntot', [m.pos[0]])[0].real), time=time.time() - t0)
    out['rows'].append(rec)
    print('U=%.3f g=%.4f E0=%.10f X=%.4f X2=%.3f n=%.3f I_bond=%.4e I_vir=%.4e dev=%.1e | Xcl=%.2f start=%s dE_starts=%.1e X_starts=%s | chi=%d terr=%.1e sw=%d dE=%.1e pmax=%.1e Nph=%d t=%.0fs' %
          (U, g, E0, o['X'], o['X2'], o['n'], o['I'], Ivir, abs(o['I'] - Ivir), X0, best,
           (max(c[0] for c in cands.values()) - min(c[0] for c in cands.values())),
           ','.join('%.2f' % v for v in rec['X_starts'].values()), o['chi_max'], stat['trunc_err'], stat['sweeps'], stat['dE_last'], o['pn'][-1], Nph, rec['time']), flush=True)
    json.dump(out, open(outfile, 'w'))
