"""Full DMRG (electrons + photon) versus the screening parameter beta, with the two-start protocol of run_full_grid.py
(ground state from a warm start A and from a polarized coherent-state start C; the lower energy is kept; the sum rule
<I> = omega <X> / 2g is recorded as a convergence measure) and the first excited state (DMRG orthogonal to the ground state).

usage: python run_beta_scan.py <model> <Nr> <EJ/omega> <beta list> <chi> <Nph> <outfile> [omega]
  model = 'weaklink'  : all-SC ring, ordinary 0-junction (tJ=0.5) + external flux Phi0/2 (phi_cl = pi/2); E(phi) from BdG
  model = 'udot:U'    : SC ring with interacting dot (eps=-U/2), no external flux and no seed (phi_cl = 0, cat state);
                        E(phi) from data/effective_Nr<Nr>.json (DMRG at g=0, doublet sector)
  omega: optional override of EJ/ratio (e.g. the value used on the (U,g) grid)
Environment: STARTS (default 'AC'; 'S' adds a vacuum product start, used automatically when no warm start exists).
"""
import numpy as np, time, json, sys, os
from scipy.special import gammaln
from dmrg_ring import RingCavityModel, observables
from bdg_general import sc_ring, harmonics as bdg_harmonics
from bo import V_of
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from ring_ephi import init_states

model, Nr, ratio = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
betas = [float(b) for b in sys.argv[4].split(',')]
chi, Nph0, outfile = int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
om_in = float(sys.argv[8]) if len(sys.argv) > 8 else None
starts = os.environ.get('STARTS', 'AC')
max_sweeps = int(os.environ.get('SWEEPS', 40))
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

if model == 'weaklink':
    base = sc_ring(Nr, 1.0, 1.0, 0.5)
    c = bdg_harmonics(base, M=16)
    EJ = abs(c[2].real)
    phi_cl = np.pi / 2
    init = (['up', 'down'] * Nr)[:Nr]
else:
    U = float(model.split(':')[1])
    base = sc_ring(Nr, 1.0, 1.0, 0.5, Udot=U, dot=True)
    eff = json.load(open(os.path.join(D, 'effective_Nr%d.json' % Nr)))
    c = np.array(eff['harm'][str(U)]['odd'], complex)
    EJ = c[2].real
    phi_cl = 0.0
    init = init_states(Nr)['odd']
omega = om_in if om_in else EJ / ratio
if model == 'weaklink':
    from mf_general import solve_mf


def X_classical(g):
    """minimum of the Landau energy omega X^2/4 + E(phi_cl + g X); for the tilted (weak-link) case the sign is
    selected by the odd harmonics, for the symmetric case the positive minimum is returned"""
    Xg = np.linspace(-np.pi / g, np.pi / g, 40001) if model == 'weaklink' else np.linspace(0, np.pi / (2 * g), 20001)
    F = omega * Xg ** 2 / 4 + V_of(c, phi_cl + g * Xg)
    k = np.argmin(F)
    return float(Xg[k]) if F[k] < F[np.argmin(np.abs(Xg))] - 1e-14 else 0.0


def coherent(Nph, X0):
    n = np.arange(Nph); a = X0 / 2
    amp = np.exp(-a * a / 2 + n * np.log(abs(a) + 1e-300) - 0.5 * gammaln(n + 1)) * np.sign(a) ** n
    return amp / np.linalg.norm(amp)


def rebind(psi, m):
    psi = psi.copy(); psi.sites = list(m.lat.mps_sites()); return psi


def prod(m, Nph, X_init):
    ph0 = coherent(Nph, X_init) if abs(X_init) > 0.3 else 0
    st = [ph0] + [None] * Nr
    for r, s in enumerate(init):
        st[m.pos[r]] = s
    return MPS.from_product_state(m.lat.mps_sites(), st, bc='finite')


def run(m, psi, sweeps, min_sweeps=10, orth=()):
    opts = {'mixer': True, 'max_E_err': 1e-11, 'max_S_err': 1e-2, 'max_trunc_err': 1.0, 'max_sweeps': sweeps, 'min_sweeps': min_sweeps,
            'trunc_params': {'chi_max': chi, 'svd_min': 1e-11}, 'chi_list': {0: 64, 2: chi},
            'mixer_params': {'amplitude': 1e-5, 'decay': 1.5, 'disable_after': 6}}
    info = dmrg.run(psi, m, opts, orthogonal_to=list(orth))
    st = info['sweep_statistics']
    return info['E'], psi, dict(sweeps=int(st['sweep'][-1]), trunc_err=float(max(st['max_trunc_err'][-2:])),
                                dE_last=float(abs(st['E'][-1] - st['E'][-2])) if len(st['E']) > 1 else float('nan'))


out = dict(model=model, Nr=Nr, EJ=EJ, omega=omega, ratio=EJ / omega, chi=chi, Nph=Nph0, phi_cl=phi_cl, starts=starts,
           harm=[c.real.tolist(), c.imag.tolist()], rows=[])
if os.path.exists(outfile):
    out = json.load(open(outfile))
    done = {round(r['beta'], 6) for r in out['rows']}
    betas = [b for b in betas if round(b, 6) not in done]
    print('resuming, %d points left' % len(betas), flush=True)
psi_prev = None
for b in betas:
    g = np.sqrt(b * omega / (8 * EJ))
    X0 = X_classical(g)
    Nph = int(np.clip(1.6 * (X0 / 2) ** 2 + 36, Nph0, 90))
    p = dict(base, omega=omega, g=g, phi_cl=phi_cl, Nph=Nph)
    m = RingCavityModel(p)
    t0 = time.time()
    cands = {}
    if psi_prev is not None and 'A' in starts and psi_prev.sites[0].dim == Nph:
        cands['A'] = run(m, rebind(psi_prev, m), max_sweeps)
    if 'S' in starts or not cands:
        cands['S'] = run(m, prod(m, Nph, 0.0), max_sweeps)
    if 'C' in starts and abs(X0) > 0.3:
        cands['C'] = run(m, prod(m, Nph, X0), max_sweeps)
    best = min(cands, key=lambda k: cands[k][0])
    E0, psi0, stat0 = cands[best]
    o0 = observables(m, psi0)
    # first excited state, orthogonal to the ground state; started in the other well (tilted case) or at the
    # polarized classical flux (symmetric case: the polarized state overlaps with both cat states)
    Xe = -X0 if model == 'weaklink' else X0
    E1, psi1, stat1 = run(m, prod(m, Nph, Xe), max_sweeps, orth=[psi0])
    o1 = observables(m, psi1)
    psi_prev = psi0
    tmp = psi1.copy(); tmp.apply_local_op(0, 'X')
    X01 = abs(psi0.overlap(tmp))
    rec = dict(beta=b, g=g, E0=E0, E1=E1, X01=X01, I0=o0['I'], I1=o1['I'], I_vir0=omega * o0['X'] / (2 * g), I_vir1=omega * o1['X'] / (2 * g),
               n0=o0['n'], n1=o1['n'], X0=o0['X'], X20=o0['X2'], X1=o1['X'], X_cl=X0, Nph=Nph, start=best,
               E_starts={k: cands[k][0] for k in cands}, X_starts={k: float(observables(m, cands[k][1])['X']) for k in cands},
               pn0=o0['pn'].tolist(), S_ph=o0['S_ph'], rho_ph=[o0['rho_ph'].real.tolist(), o0['rho_ph'].imag.tolist()],
               rho_ph1=[o1['rho_ph'].real.tolist(), o1['rho_ph'].imag.tolist()], chi_max=int(o0['chi_max']),
               stat0=stat0, stat1=stat1, time=time.time() - t0)
    if model == 'weaklink':
        mf = solve_mf(base, g, omega, phi_cl, Nphot=120)
        rec.update(E_mf=mf['E'], I_mf=mf['I'], n_mf=mf['n'])
    out['rows'].append(rec)
    print('beta=%.3f g=%.4f E0=%.10f E1-E0=%.3e I0=%.4e I_vir0=%.4e dev=%.1e I1=%.4e |X01|=%.3f n0=%.3f X0=%.3e Xcl=%.2f start=%s dE_starts=%.1e X_starts=%s sw=%d/%d terr=%.1e pmax=%.1e Nph=%d t=%.0fs' %
          (b, g, E0, E1 - E0, o0['I'], rec['I_vir0'], abs(o0['I'] - rec['I_vir0']), o1['I'], X01, o0['n'], o0['X'], X0, best,
           max(v[0] for v in cands.values()) - min(v[0] for v in cands.values()),
           ','.join('%s:%.2f' % (k, v) for k, v in rec['X_starts'].items()), stat0['sweeps'], stat1['sweeps'], stat0['trunc_err'], o0['pn'][-1], Nph, rec['time']), flush=True)
    json.dump(out, open(outfile, 'w'))
