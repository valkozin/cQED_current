"""Ring + photon with a TWO-ORBITAL dot and spin-orbit coupling (S^z conserving, time-reversal symmetric).

Dot orbitals a (= ring site 0) and b (extra site), both with level eps and Hubbard U, inter-orbital U_ab n_a n_b.
Both orbitals couple to the two ends of the BCS chain:
    (Nr-1 -> a): -t_a,   (Nr-1 -> b): -t_b                       (no photon, no SOI)
    (a -> 1):    -t_a e^{i phi_cl} D,   (b -> 1): -t_b e^{i phi_cl} e^{i alpha sigma^z} D,   D = e^{ig(a+a^dag)}
The spin-dependent phase alpha sigma^z on one of the two interfering paths is a non-abelian (Aharonov-Casher)
flux that cannot be gauged away; it is invariant under time reversal (spin flip + complex conjugation).
A single path would make the phase removable, which is the statement that SOI alone does not split the
Andreev levels of a single-channel junction [Padurariu-Nazarov]; the second orbital provides the second channel.
MPS order: photon, a, b, 1, Nr-1, 2, Nr-2, ... (all bonds of range <= 3)."""
import numpy as np
from scipy.linalg import expm
from tenpy.networks.site import SpinHalfFermionSite, BosonSite, set_common_charges
from tenpy.models.lattice import TrivialLattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.mps import MPS
from dmrg_ring import photon_site, folded_order


class RingCavitySOModel(CouplingModel, MPOModel):
    """params: Nr, t, Delta, ta, tb, eps_a, eps_b, U, Uab, alpha, omega, g, phi_cl, Nph"""

    def __init__(self, p):
        self.p = p
        Nr = p['Nr']
        order = folded_order(Nr)
        order = [order[0], 'b'] + order[1:]
        self.pos = {r: k + 1 for k, r in enumerate(order)}
        fs = SpinHalfFermionSite(cons_N='parity', cons_Sz='Sz')
        ph = photon_site(p['Nph'], p['g'])
        set_common_charges([ph, fs], new_charges='independent')
        self.sites = [ph] + [fs] * (Nr + 1)
        lat = TrivialLattice(self.sites)
        CouplingModel.__init__(self, lat)
        z = [0]
        self.add_onsite_term(p['omega'], 0, 'N')
        a, b = self.pos[0], self.pos['b']
        for site, eps in ((a, p['eps_a']), (b, p['eps_b'])):
            self.add_onsite_term(eps, site, 'Ntot')
            self.add_onsite_term(p['U'], site, 'NuNd')
        self.add_coupling(p['Uab'], a, 'Ntot', b, 'Ntot', z)
        for r in range(1, Nr):
            self.add_onsite_term(p['Delta'], self.pos[r], 'Cdu Cdd', plus_hc=True)
        for r in range(1, Nr - 1):                     # chain bonds r -> r+1
            for s in ('u', 'd'):
                self.add_coupling(-p['t'], self.pos[r], 'Cd' + s, self.pos[r + 1], 'C' + s, z, plus_hc=True)
        last = self.pos[Nr - 1]
        for s in ('u', 'd'):                           # (Nr-1 -> a), (Nr-1 -> b)
            self.add_coupling(-p['ta'], last, 'Cd' + s, a, 'C' + s, z, plus_hc=True)
            self.add_coupling(-p['tb'], last, 'Cd' + s, b, 'C' + s, z, plus_hc=True)
        one = self.pos[1]
        for s, sgn in (('u', +1), ('d', -1)):          # photon-dressed bonds a -> 1 and b -> 1 (SOI phase on b)
            amp_a = -p['ta'] * np.exp(1j * p['phi_cl'])
            amp_b = -p['tb'] * np.exp(1j * (p['phi_cl'] + sgn * p['alpha']))
            self.add_multi_coupling(amp_a, [('D', z, 0), ('Cd' + s, z, one), ('C' + s, z, a)], plus_hc=True)
            self.add_multi_coupling(amp_b, [('D', z, 0), ('Cd' + s, z, one), ('C' + s, z, b)], plus_hc=True)
        MPOModel.__init__(self, lat, self.calc_H_MPO())


def init_states_so(Nr):
    M = Nr + 1                                        # fermionic sites: a, b, 1..Nr-1
    k = (M - 1) // 2
    even = ['empty'] + ['up', 'down'] * k if M % 2 else ['up', 'down'] * (M // 2)
    odd_up = ['up'] + ['up', 'down'] * k if M % 2 else ['up', 'up', 'down'] + ['up', 'down'] * (M // 2 - 2) + ['empty']
    odd_dn = ['down'] + ['up', 'down'] * k if M % 2 else ['down', 'up', 'down'] + ['up', 'down'] * (M // 2 - 2) + ['empty']
    return {'even': even, 'odd_up': odd_up, 'odd_dn': odd_dn}


def product_state_so(model, ring_states, photon=0):
    Nr = model.p['Nr']
    st = [photon] + [None] * (Nr + 1)
    keys = [0, 'b'] + list(range(1, Nr))
    for key, s in zip(keys, ring_states):
        st[model.pos[key]] = s
    return MPS.from_product_state(model.lat.mps_sites(), st, bc='finite')


def observables_so(model, psi):
    out = {}
    out['n'] = psi.expectation_value('N', [0])[0].real
    out['X'] = psi.expectation_value('X', [0])[0].real
    out['X2'] = psi.expectation_value('X2', [0])[0].real
    rho = psi.get_rho_segment([0]).to_ndarray()
    out['pn'] = np.real(np.diag(rho)); out['rho_ph'] = rho
    w = np.linalg.eigvalsh(rho); w = w[w > 1e-15]; out['S_ph'] = float(-np.sum(w * np.log(w)))
    out['na'] = float(psi.expectation_value('Ntot', [model.pos[0]])[0].real)
    out['nb'] = float(psi.expectation_value('Ntot', [model.pos['b']])[0].real)
    out['Sz_a'] = float(psi.expectation_value('Sz', [model.pos[0]])[0].real)
    out['Sz_b'] = float(psi.expectation_value('Sz', [model.pos['b']])[0].real)
    out['chi_max'] = int(max(psi.chi))
    return out
