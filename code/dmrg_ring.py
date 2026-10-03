"""DMRG for a superconducting ring with a junction, threaded by the flux of a quantum LC mode.

H = omega a^dag a
    + sum_{r,s} eps_{r s} n_{r s} + sum_r U_r n_{r up} n_{r dn} + sum_r Delta_r (c^dag_{r up} c^dag_{r dn} + h.c.)
    - sum_{bonds (r,r+1), s} t_r (c^dag_{r+1 s} c_{r s} + h.c.)           (ring, r = 0..Nr-1, r=0 is the junction site)
    - t_J sum_s [ e^{i phi_cl} D c^dag_{1 s} c_{0 s} + h.c. ],   D = exp(i g (a + a^dag))

The electron Peierls phase phi = phi_cl + g X sits on the junction bond (0,1); one superconducting
flux quantum h/2e corresponds to phi = pi.  The ring is stored in folded MPS order
(photon, 0, 1, Nr-1, 2, Nr-2, ...) so that all ring bonds have range <= 2.
"""
import numpy as np
from scipy.linalg import expm
from tenpy.networks.site import SpinHalfFermionSite, BosonSite, set_common_charges
from tenpy.models.lattice import TrivialLattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg


def photon_site(Nph, g, pad=60):
    site = BosonSite(Nmax=Nph - 1, conserve=None)
    M = Nph + pad
    n = np.arange(1, M)
    X = np.diag(np.sqrt(n), 1) + np.diag(np.sqrt(n), -1)
    D = expm(1j * g * X)[:Nph, :Nph]
    site.add_op('D', D)
    site.add_op('Dd', D.conj().T)
    site.add_op('X', X[:Nph, :Nph])
    site.add_op('X2', (X @ X)[:Nph, :Nph])
    return site


def folded_order(Nr):
    order, lo, hi = [0], 1, Nr - 1
    while lo <= hi:
        order.append(lo)
        if hi != lo:
            order.append(hi)
        lo, hi = lo + 1, hi - 1
    return order


class RingCavityModel(CouplingModel, MPOModel):
    """params: Nr, t (list per bond r->r+1, bond 0 is the junction), tJ, eps_up, eps_dn (lists),
    U (list), Delta (list), omega, g, phi_cl, Nph."""

    def __init__(self, p):
        self.p = p
        Nr = p['Nr']
        self.order = folded_order(Nr)
        self.pos = {r: k + 1 for k, r in enumerate(self.order)}  # MPS index of ring site r
        fs = SpinHalfFermionSite(cons_N='parity', cons_Sz='Sz')
        ph = photon_site(p['Nph'], p['g'])
        set_common_charges([ph, fs], new_charges='independent')
        self.sites = [ph] + [fs] * Nr
        lat = TrivialLattice(self.sites)
        CouplingModel.__init__(self, lat)
        z = [0]
        self.add_onsite_term(p['omega'], 0, 'N')
        for r in range(Nr):
            i = self.pos[r]
            if p['eps_up'][r]:
                self.add_onsite_term(p['eps_up'][r], i, 'Nu')
            if p['eps_dn'][r]:
                self.add_onsite_term(p['eps_dn'][r], i, 'Nd')
            if p['U'][r]:
                self.add_onsite_term(p['U'][r], i, 'NuNd')
            if p['Delta'][r]:
                self.add_onsite_term(p['Delta'][r], i, 'Cdu Cdd', plus_hc=True)
        for r in range(1, Nr):  # ordinary ring bonds r -> r+1 (mod Nr), bond 0 is the junction
            i, j = self.pos[r], self.pos[(r + 1) % Nr]
            for s in ('u', 'd'):
                self.add_coupling(-p['t'][r], i, 'Cd' + s, j, 'C' + s, z, plus_hc=True)
        # junction bond with photon displacement operator
        amp = -p['tJ'] * np.exp(1j * p['phi_cl'])
        i0, i1 = self.pos[0], self.pos[1]
        for s in ('u', 'd'):
            self.add_multi_coupling(amp, [('D', z, 0), ('Cd' + s, z, i1), ('C' + s, z, i0)], plus_hc=True)
        MPOModel.__init__(self, lat, self.calc_H_MPO())

    def current_terms(self):
        """I = -dH/dphi_cl on the junction bond, as list of (coef, term)."""
        p = self.p
        i0, i1 = self.pos[0], self.pos[1]
        out = []
        for s in ('u', 'd'):
            out.append((1j * p['tJ'] * np.exp(1j * p['phi_cl']), [('D', 0), ('Cd' + s, i1), ('C' + s, i0)]))
            out.append((-1j * p['tJ'] * np.exp(-1j * p['phi_cl']), [('Dd', 0), ('Cd' + s, i0), ('C' + s, i1)]))
        return out


def product_state(model, ring_states, photon=0):
    st = [photon] + [None] * model.p['Nr']
    for r, s in enumerate(ring_states):
        st[model.pos[r]] = s
    return MPS.from_product_state(model.lat.mps_sites(), st, bc='finite')


def run_dmrg(model, ring_states, chi=300, sweeps=30, orthogonal_to=(), psi0=None, verbose=False):
    psi = psi0.copy() if psi0 is not None else product_state(model, ring_states)
    opts = {'mixer': True, 'max_E_err': 1e-10, 'max_trunc_err': 1.0, 'max_sweeps': sweeps, 'min_sweeps': 6,
            'trunc_params': {'chi_max': chi, 'svd_min': 1e-11},
            'chi_list': {0: 64, 4: chi}}
    info = dmrg.run(psi, model, opts, orthogonal_to=list(orthogonal_to))
    return info['E'], psi


def observables(model, psi):
    out = {}
    out['n'] = psi.expectation_value('N', [0])[0].real
    out['X'] = psi.expectation_value('X', [0])[0].real
    out['X2'] = psi.expectation_value('X2', [0])[0].real
    out['ReD'] = psi.expectation_value('D', [0])[0]
    out['I'] = sum(c * psi.expectation_value_term(t) for c, t in model.current_terms()).real
    rho = psi.get_rho_segment([0]).to_ndarray()
    out['pn'] = np.real(np.diag(rho))
    out['rho_ph'] = rho
    out['S_ph'] = float(psi.entanglement_entropy(bonds=[0])[0])
    out['chi_max'] = max(psi.chi)
    return out
