"""Circuit analogue of the magnetorotational instability (MRI): a pi junction stabilized by a gyrator and
destabilized by a weak inductive spring.

Two flux (electron-phase) coordinates phi1, phi2 with charging energy E_C each; phi1 sits across the pi junction
E(phi1) (DMRG, t_J = t, U = 6t, maximum at phi1 = 0); both nodes are shunted by inductive springs K; the nodes are
coupled by a gyrator (a uniform "magnetic field" gamma in the (phi1, phi2) plane, symmetric gauge):

    H = E_C (p1 + gamma phi2 / 2)^2 + E_C (p2 - gamma phi1 / 2)^2 + K (phi1^2 + phi2^2) / 2 + E(phi1),  [phi_j, p_j] = i.

Equations of motion (v_j = dphi_j/dt):  v1' - w_c v2 = -2 E_C (K phi1 + E'(phi1)),  v2' + w_c v1 = -2 E_C K phi2,
w_c = 2 E_C gamma.  Linearized (E' = -k_t phi1) these are Hill's equations of the shearing sheet:
    Coriolis 2 Omega <-> w_c,  tidal term 2 q Omega^2 = -dOmega^2/dln r <-> w_t^2 = 2 E_C k_t,
    magnetic tension k^2 v_A^2 <-> w_K^2 = 2 E_C K.
Results: unstable iff 0 < K < k_t (for any gyrator); at K = 0 stable iff w_c > w_t (Rayleigh); max growth rate
w_t^2 / (2 w_c) (Balbus-Hawley: |dOmega/dln r| / 2).

Quantum part: the same H with the adiabatic (Born-Oppenheimer) junction energy E(phi1), solved on a 2D grid by
the split-operator method (each kinetic term is diagonal in a mixed representation); initial state = Gaussian at the
symmetric point phi = 0 (ground state of E_C p^2 + k_t phi^2 / 2 in both coordinates).

usage: python mri_circuit.py [case ...]   -> ../data/mri_circuit_linear.json, ../data/mri_circuit_<case>.json
       (TMAX env: duration in units of 1/w_t, default 40);  figure: python make_fig_mri.py
"""
import json, sys, time
import numpy as np
import phase_diagram_v2 as pd

# ---------------------------------------------------------------- parameters
coef = pd.coeffs(pd.load_curves('../data/tJ1.0/udot_phi*.json'))['odd'][6.0]       # E(phi) = sum c_m cos(m phi)
m = np.arange(len(coef))
k_t = float(np.sum(m ** 2 * coef))                     # -E''(0) > 0: negative stiffness of the pi junction
E_C = 0.003 * 0.12 ** 2                                # hbar*omega g^2 of the mode used in the cavity runs
w_t = np.sqrt(2 * E_C * k_t)
w_c = 3.0 * w_t                                        # gyrator: Rayleigh-stable at K = 0 (kappa^2 = 8 w_t^2)


def E(phi):
    return np.cos(np.outer(np.ravel(phi), m)) @ coef - coef.sum()


# ---------------------------------------------------------------- linear theory
def growth(kK, wc):
    """max Re lambda of the linearized equations, frequencies in units of w_t, K in units of k_t"""
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1 - kK, 0, 0, wc], [0, -kK, -wc, 0]], float)
    return max(np.linalg.eigvals(A).real)


def linear():
    ks = np.linspace(0, 1.4, 701)
    out = {}
    for wc in (0.0, 1.5, 3.0, 6.0):
        out['%g' % wc] = [growth(k, wc) for k in ks]
    wcs = np.linspace(0, 6, 121)
    smap = [[growth(k, wc) for k in ks[::10]] for wc in wcs]
    return dict(K=ks.tolist(), s=out, map_K=ks[::10].tolist(), map_wc=wcs.tolist(), map_s=smap)


# ---------------------------------------------------------------- quantum (split operator)
def evolve(K, wc, tmax, dt=0.5, Lbox=3.5, N=512, nout=200):
    gamma = wc / (2 * E_C)
    x = (np.arange(N) - N // 2) * (2 * Lbox / N)
    p = 2 * np.pi * np.fft.fftfreq(N, d=2 * Lbox / N)
    X1, X2 = np.meshgrid(x, x, indexing='ij')
    V = E(X1).reshape(N, N) + 0.5 * K * (X1 ** 2 + X2 ** 2)
    eV = np.exp(-0.5j * dt * V)
    eT1 = np.exp(-0.5j * dt * E_C * (p[:, None] + 0.5 * gamma * x[None, :]) ** 2)   # (p1, phi2) representation
    eT2 = np.exp(-1.0j * dt * E_C * (p[None, :] - 0.5 * gamma * x[:, None]) ** 2)   # (phi1, p2) representation
    s2 = 0.5 * np.sqrt(2 * E_C / k_t)                   # <phi^2> of the reference Gaussian
    psi = np.exp(-(X1 ** 2 + X2 ** 2) / (4 * s2)).astype(complex)
    psi /= np.linalg.norm(psi)
    nsteps = int(round(tmax / dt)); every = max(1, nsteps // nout)
    rec = dict(t=[], phi1sq=[], phi2sq=[], Irms=[], edge=[])
    dE = -(np.sin(np.outer(x, m)) * m) @ coef            # E'(phi1)
    for n in range(nsteps + 1):
        if n % every == 0:
            P = np.abs(psi) ** 2
            P1 = P.sum(1)
            rec['t'].append(n * dt); rec['phi1sq'].append(float(P1 @ x ** 2)); rec['phi2sq'].append(float(P.sum(0) @ x ** 2))
            rec['Irms'].append(float(np.sqrt(P1 @ dE ** 2))); rec['edge'].append(float(P[:N // 16].sum() + P[-N // 16:].sum() + P[:, :N // 16].sum() + P[:, -N // 16:].sum()))
        if n == nsteps:
            break
        psi *= eV
        psi = np.fft.ifft(eT1 * np.fft.fft(psi, axis=0), axis=0)
        psi = np.fft.ifft(eT2 * np.fft.fft(psi, axis=1), axis=1)
        psi = np.fft.ifft(eT1 * np.fft.fft(psi, axis=0), axis=0)
        psi *= eV
    P = np.abs(psi) ** 2
    rec['P_final'] = P[::4, ::4].tolist(); rec['x_final'] = x[::4].tolist()
    return rec


CASES = {   # name: (K/k_t, w_c/w_t, label)
    'loop': (0.5, 0.0, r'no gyrator, $K=0.5k_t$ (plain $\pi$ loop)'),
    'rayleigh': (0.0, 3.0, r'gyrator, $K=0$ (Rayleigh-stable)'),
    'mri': (0.472, 3.0, r'gyrator, weak spring $K=0.47k_t$ (MRI)'),
    'strong': (1.3, 3.0, r'gyrator, strong spring $K=1.3k_t$'),
}

if __name__ == '__main__':
    print('k_t = %.4f t, E_C = %.2e t, w_t = %.3e t, w_c = %.3e t, gamma = %.1f' % (k_t, E_C, w_t, w_c, w_c / (2 * E_C)))
    json.dump(dict(k_t=k_t, E_C=E_C, w_t=w_t, w_c=w_c, linear=linear()), open('../data/mri_circuit_linear.json', 'w'))
    tmax = float(__import__('os').environ.get('TMAX', 40)) / w_t
    for name in (sys.argv[1:] or CASES):
        kK, wcr, lab = CASES[name]
        t0 = time.time()
        r = evolve(kK * k_t, wcr * w_t, tmax)
        r.update(K=kK, wc=wcr, label=lab, s_lin=growth(kK, wcr) * w_t)
        json.dump(r, open('../data/mri_circuit_%s.json' % name, 'w'))
        print('%-8s s_lin/w_t=%.3f  <phi1^2>: %.3f -> max %.3f  edge=%.1e  t=%.0fs' %
              (name, growth(kK, wcr), r['phi1sq'][0], max(r['phi1sq']), max(r['edge']), time.time() - t0), flush=True)
