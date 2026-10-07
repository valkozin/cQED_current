"""Gap equation with an ANGULAR cutoff phi > phi_0 (units v_F = kappa = 1).

Small-angle kinematics: q^2 = xi^2 + k_F^2 phi^2, so phi > phi_0 means
    q > q_f(xi) = sqrt(xi^2 + q_f^2),   q_f = k_F phi_0 ,
i.e. at fixed q the xi-integral runs over |xi| < b = sqrt(q^2 - q_f^2).  Exact:
    int_{-b}^{b} dxi / ((w'^2+xi^2) sqrt(q^2-xi^2))
      = 2/(|w'| sqrt(w'^2+q^2)) * arctan( sqrt(w'^2+q^2) * b / (|w'| q_f) ),
which reduces to pi/(|w'| sqrt(w'^2+q^2)) for q_f -> 0.
Plasmon condition on the transfer: q < min(K, |nu|).  Frequency cutoff |w| < wmax_fac*Omega_K.
Kernel:  Delta(w) = -(alpha/2pi) int dw' int dq nu^2/(nu^2+q/2) W(q,w') Delta(w'),
with W the weight above; the matrix returned is A with  Delta = -alpha A Delta.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

def build(a, qf, wmax_fac=10.0, wmin=1e-6, N=None, nt=140):
    OmK = np.sqrt(a/2); wmax = wmax_fac*OmK
    if N is None: N = int(34*np.log10(wmax/wmin))
    lw = np.linspace(np.log(wmin), np.log(wmax), N); w = np.exp(lw); dt = lw[1]-lw[0]
    wq = np.full(N, dt); wq[0] *= 0.5; wq[-1] *= 0.5
    x, gw = leggauss(nt); A = np.zeros((N, N))
    if qf >= a: return w, A
    for j in range(N):
        wp = w[j]; col = np.zeros(N)
        for sgn in (1, -1):
            nu = w - sgn*wp
            qmax = np.minimum(a, np.abs(nu))
            ok = qmax > qf
            T0 = np.arcsinh(qf/wp); T1 = np.arcsinh(np.maximum(qmax, qf)/wp)
            t = T0 + 0.5*(T1-T0)[:, None]*(x[None, :]+1); gwt = 0.5*(T1-T0)[:, None]*gw[None, :]
            q = wp*np.sinh(t)
            den = nu[:, None]**2 + q/2
            fk = np.where(den > 0, nu[:, None]**2/np.where(den > 0, den, 1), 0.0)
            s = np.sqrt(wp*wp + q*q)                       # sqrt(w'^2+q^2)
            b = np.sqrt(np.maximum(q*q - qf*qf, 0.0))
            # angular weight relative to the full one pi/(|w'| s): (2/pi) arctan(s b/(|w'| q_f))
            ang = (2/np.pi)*np.arctan(s*b/(wp*qf)) if qf > 0 else np.ones_like(q)
            col += np.sum(fk*ang*gwt, axis=1)*ok
        A[:, j] = col*wq[j]/(2*np.pi)
    return w, A

def mu_min(a, qf, **kw):
    w, A = build(a, qf, **kw)
    if not A.any(): return 0.0
    ev = np.linalg.eigvals(A)
    real = np.abs(ev.imag) < 1e-8*np.maximum(1, np.abs(ev.real))
    r = ev.real[real]
    return r.min() if (r < 0).any() else 0.0
