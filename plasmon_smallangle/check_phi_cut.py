"""Is a cut on the angle phi the same as a cut on q after xi is integrated out?
Units v_F = kappa = 1.  U(nu,q) = 2 pi alpha/q * nu^2/(nu^2+q/2), alpha=1 (drops out of ratios).
Compare, for one (w', nu):
  (D) direct 2D integral over (xi, phi), exact kinematics q^2 = xi^2 + 2 kF (kF+xi)(1-cos phi),
      measure (kF+xi) dxi dphi /(2pi)^2, cuts |phi|>phi0 and q<qmax
  (S) same in small-angle kinematics q^2 = xi^2 + kF^2 phi^2, measure kF dxi dphi/(2pi)^2
  (A) reduced formula with the arctan weight (what the angular-cutoff notes use)
  (Q) reduced formula with a plain lower cut q>q_f (NOT an angle cut)
"""
import numpy as np
from scipy.integrate import quad, dblquad

def U(nu, q): return 2*np.pi/q * nu**2/(nu**2 + q/2)

def direct(wp, nu, kF, phi0, qmax, exact=True):
    def q_of(xi, ph):
        return np.sqrt(xi**2 + (2*kF*(kF+xi)*(1-np.cos(ph)) if exact else kF**2*ph**2))
    def f(ph, xi):
        q = q_of(xi, ph)
        if q >= qmax or q == 0: return 0.0
        meas = (kF+xi) if exact else kF
        return meas*U(nu, q)/(wp**2+xi**2)/(2*np.pi)**2
    tot = 0.0
    # phi>phi0 (x2 for phi<-phi0); upper phi limit where q can still be < qmax
    phmax = lambda xi: min(np.pi, 2*np.arcsin(min(1, qmax/(2*kF)))+1e-12) if exact else qmax/kF
    val, _ = dblquad(f, -qmax, qmax, lambda xi: phi0, phmax, epsabs=1e-11, epsrel=1e-8)
    return 2*val

def reduced_arctan(wp, nu, kF, phi0, qmax):
    qf = kF*phi0
    def g(q):
        s = np.sqrt(wp**2+q**2); b = np.sqrt(max(q*q-qf*qf, 0))
        ang = 1.0 if qf == 0 else (2/np.pi)*np.arctan(s*b/(wp*qf))
        return q*U(nu, q)*ang/(wp*s)/(2*np.pi)
    return quad(g, qf, qmax, limit=400, epsabs=1e-12)[0]

def reduced_qcut(wp, nu, kF, phi0, qmax):
    qf = kF*phi0
    return quad(lambda q: q*U(nu, q)/(wp*np.sqrt(wp**2+q**2))/(2*np.pi), qf, qmax, limit=400)[0]

kF = 5.0          # = 1/(N alpha), e.g. N=2, alpha=0.1
print(f"kF = {kF} kappa\n")
print(" w'      nu     phi0   qmax |  direct(exact)  direct(small-ang)  arctan-formula   q-cut formula")
for wp, nu, phi0, qmax in [(0.3,0.8,0.02,0.3),(0.01,0.5,0.02,0.3),(1e-3,0.5,0.04,0.3),
                           (0.05,0.5,0.0,0.3),(0.02,0.3,0.03,0.25),(0.2,0.9,0.05,0.3)]:
    D  = direct(wp, nu, kF, phi0, qmax, exact=True)
    Ds = direct(wp, nu, kF, phi0, qmax, exact=False)
    A  = reduced_arctan(wp, nu, kF, phi0, qmax)
    Q  = reduced_qcut(wp, nu, kF, phi0, qmax)
    print(f"{wp:6g} {nu:5g} {phi0:6g} {qmax:5g} | {D:13.6e}  {Ds:15.6e}  {A:15.6e}  {Q:14.6e}")
