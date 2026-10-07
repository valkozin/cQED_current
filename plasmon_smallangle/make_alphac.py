"""Critical coupling of the angular-cutoff gap equation (Eq. ang of plasmon_smallangle.tex).

A depends on alpha only through q_f = phi_0/(N alpha) (units kappa = v_F = 1), so
  * the curve alpha_c(phi_0) is read off parametrically from mu(q_f):
        alpha_c = -1/mu(q_f),  phi_0 = N alpha_c q_f      (Fig. a, no interpolation),
  * the table at fixed phi_0 solves  alpha * mu(phi_0/(N alpha)) = -1  by root finding.
Writes mu_qf.json, alphac_table.txt and alphac_phi0.png.
Run:  python3 make_alphac.py
"""
import json
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solve_angular_cutoff import mu_min

NF = 2
CASES = [(a, E) for a in (0.3, 1.0) for E in (1e-4, 1e-6, 1e-8)]
FR = [0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375, 0.45, 0.525, 0.6,
      0.675, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97]
PHI0 = [0.05, 0.1, 0.2, 0.3]


def alpha_c(a, E, phi0):
    g = lambda al: al*mu_min(a, phi0/(NF*al), wmin=E) + 1.0
    return brentq(g, phi0/(NF*a)*1.0001, 10.0, xtol=1e-5)


res = {f"{a}_{E:g}": [mu_min(a, f*a, wmin=E) for f in FR] for a, E in CASES}
json.dump({"fr": FR, "res": res}, open("mu_qf.json", "w"))

lines = ["alpha_c (N=2)      " + "  ".join(f"phi0={p:<5g}" for p in PHI0)]
for a, E in CASES:
    lines.append(f"a={a}, E={E:g}  " + "  ".join(f"{alpha_c(a, E, p):10.3f}" for p in PHI0))
for a in (0.3, 1.0):
    lines.append(f"bound, a={a}     " + "  ".join(f"{p/(NF*a):10.3f}" for p in PHI0))
open("alphac_table.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4))
blues = {1e-4: "#9ec2e6", 1e-6: "#2f67b1", 1e-8: "#163d7a"}
for a, E in CASES:
    mu = np.array(res[f"{a}_{E:g}"]); al = -1/mu; phi0 = NF*al*np.array(FR)*a
    ax.plot(phi0, al, color=blues[E], lw=2.2, ls="-" if a == 0.3 else "--",
            label=rf"$a={a}$, $E=10^{{{int(np.log10(E))}}}\,\kappa v_F$")
p = np.linspace(0, 0.5, 2)
ax.plot(p, p/(NF*0.3), color="0.6", lw=1.2, label=r"$\alpha=\phi_0/Na$ ($a=0.3$)")
ax.plot(p, p/(NF*1.0), color="0.6", lw=1.2, ls="--", label=r"$\alpha=\phi_0/Na$ ($a=1.0$)")
ax.set(xlim=(0, 0.5), ylim=(0, 4), xlabel=r"small-angle cutoff $\phi_0$ (rad)",
       ylabel=r"critical coupling $\alpha_c$")
ax.set_title(r"(a) $\alpha_c(\phi_0)$, $N=2$; grey: window closure $\alpha=\phi_0/(Na)$",
             loc="left", fontsize=11)
ax.legend(ncol=2, fontsize=8.5, frameon=False, loc="upper left")
mu = np.array(res["0.3_1e-06"])
bx.semilogy(FR, np.abs(mu)/abs(mu[0]), "o-", color=blues[1e-6], lw=2.2, ms=5)
bx.set(xlabel=r"$q_f/K$  (angles $|\phi|<q_f/k_F$ removed)",
       ylabel=r"$|\mu(q_f)|\,/\,|\mu(0.02K)|$")
bx.set_title(r"(b) pairing eigenvalue vs removed angles ($a=0.3$, $E=10^{-6}\kappa v_F$)",
             loc="left", fontsize=11)
for z in (ax, bx):
    z.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("alphac_phi0.png", dpi=150)
