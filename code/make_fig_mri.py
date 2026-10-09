"""Figure for the circuit analogue of the magnetorotational instability (code/mri_circuit.py)."""
import json, os
import numpy as np
from make_figures import plt, C1, C2, C3, C4, INK, MUTED, D, F, tag

lin = json.load(open(D + 'mri_circuit_linear.json'))
w_t = lin['w_t']; L = lin['linear']
cases = {c: json.load(open(D + 'mri_circuit_%s.json' % c)) for c in ('loop', 'rayleigh', 'mri', 'strong')
         if os.path.exists(D + 'mri_circuit_%s.json' % c)}
col = {'loop': INK, 'rayleigh': C3, 'mri': C2, 'strong': C1}

fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), constrained_layout=True)
# (a) growth rate vs spring stiffness
a = axs[0]
for wc, c in zip(('0', '1.5', '3', '6'), (INK, C3, C2, C1)):
    a.plot(L['K'], L['s'][wc], color=c, lw=1.2, label=r'$\omega_c=%s\,\omega_t$' % wc)
    if wc != '0':
        a.axhline(1 / (2 * float(wc)), color=c, lw=0.5, ls=':')
a.set_xlabel(r'spring $K/k_t$  ($\leftrightarrow k^2v_A^2$)'); a.set_ylabel(r'growth rate $s/\omega_t$')
a.set_xlim(0, 1.4); a.set_ylim(0, 1.05); a.legend(fontsize=5.5, frameon=False); tag(a, '(a)')
# (b) stability map
b = axs[1]
S = np.array(L['map_s'])
pc = b.pcolormesh(L['map_K'], L['map_wc'], S, cmap='Blues', vmin=0, vmax=1, shading='nearest', rasterized=True)
b.axvline(1, color=MUTED, lw=0.6, ls='--')
b.plot([0], [1], 'o', ms=3, color=C2, clip_on=False)
b.text(0.03, 4.8, 'unstable\n' + r'$0<K<k_t$', fontsize=5.5, color=INK)
b.text(1.04, 4.8, 'stable', fontsize=5.5, color=INK)
b.annotate(r'$K=0$: stable iff $\omega_c>\omega_t$', xy=(0, 1), xytext=(0.25, 2.2), fontsize=5, color=C2,
           arrowprops=dict(arrowstyle='->', color=C2, lw=0.6))
b.set_xlabel(r'$K/k_t$'); b.set_ylabel(r'gyrator $\omega_c/\omega_t$'); tag(b, '(b)')
cb = fig.colorbar(pc, ax=b, pad=0.02, aspect=25); cb.set_label(r'$s/\omega_t$', fontsize=6); cb.ax.tick_params(labelsize=5.5)
# (c) quantum dynamics of the junction phase
c_ = axs[2]
for name, r in cases.items():
    t = np.array(r['t']) * w_t
    c_.semilogy(t, r['phi1sq'], color=col[name], lw=1.1, label=r['label'])
    if r['s_lin'] > 1e-6 * w_t:
        y0 = r['phi1sq'][0]
        tt = t[np.array(r['phi1sq']) < 0.3]
        c_.semilogy(tt, 0.5 * y0 * np.exp(2 * r['s_lin'] / w_t * tt), color=col[name], lw=0.6, ls='--')
c_.set_xlabel(r'$\omega_t\,t$'); c_.set_ylabel(r'$\langle\varphi_1^2\rangle$ (quantum, adiabatic)')
c_.set_ylim(5e-3, 8); tag(c_, '(c)')
fig.legend(*c_.get_legend_handles_labels(), loc='outside lower center', ncol=4, fontsize=5.6, frameon=False)
# (d) final distribution in the MRI case
d = axs[3]
if 'mri' in cases:
    r = cases['mri']; x = np.array(r['x_final']); P = np.array(r['P_final'])
    d.pcolormesh(x, x, P.T / P.max(), cmap='Blues', shading='nearest', rasterized=True)
    d.set_xlim(-3, 3); d.set_ylim(-3, 3)
    # minima of the static potential E(phi1) + K phi1^2 / 2 along phi1 (dotted)
    import mri_circuit as mc
    xx = np.linspace(0, 3, 3001); Vx = mc.E(xx) + 0.5 * r['K'] * mc.k_t * xx ** 2
    for s in (-1, 1):
        d.axvline(s * xx[np.argmin(Vx)], color=MUTED, lw=0.5, ls=':')
d.set_xlabel(r'$\varphi_1$ (junction)'); d.set_ylabel(r'$\varphi_2$'); d.set_title('MRI case, final $|\\psi|^2$', fontsize=6.5)
tag(d, '(d)')
fig.savefig(F + 'fig_mri_circuit.pdf'); fig.savefig(F + 'fig_mri_circuit.png')
print('done', list(cases))
