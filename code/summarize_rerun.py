"""Tables of the recomputed beta scans (data/rerun) against the adiabatic theory and the old (warm-start) runs.
usage: python summarize_rerun.py"""
import json, os, glob, numpy as np
from bo import solve
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')


def rows(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(D, pattern))):
        out += json.load(open(f))['rows']
    return sorted(out, key=lambda r: r['beta'])


print('=== interacting-dot pi loop, L=10, U=2, E_J/omega=10, no seed (cat state)')
c = np.load(os.path.join(D, 'harmonics_udot_U2_Nr10.npy')); om = 7.39e-4; EJ = c[2].real
d4 = {r['beta']: r for r in rows('rerun/dmrg_udot_U2_Nr10_r10_chi400.json')}
old = {r['beta']: r for r in json.load(open(os.path.join(D, 'dmrg_udot_U2_r10.json')))['rows']}
print('beta   g      Phi*/(Phi0/2) DMRG / BO    delta/om DMRG(chi300) / chi400 / BO     n DMRG/BO   <X> DMRG   E0 DMRG-BO   sumrule dev   dE(C-S)  old L14: Phi* delta')
for r in rows('rerun/dmrg_udot_U2_Nr10_r10.json'):
    b, g = r['beta'], r['g']
    o = solve(c, g, om, 0.0, Nphot=300)
    r4 = d4.get(b)
    print('%.2f  %.4f  %.4f / %.4f   %.3e / %s / %.3e   %.3f/%.3f   %+.1e   %+.1e   %.1e   %s   %s' % (
        b, g, r['X01'] * g / (np.pi / 2), o['X01'] * g / (np.pi / 2), (r['E1'] - r['E0']) / om,
        ('%.3e' % ((r4['E1'] - r4['E0']) / om)) if r4 else '   -    ', (o['E'][1] - o['E'][0]) / om,
        r['n0'], o['n'][0], r['X0'], r['E0'] - o['E'][0], abs(r['I0'] - r['I_vir0']),
        ('%.1e' % (r['E_starts']['C'] - r['E_starts']['S'])) if 'C' in r['E_starts'] else '  -  ',
        ('%.3f %.2e' % (old[b]['X01'] * old[b]['g'] / (np.pi / 2), (old[b]['E1'] - old[b]['E0']) / 7.61e-4)) if b in old else ''))

print('\n=== weak link + Phi0/2, L=14, E_J/omega=10 (two-start protocol)')
bw = json.load(open(os.path.join(D, 'bo_weaklink.json'))); cw = np.array(bw['c'][0]) + 1j * np.array(bw['c'][1]); EJw = bw['EJ']; omw = EJw / 10
m = np.arange(len(cw)); scale = abs(np.real(np.sum(-(m ** 2) * cw * np.exp(1j * m * np.pi / 2)))) / (4 * EJw); Ic = 2 * EJw
oldw = {r['beta']: r for r in json.load(open(os.path.join(D, 'dmrg_weaklink_r10.json')))['rows']}
print('beta_nom beta   I/Ic DMRG / BO / MF / old   (E0-E_BO)/om  (E_MF-E0)/om  gap/om DMRG / BO   n DMRG/BO   sumrule dev/Ic   starts: dE, X_A, X_C')
for r in rows('rerun/dmrg_weaklink_Nr14_r10_*.json'):
    b, g = r['beta'], r['g']
    o = solve(cw, g, omw, np.pi / 2, Nphot=400)
    st = r['E_starts']; xs = r['X_starts']
    print('%.2f    %.3f  %.4f / %.4f / %.4f / %s   %+.1e   %+.2f   %.4f / %.4f   %.3f/%.3f   %.1e   %.1e %s' % (
        b, b * scale, abs(r['I0']) / Ic, abs(o['I'][0]) / Ic, abs(r['I_mf']) / Ic,
        ('%.4f' % (abs(omw * oldw[b]['X0'] / (2 * oldw[b]['g'])) / Ic)) if b in oldw else '  -   ',
        (r['E0'] - o['E'][0]) / omw, (r['E_mf'] - r['E0']) / omw, (r['E1'] - r['E0']) / omw, (o['E'][1] - o['E'][0]) / omw,
        r['n0'], o['n'][0], abs(r['I0'] - r['I_vir0']) / Ic, max(st.values()) - min(st.values()),
        ' '.join('%s:%.3f' % (k, v) for k, v in xs.items())))
