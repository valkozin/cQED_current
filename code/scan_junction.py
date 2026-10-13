"""How strong can the interaction-driven pi junction be made?  DMRG of the ring without photon (L=14, t=1),
scanning the dot-lead tunnelling t_J, the charging energy U (eps_d=-U/2) and the lead gap Delta.

For each (t_J, Delta, U): odd (doublet) sector at phi = 0, pi/4, pi/2 -> E(phi) = E0 + E2 cos 2phi + E4 cos 4phi,
E_J := |E''(0)|/4 = |E2 + 4 E4| (the quantity entering beta and E_J/hbar*omega), I_c ~ max |dE/dphi|;
even (singlet) sector at phi = 0 -> sign of E_odd - E_even (pi junction requires the doublet to be lower).

usage: python scan_junction.py tJ Delta U1,U2,... chi outfile
"""
import sys, json, time, warnings
import numpy as np
from dmrg_ring import RingCavityModel, run_dmrg
from bdg_general import sc_ring

warnings.filterwarnings('ignore')
tJ, Delta, Us, chi, outfile = float(sys.argv[1]), float(sys.argv[2]), [float(u) for u in sys.argv[3].split(',')], int(sys.argv[4]), sys.argv[5]
init = {'even': ['up', 'down'] * 7, 'odd': ['up'] + ['up', 'down'] * 6 + ['empty']}
out = []
for U in Us:
    t0 = time.time()
    rec = dict(tJ=tJ, Delta=Delta, U=U)
    for sec, phis in (('odd', (0.0, np.pi / 4, np.pi / 2)), ('even', (0.0,))):
        for phi in phis:
            p = sc_ring(14, 1.0, Delta, tJ, Udot=U, dot=True)
            p.update(omega=1.0, g=0.0, phi_cl=phi, Nph=2)
            E, _ = run_dmrg(RingCavityModel(p), init[sec], chi=chi, sweeps=12)
            rec['%s_%.4f' % (sec, phi)] = float(E)
    e0, e1, e2 = rec['odd_0.0000'], rec['odd_0.7854'], rec['odd_1.5708']
    # E(0)=E0+E2+E4, E(pi/4)=E0-E4, E(pi/2)=E0-E2+E4
    E2 = (e0 - e2) / 2; E4 = ((e0 + e2) / 2 - e1) / 2
    rec.update(E2=E2, E4=E4, EJ=abs(E2 + 4 * E4), pi=E2 > 0, odd_minus_even=e0 - rec['even_0.0000'])
    phi = np.linspace(0, np.pi, 400)
    rec['Ic'] = float(np.max(np.abs(2 * E2 * np.sin(2 * phi) + 4 * E4 * np.sin(4 * phi))))
    out.append(rec)
    print('tJ=%.2f Delta=%.2f U=%.2f  E2=%+.5f E4=%+.6f EJ=%.5f Ic=%.5f  E_odd-E_even(0)=%+.4f  %s  t=%.0fs' %
          (tJ, Delta, U, E2, E4, rec['EJ'], rec['Ic'], rec['odd_minus_even'],
           'pi (doublet ground)' if (E2 > 0 and rec['odd_minus_even'] < 0) else '-', time.time() - t0), flush=True)
    json.dump(out, open(outfile, 'w'))
