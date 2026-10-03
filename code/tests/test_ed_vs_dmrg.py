import sys; sys.path.insert(0,'..')
sys.path.insert(0, '.')
import numpy as np
from dmrg_ring import *
from bdg_general import sc_ring
from tenpy.algorithms.exact_diag import ExactDiag
p = sc_ring(4, 1.0, 1.0, 0.5); p.update(omega=0.1, g=0.6, phi_cl=np.pi/2+0.05, Nph=12)
m = RingCavityModel(p)
E, psi = run_dmrg(m, ['up','down']*2, chi=200, sweeps=20)
o = observables(m, psi)
print('DMRG E', E, 'I_bond', o['I'], 'virial', p['omega']*o['X']/(2*p['g']), 'X', o['X'])
ed = ExactDiag(m, charge_sector=psi.get_total_charge(), max_size=1e9)
ed.build_full_H_from_mpo(); ed.full_diagonalization()
E0, v0 = ed.groundstate()
psi_ed = ed.full_to_mps(v0)
o2 = observables(m, psi_ed)
print('ED   E', E0, 'I_bond', o2['I'], 'virial', p['omega']*o2['X']/(2*p['g']), 'X', o2['X'])
