"""Compare every full-DMRG point with the effective circuit model; flag points that deviate by more than
`tol` of I_c outside the crossover (where the brackets are expected). usage: python validate_grid.py [tol]"""
import json, sys, glob, os, numpy as np
from make_phase_figs import load_full, load_eff, Ic_of
tol = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
full, eff = load_full(), load_eff(); Icf = Ic_of(eff)
flag = []
for U in sorted(full):
    d = full[U]
    if U not in eff['Us']:
        continue
    ke = eff['Us'].index(U); ic = Icf(U)
    Ie = np.abs(np.interp(d['g'], eff['gs'], eff['I'][ke])); Id = np.abs(d['I'])
    ne = np.interp(d['g'], eff['gs'], eff['n'][ke])
    dev = (Id - Ie) / ic
    bad = np.abs(dev) > tol
    print('U=%.2f  n=%d  max|dI|/Ic=%.3f  max|dn|=%.2f  flagged g: %s' % (U, len(d['g']), np.max(np.abs(dev)), np.max(np.abs(d['n'] - ne)),
          ['%.2f(%+.3f,%s)' % (g, v, 'crossover' if (e > 0.02 and e < 0.9 * np.max(Ie / ic)) else 'ordered') for g, v, e in zip(d['g'][bad], dev[bad], (Ie / ic)[bad])]))
    for g, v, e in zip(d['g'][bad], dev[bad], (Ie / ic)[bad]):
        flag.append((U, float(g), float(v), float(e)))
json.dump(flag, open('../data/full/flagged.json', 'w'))
