"""Notebook mean field (same algorithm, upward g sweep with warm start) for the notebook model at a given omega."""
import numpy as np, sys
from mean_field import run
from model import DEFAULTS
om = float(sys.argv[1])
p = dict(DEFAULTS, omegaR=om)
res = run(p, glist=np.arange(0, 1.2001, 0.02), Nphot=int(sys.argv[2]) if len(sys.argv) > 2 else 200)
np.save('../data/mf_notebook_om%g.npy' % om, res)
