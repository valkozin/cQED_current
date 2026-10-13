"""spline representation of a symmetric E(phi) surface (shared by effective_grid.py and analyze_nonadiabatic.py)"""
import numpy as np
from scipy.interpolate import CubicSpline


def spline_V(pts):
    """even E(phi) on [0, pi], extended by symmetry E(-phi) = E(phi), E(phi + 2pi) = E(phi) (electron phase period 2pi)"""
    ph, E = map(np.array, zip(*sorted(pts)))
    ph2 = np.concatenate([ph, 2 * np.pi - ph[-2::-1]]); E2 = np.concatenate([E, E[-2::-1]])
    cs = CubicSpline(ph2, E2, bc_type='periodic')
    return (lambda x: cs(np.mod(x, 2 * np.pi))), (lambda x: cs(np.mod(x, 2 * np.pi), 1))
