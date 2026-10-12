import json, sys, numpy as np
def harmonics(phi, E, M=4):
    # E(phi) symmetric about 0 and pi -> cosine series on [0,pi]
    A = np.cos(np.outer(phi, np.arange(M + 1)))
    c, *_ = np.linalg.lstsq(A, E, rcond=None)
    return c
if __name__ == '__main__':
    for f in sys.argv[1:]:
        rows = json.load(open(f))
        for U in sorted(set(r['U'] for r in rows)):
            rr = sorted([r for r in rows if r['U'] == U], key=lambda r: r['phi'])
            phi = np.array([r['phi'] for r in rr])
            for s in ('even', 'odd'):
                E = np.array([r[s] for r in rr])
                c = harmonics(phi, E)
                print('%s U=%.2f %-4s E(0)=%.6f  E1=%+.2e E2=%+.3e E3=%+.1e E4=%+.1e  Ic=%.4f' % (f.split('/')[-1], U, s, c[0], c[1], c[2], c[3], c[4], 2*abs(c[2])))
