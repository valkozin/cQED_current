"""Re-run selected (U, g) points with the annealed start only (STARTS=C) and merge them into data/full/U*_odd.json
if they are lower in energy.  usage: python repair_points.py U:g,g,... U:g,..."""
import json, os, subprocess, sys
for job in sys.argv[1:]:
    U, gs = job.split(':')
    tmp = '../data/full/repair_U%s.json' % U
    if os.path.exists(tmp):
        os.remove(tmp)
    env = dict(os.environ, STARTS='C', OMP_NUM_THREADS='1')
    subprocess.run(['python3', 'run_full_grid.py', '10', U, '0.5', '7.39e-4', '1e-3', 'odd', gs, '300', '50', tmp, '40'], env=env, stdout=open(tmp.replace('.json', '.log'), 'w'), stderr=subprocess.STDOUT)
    main = '../data/full/U%s_odd.json' % float(U)
    d = json.load(open(main)); new = {r['g']: r for r in json.load(open(tmp))['rows']}
    for k, r in enumerate(d['rows']):
        if r['g'] in new and new[r['g']]['E0'] < r['E0']:
            rr = new[r['g']]; rr['E_starts'] = dict(r['E_starts'], **rr['E_starts']); rr['X_starts'] = dict(r['X_starts'], **rr['X_starts'])
            print('U=%s g=%.2f replaced: E %.9f -> %.9f, X %.3f -> %.3f' % (U, r['g'], r['E0'], rr['E0'], r['X'], rr['X']))
            d['rows'][k] = rr
        elif r['g'] in new:
            print('U=%s g=%.2f kept (new E %.9f >= %.9f)' % (U, r['g'], new[r['g']]['E0'], r['E0']))
    json.dump(d, open(main, 'w'))
