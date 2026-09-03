# Deprecated: not used for reported paper results. See rerun_fair_protocol.py
# Per-seed validation selection for RA-GAT on public Cora.
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from data_gen import load_dataset
from train import run_once

FIG = os.path.join(os.path.dirname(_HERE), 'figures')
SEEDS = [42, 43, 44]
CFGS = [
    dict(name='lr01_h64', identity_init=True, extra_lr=None, lr=0.01, hidden=64, epochs=150, wd=5e-4),
    dict(name='lr005_h64', identity_init=True, extra_lr=None, lr=0.005, hidden=64, epochs=150, wd=5e-4),
    dict(name='lr02_h64', identity_init=True, extra_lr=None, lr=0.02, hidden=64, epochs=150, wd=5e-4),
    dict(name='lr01_h96', identity_init=True, extra_lr=None, lr=0.01, hidden=96, epochs=150, wd=5e-4),
    dict(name='lr01_wd1e4', identity_init=True, extra_lr=None, lr=0.01, hidden=64, epochs=150, wd=1e-4),
    dict(name='lr01_ep200', identity_init=True, extra_lr=None, lr=0.01, hidden=64, epochs=200, wd=5e-4),
    dict(name='lr01_elr001', identity_init=True, extra_lr=0.001, lr=0.01, hidden=64, epochs=150, wd=5e-4),
]


def main():
    d = load_dataset('Cora')
    all_rows = []
    chosen = []
    for s in SEEDS:
        print('=== Cora seed %d ===' % s, flush=True)
        rows = []
        for cfg in CFGS:
            t0 = time.time()
            r = run_once(d, 'RA-GAT', seed=s, identity_init=True,
                         extra_lr=cfg['extra_lr'], lr=cfg['lr'], hidden=cfg['hidden'],
                         epochs=cfg['epochs'], wd=cfg['wd'])
            rec = dict(seed=s, cfg=cfg['name'], auc=r['auc'], ap=r['ap'],
                       hits20=r['hits20'], best_val=r['best_val_auc'],
                       n_params=r['n_params'], train_time=r['train_time'],
                       sec=time.time() - t0)
            rows.append(rec)
            print('  %-12s val=%.4f AUC=%.4f AP=%.4f Hits=%.4f (%.1fs)' % (
                cfg['name'], rec['best_val'], rec['auc'], rec['ap'], rec['hits20'], rec['sec']),
                  flush=True)
        best = max(rows, key=lambda x: x['best_val'])
        print('  VAL-BEST %s val=%.4f testAUC=%.4f' % (best['cfg'], best['best_val'], best['auc']),
              flush=True)
        chosen.append(best)
        all_rows.extend(rows)
    aucs = [c['auc'] for c in chosen]
    aps = [c['ap'] for c in chosen]
    hits = [c['hits20'] for c in chosen]
    print('\nVAL-SELECTED 3-seed mean AUC=%.4f±%.4f AP=%.4f±%.4f Hits=%.4f±%.4f' % (
        np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps), np.mean(hits), np.std(hits)),
          flush=True)
    print('GAT reference AUC 0.884 AP 0.881 Hits 0.455', flush=True)
    out = {'rows': all_rows, 'chosen': chosen,
           'mean': dict(auc=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                        ap=float(np.mean(aps)), ap_std=float(np.std(aps)),
                        hits=float(np.mean(hits)), hits_std=float(np.std(hits)))}
    path = os.path.join(FIG, 'ragat_cora_seed_sweep.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
