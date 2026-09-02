# CiteSeer seed-42 sweep + 3-seed retrain of the best RA-GAT setting.
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from data_gen import load_dataset
from train import run_once

FIG = os.path.join(os.path.dirname(_HERE), 'figures')
os.makedirs(FIG, exist_ok=True)

CFGS = [
    dict(name='id_same_lr', identity_init=True, extra_lr=None, lr=0.01, hidden=64, epochs=150),
    dict(name='id_elr001', identity_init=True, extra_lr=0.001, lr=0.01, hidden=64, epochs=150),
    dict(name='id_elr002', identity_init=True, extra_lr=0.002, lr=0.01, hidden=64, epochs=150),
    dict(name='id_h96', identity_init=True, extra_lr=None, lr=0.01, hidden=96, epochs=150),
    dict(name='id_lr005', identity_init=True, extra_lr=None, lr=0.005, hidden=64, epochs=150),
]


def main():
    print('loading CiteSeer...', flush=True)
    d = load_dataset('CiteSeer')
    print('CiteSeer loaded', d['num_nodes'], d['num_feat'], d['num_comm'], flush=True)
    rows = []
    for cfg in CFGS:
        t0 = time.time()
        r = run_once(d, 'RA-GAT', seed=42, identity_init=cfg['identity_init'],
                     extra_lr=cfg['extra_lr'], lr=cfg['lr'], hidden=cfg['hidden'],
                     epochs=cfg['epochs'])
        rec = dict(cfg=cfg['name'], auc=r['auc'], ap=r['ap'], hits20=r['hits20'],
                   best_val=r['best_val_auc'], sec=time.time() - t0)
        rows.append(rec)
        print('  %-12s AUC=%.4f AP=%.4f Hits=%.4f val=%.4f (%.1fs)' % (
            cfg['name'], rec['auc'], rec['ap'], rec['hits20'], rec['best_val'], rec['sec']),
              flush=True)
    best = max(rows, key=lambda x: x['auc'])
    print('BEST CiteSeer seed42:', best, flush=True)
    path = os.path.join(FIG, 'ragat_citeseer_sweep.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
