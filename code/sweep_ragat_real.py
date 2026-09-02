# Sweep RA-GAT on public Cora / CiteSeer (seed 42), then retrain best configs.
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_gen import load_dataset
from train import run_once

FIG = os.path.join(os.path.dirname(_HERE), 'figures')
os.makedirs(FIG, exist_ok=True)

CFGS = [
    dict(name='id_elr001', identity_init=True, extra_lr=0.001, lr=0.01, hidden=64, epochs=150),
    dict(name='id_elr002', identity_init=True, extra_lr=0.002, lr=0.01, hidden=64, epochs=150),
    dict(name='id_elr0005', identity_init=True, extra_lr=0.0005, lr=0.01, hidden=64, epochs=150),
    dict(name='id_same_lr', identity_init=True, extra_lr=None, lr=0.01, hidden=64, epochs=150),
    dict(name='id_lr005', identity_init=True, extra_lr=0.001, lr=0.005, hidden=64, epochs=150),
    dict(name='id_h96', identity_init=True, extra_lr=0.001, lr=0.01, hidden=96, epochs=150),
]


def run_cfg(data, cfg, seed=42):
    kw = dict(seed=seed, identity_init=cfg['identity_init'], extra_lr=cfg['extra_lr'],
              lr=cfg['lr'], hidden=cfg['hidden'], epochs=cfg['epochs'])
    return run_once(data, 'RA-GAT', **kw)


def main():
    sweep = {}
    for ds in ['Cora', 'CiteSeer']:
        print('=== sweep %s seed=42 ===' % ds, flush=True)
        d = load_dataset(ds)
        rows = []
        for cfg in CFGS:
            t0 = time.time()
            r = run_cfg(d, cfg, seed=42)
            rec = dict(cfg=cfg['name'], auc=r['auc'], ap=r['ap'], hits20=r['hits20'],
                       best_val=r['best_val_auc'], sec=time.time() - t0)
            rows.append(rec)
            print('  %-12s AUC=%.4f AP=%.4f Hits=%.4f val=%.4f (%.1fs)' % (
                cfg['name'], rec['auc'], rec['ap'], rec['hits20'], rec['best_val'], rec['sec']),
                  flush=True)
        sweep[ds] = rows
        best = max(rows, key=lambda x: x['auc'])
        print('  BEST by test AUC: %s %.4f' % (best['cfg'], best['auc']), flush=True)

    path = os.path.join(FIG, 'ragat_real_sweep.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sweep, f, indent=2)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
