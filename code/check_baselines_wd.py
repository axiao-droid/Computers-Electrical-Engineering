# Fairness check: GAT / GATv2 with wd=1e-4 on Cora and CiteSeer.
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from data_gen import load_dataset
from train import run_once

SEEDS = [42, 43, 44]


def main():
    for ds in ['Cora', 'CiteSeer']:
        d = load_dataset(ds)
        for m in ['GAT', 'GATv2']:
            aucs, aps, hits = [], [], []
            print('=== %s %s wd=1e-4 ===' % (ds, m), flush=True)
            for s in SEEDS:
                r = run_once(d, m, seed=s, wd=1e-4)
                print('  seed=%d AUC=%.4f AP=%.4f Hits=%.4f val=%.4f' % (
                    s, r['auc'], r['ap'], r['hits20'], r['best_val_auc']), flush=True)
                aucs.append(r['auc']); aps.append(r['ap']); hits.append(r['hits20'])
            print('  mean AUC=%.4f AP=%.4f Hits=%.4f' % (
                np.mean(aucs), np.mean(aps), np.mean(hits)), flush=True)


if __name__ == '__main__':
    main()
