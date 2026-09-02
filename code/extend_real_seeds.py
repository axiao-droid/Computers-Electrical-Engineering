# Extend public Cora/CiteSeer runs with seeds 45 and 46 under the frozen protocol.
# Baselines: default Adam (lr=0.01, wd=5e-4).
# RA-GAT: identity-init extra operators; Cora lr=0.01 wd=1e-4; CiteSeer lr=0.005 wd=1e-4.
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from data_gen import load_dataset
from train import run_once

FIG = os.path.join(os.path.dirname(_HERE), 'figures')
SRC = os.path.join(FIG, 'real_citation_results.json')
MODELS = ['GraphSAGE', 'GAT', 'GATv2', 'RA-GAT']
DATASETS = ['Cora', 'CiteSeer']
NEW_SEEDS = [45, 46]
ALL_SEEDS = [42, 43, 44, 45, 46]
RAGAT_CFG = {
    'Cora': dict(identity_init=True, extra_lr=None, lr=0.01, hidden=64, epochs=150, wd=1e-4),
    'CiteSeer': dict(identity_init=True, extra_lr=None, lr=0.005, hidden=64, epochs=150, wd=1e-4),
}


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1))


def main():
    with open(SRC, 'r', encoding='utf-8') as f:
        rec = json.load(f)
    raw = list(rec['raw'])
    have = {(r['dataset'], r['model'], int(r['seed'])) for r in raw}
    print('existing raw rows: %d' % len(raw), flush=True)

    for ds in DATASETS:
        print('=== %s extra seeds ===' % ds, flush=True)
        d = load_dataset(ds)
        for m in MODELS:
            for s in NEW_SEEDS:
                if (ds, m, s) in have:
                    print('  skip %s seed=%d (already present)' % (m, s), flush=True)
                    continue
                t0 = time.time()
                if m == 'RA-GAT':
                    r = run_once(d, m, seed=s, **RAGAT_CFG[ds])
                else:
                    r = run_once(d, m, seed=s)
                row = {k: v for k, v in r.items() if not k.startswith('_')}
                row['dataset'] = ds
                if m == 'RA-GAT':
                    row['identity_init'] = True
                    row['lr'] = RAGAT_CFG[ds]['lr']
                    row['wd'] = RAGAT_CFG[ds]['wd']
                print('  %s seed=%d AUC=%.4f AP=%.4f Hits=%.4f (%.1fs)' % (
                    m, s, row['auc'], row['ap'], row['hits20'], time.time() - t0), flush=True)
                raw.append(row)
                have.add((ds, m, s))

    table = {}
    for ds in DATASETS:
        table[ds] = {}
        for m in MODELS:
            rows = [r for r in raw if r['dataset'] == ds and r['model'] == m
                    and int(r['seed']) in ALL_SEEDS]
            rows = sorted(rows, key=lambda x: int(x['seed']))
            # keep one row per seed (last write wins)
            by_seed = {}
            for r in rows:
                by_seed[int(r['seed'])] = r
            rows = [by_seed[s] for s in ALL_SEEDS if s in by_seed]
            aucs = [r['auc'] for r in rows]
            aps = [r['ap'] for r in rows]
            hits = [r['hits20'] for r in rows]
            times = [r['train_time'] for r in rows]
            auc_m, auc_s = summarize(aucs)
            ap_m, ap_s = summarize(aps)
            h_m, h_s = summarize(hits)
            entry = dict(
                auc_mean=auc_m, auc_std=auc_s,
                ap_mean=ap_m, ap_std=ap_s,
                hits20_mean=h_m, hits20_std=h_s,
                train_time_mean=float(np.mean(times)),
                n_params=int(np.mean([r['n_params'] for r in rows])),
                n_seeds=len(rows),
                seeds=[int(r['seed']) for r in rows],
                std_type='sample',
            )
            if m == 'RA-GAT':
                entry['identity_init'] = True
                entry['lr'] = RAGAT_CFG[ds]['lr']
                entry['wd'] = RAGAT_CFG[ds]['wd']
            table[ds][m] = entry
            print('[%s] %-10s n=%d AUC %.4f±%.4f  AP %.4f±%.4f  Hits %.4f±%.4f' % (
                ds, m, len(rows), auc_m, auc_s, ap_m, ap_s, h_m, h_s), flush=True)

    rec['table'] = table
    rec['raw'] = raw
    rec['ragat_protocol'] = {
        'identity_init': True,
        'wd': 1e-4,
        'Cora_lr': 0.01,
        'CiteSeer_lr': 0.005,
        'seeds': ALL_SEEDS,
        'std': 'sample (ddof=1)',
        'selection': 'identity-init extra operators; wd=1e-4 selected by Cora per-seed validation AUC; extra seeds 45-46 reuse the frozen protocol',
    }
    with open(SRC, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2)
    print('wrote', SRC, flush=True)


if __name__ == '__main__':
    main()
