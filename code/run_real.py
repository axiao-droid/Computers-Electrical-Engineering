# Train GraphSAGE / GAT / GATv2 / RA-GAT on public Cora and CiteSeer.
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
from data_gen import load_dataset, REAL_DATASETS
from train import run_once

FIG = os.path.join(os.path.dirname(_HERE), 'figures')
os.makedirs(FIG, exist_ok=True)

MODELS = ['GraphSAGE', 'GAT', 'GATv2', 'RA-GAT']
SEEDS = [42, 43, 44]
DATASETS = ['Cora', 'CiteSeer']


def gini(a):
    a = np.sort(a.astype(float)); n = len(a)
    if a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum() - (n + 1) * a.sum()) / (n * a.sum()))


def graph_stats(d):
    ei = d['edge_index']
    row = ei[0]
    deg = torch.zeros(d['num_nodes']).index_add_(0, row, torch.ones(row.size(0)))
    n_edges = ei.size(1) // 2
    y = d['y'].numpy()
    mask = ei[0] < ei[1]
    u, v = ei[0][mask].numpy(), ei[1][mask].numpy()
    homoph = float(np.mean(y[u] == y[v]))
    return dict(
        num_nodes=d['num_nodes'], num_edges=int(n_edges),
        num_feat=d['num_feat'], num_comm=d['num_comm'],
        avg_degree=float(deg.mean()), max_degree=int(deg.max()),
        median_degree=float(deg.median()),
        degree_gini=gini(deg.numpy()),
        edge_homophily=homoph,
    )


def main():
    stats, table, raw = {}, {}, []
    for ds in DATASETS:
        print('=== %s ===' % ds, flush=True)
        d = load_dataset(ds)
        stats[ds] = graph_stats(d)
        print('  nodes=%d edges=%d feat=%d classes=%d mean_deg=%.2f' % (
            stats[ds]['num_nodes'], stats[ds]['num_edges'],
            stats[ds]['num_feat'], stats[ds]['num_comm'],
            stats[ds]['avg_degree']), flush=True)
        table[ds] = {}
        for m in MODELS:
            aucs, aps, hits, ntimes, nparams = [], [], [], [], []
            for s in SEEDS:
                t0 = time.time()
                r = run_once(d, m, seed=s, identity_init=(m == 'RA-GAT'))
                print('  %s seed=%d AUC=%.4f AP=%.4f Hits@20=%.4f (%.1fs)' % (
                    m, s, r['auc'], r['ap'], r['hits20'], time.time() - t0), flush=True)
                aucs.append(r['auc']); aps.append(r['ap']); hits.append(r['hits20'])
                ntimes.append(r['train_time']); nparams.append(r['n_params'])
                raw.append({k: v for k, v in r.items() if not k.startswith('_')})
                raw[-1]['dataset'] = ds
            table[ds][m] = dict(
                auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                ap_mean=float(np.mean(aps)), ap_std=float(np.std(aps)),
                hits20_mean=float(np.mean(hits)), hits20_std=float(np.std(hits)),
                train_time_mean=float(np.mean(ntimes)), n_params=int(np.mean(nparams)))
            print('  -> %s mean AUC=%.4f AP=%.4f Hits=%.4f' % (
                m, table[ds][m]['auc_mean'], table[ds][m]['ap_mean'],
                table[ds][m]['hits20_mean']), flush=True)
    out = {'stats': stats, 'table': table, 'raw': raw}
    path = os.path.join(FIG, 'real_citation_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('wrote', path, flush=True)
    for ds in DATASETS:
        print('\n[%s]' % ds)
        for m in MODELS:
            t = table[ds][m]
            print('  %-10s AUC %.3f±%.3f  AP %.3f±%.3f  Hits %.3f±%.3f' % (
                m, t['auc_mean'], t['auc_std'], t['ap_mean'], t['ap_std'],
                t['hits20_mean'], t['hits20_std']))


if __name__ == '__main__':
    main()
