# Deprecated: not used for reported paper results. See rerun_fair_protocol.py
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

# Master experiment driver. Runs every analysis and writes figures/*.json.
# No plotting here (paper-figure step handles that). Numbers are all computed.
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json, time
import numpy as np
import torch
import torch.nn.functional as F
from data_gen import DATASETS, load_dataset
from train import run_once, edge_split, compute_region
from utils import link_metrics, hits_at_k

FIG = os.path.join(os.path.dirname(_HERE), 'figures')
os.makedirs(FIG, exist_ok=True)

MODELS = ['GCN', 'GraphSAGE', 'GAT', 'VGAE', 'RA-GAT']
SEEDS = [42, 43, 44]
DS = list(DATASETS.keys())


def save(name, obj):
    p = os.path.join(FIG, name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f'  wrote {name} ({os.path.getsize(p)} bytes)')


def clean(res):
    """Strip heavy torch objects before JSON serialization."""
    return {k: v for k, v in res.items() if not k.startswith('_')}


def dataset_stats():
    print('[1/8] dataset statistics')
    stats = {}
    for name in DS:
        d = load_dataset(name)
        ei = d['edge_index']
        row = ei[0]
        deg = torch.zeros(d['num_nodes']).index_add_(0, row, torch.ones(row.size(0)))
        n_edges = ei.size(1) // 2
        y = d['y'].numpy()
        # edge homophily on canonical edges
        mask = ei[0] < ei[1]
        u, v = ei[0][mask].numpy(), ei[1][mask].numpy()
        homoph = float(np.mean(y[u] == y[v]))
        stats[name] = dict(
            num_nodes=d['num_nodes'], num_edges=int(n_edges),
            num_feat=d['num_feat'], num_comm=d['num_comm'],
            avg_degree=float(deg.mean()), max_degree=int(deg.max()),
            median_degree=float(deg.median()),
            degree_gini=float(gini(deg.numpy())),
            edge_homophily=homoph,
            degree_hist=np.histogram(deg.numpy(), bins=30)[0].tolist(),
            degree_hist_edges=np.histogram(deg.numpy(), bins=30)[1].tolist(),
        )
    save('dataset_stats.json', stats)
    save('descriptive_stats.json', stats)
    return stats


def gini(a):
    a = np.sort(a.astype(float)); n = len(a)
    if a.sum() == 0: return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum() - (n + 1) * a.sum()) / (n * a.sum()))


def main_comparison():
    print('[2/8] main comparison (5 models x 3 datasets x 3 seeds)')
    table = {}
    raw = []
    for ds in DS:
        table[ds] = {}
        for m in MODELS:
            aucs, aps, hits, ntimes, nparams = [], [], [], [], []
            for s in SEEDS:
                d = load_dataset(ds, seed=42)  # fixed graph; seed varies init+split
                r = run_once(d, m, seed=s)
                aucs.append(r['auc']); aps.append(r['ap']); hits.append(r['hits20'])
                ntimes.append(r['train_time']); nparams.append(r['n_params'])
                raw.append(clean(r))
                print(f'    {ds:14} {m:10} seed={s} AUC={r["auc"]:.4f} AP={r["ap"]:.4f}')
            table[ds][m] = dict(
                auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                ap_mean=float(np.mean(aps)), ap_std=float(np.std(aps)),
                hits20_mean=float(np.mean(hits)), hits20_std=float(np.std(hits)),
                train_time_mean=float(np.mean(ntimes)), n_params=int(np.mean(nparams)))
    save('main_results.json', table)
    save('main_results_raw.json', raw)
    return table


def ablation_study():
    print('[3/8] ablation (RA-GAT components)')
    variants = {'full': None, 'no_gate': 'no_gate', 'no_temp': 'no_temp',
                'no_density': 'no_density'}
    out = {}
    for ds in DS:
        out[ds] = {}
        for vname, ab in variants.items():
            aucs, aps = [], []
            for s in SEEDS:
                d = load_dataset(ds, seed=42)
                r = run_once(d, 'RA-GAT', seed=s, ablation=ab)
                aucs.append(r['auc']); aps.append(r['ap'])
            out[ds][vname] = dict(auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                                  ap_mean=float(np.mean(aps)))
            print(f'    {ds:14} {vname:12} AUC={np.mean(aucs):.4f}')
    save('ablation_results.json', out)
    return out


if __name__ == '__main__':
    print('=== RA-GAT experiment suite ===')
    t0 = time.time()
    stats = dataset_stats()
    main_tab = main_comparison()
    abl = ablation_study()
    # remaining analyses appended by main_part2 import
    from main_part2 import run_part2
    run_part2(save, load_dataset, run_once, DS, SEEDS, MODELS, main_tab)
    print(f'=== done in {time.time()-t0:.1f}s ===')
