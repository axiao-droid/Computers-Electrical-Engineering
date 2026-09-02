# Fair-protocol re-run after exclusive negatives, identity init, and density removal.
# Shared search space for every encoder. One frozen (lr, wd) per dataset, selected
# on seed-42 validation AUC only. Test is scored once after the val checkpoint.
import os, sys, json, time, hashlib
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
from data_gen import load_dataset, DATASETS
from train import run_once, edge_split

ROOT = os.path.dirname(_HERE)
FIG = os.path.join(ROOT, 'figures')
CACHE = os.path.join(FIG, '_cache_v2')
os.makedirs(CACHE, exist_ok=True)

SYN_DS = list(DATASETS.keys())
PUB_DS = ['Cora', 'CiteSeer']
MODELS = ['GCN', 'VGAE', 'GraphSAGE', 'GAT', 'GATv2', 'RA-GAT']
SYN_SEEDS = [42, 43, 44]
PUB_SEEDS = [42, 43, 44, 45, 46]
LR_GRID = [0.005, 0.01]
WD_GRID = [1e-4, 5e-4]
SHARED_DEFAULT = dict(lr=0.01, wd=5e-4, hidden=64, epochs=150)
CHANNEL_VARS = ['ch_deg', 'ch_neigh', 'ch_feat', 'loo_deg', 'loo_neigh', 'loo_feat']
COMP_VARS = [None, 'no_gate', 'no_temp']


def _key(*parts):
    raw = '__'.join(str(p) for p in parts).replace('/', '-')
    return os.path.join(CACHE, raw + '.json')


def _load(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _dump(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def assert_exclusive_negatives(data, seed=42):
    split = edge_split(data['edge_index'], data['num_nodes'], seed=seed)

    def pairs(t):
        a = t.numpy().T
        out = set()
        for u, v in a:
            out.add((int(min(u, v)), int(max(u, v))))
        return out

    trn, vn, ten = pairs(split['train_neg']), pairs(split['val_neg']), pairs(split['test_neg'])
    trp, vp, tep = pairs(split['train_pos']), pairs(split['val_pos']), pairs(split['test_pos'])
    all_pos = trp | vp | tep
    assert not (trn & vn), 'train/val negative overlap'
    assert not (trn & ten), 'train/test negative overlap'
    assert not (vn & ten), 'val/test negative overlap'
    assert not (trn & all_pos), 'train neg hits a positive'
    assert not (vn & all_pos), 'val neg hits a positive'
    assert not (ten & all_pos), 'test neg hits a positive'
    return {
        'n_train_neg': len(trn), 'n_val_neg': len(vn), 'n_test_neg': len(ten),
        'overlap_train_val': 0, 'overlap_train_test': 0, 'overlap_val_test': 0,
    }


def cached_run(ds_name, data, model, seed, lr, wd, ablation=None, identity_init=True):
    tag = ablation if ablation else ('full' if model == 'RA-GAT' else 'na')
    path = _key('run', ds_name, model, 's%s' % seed, tag,
                'lr%s' % lr, 'wd%s' % wd, 'id%s' % int(identity_init))
    hit = _load(path)
    if hit is not None:
        return hit
    t0 = time.time()
    r = run_once(data, model, seed=seed, lr=lr, wd=wd, ablation=ablation,
                 identity_init=identity_init if model == 'RA-GAT' else False,
                 **{k: SHARED_DEFAULT[k] for k in ('hidden', 'epochs')})
    clean = {k: v for k, v in r.items() if not k.startswith('_')}
    clean.update(dataset=ds_name, lr=lr, wd=wd, ablation=ablation,
                 identity_init=bool(identity_init) if model == 'RA-GAT' else False,
                 wall=time.time() - t0)
    _dump(path, clean)
    print('    %s %s seed=%d ab=%s AUC=%.4f val=%.4f (%.1fs)' % (
        ds_name, model, seed, tag, clean['auc'], clean['best_val_auc'],
        clean['wall']), flush=True)
    return clean


def summarize(rows, sample=False):
    auc = np.array([r['auc'] for r in rows], dtype=float)
    ap = np.array([r['ap'] for r in rows], dtype=float)
    h = np.array([r['hits20'] for r in rows], dtype=float)
    ddof = 1 if sample and len(rows) > 1 else 0
    return dict(
        auc_mean=float(auc.mean()), auc_std=float(auc.std(ddof=ddof)),
        ap_mean=float(ap.mean()), ap_std=float(ap.std(ddof=ddof)),
        hits20_mean=float(h.mean()), hits20_std=float(h.std(ddof=ddof)),
        n_params=int(np.mean([r['n_params'] for r in rows])),
        train_time_mean=float(np.mean([r['train_time'] for r in rows])),
        n_seeds=len(rows),
        seeds=[int(r['seed']) for r in rows],
        std_type='sample' if sample else 'population',
        lr=rows[0]['lr'], wd=rows[0]['wd'],
        identity_init=bool(rows[0].get('identity_init', False)),
    )


def select_shared_config(ds_name, data):
    """One (lr, wd) per dataset: maximize mean seed-42 validation AUC across models."""
    path = _key('hpsel', ds_name)
    hit = _load(path)
    if hit is not None:
        print('  [hp] %s cached lr=%s wd=%s' % (ds_name, hit['lr'], hit['wd']), flush=True)
        return hit
    grid = []
    for lr in LR_GRID:
        for wd in WD_GRID:
            vals = []
            for m in MODELS:
                r = cached_run(ds_name, data, m, 42, lr, wd)
                vals.append(r['best_val_auc'])
            rec = dict(lr=lr, wd=wd, mean_val=float(np.mean(vals)),
                       per_model=dict(zip(MODELS, [float(v) for v in vals])))
            grid.append(rec)
            print('  [hp] %s lr=%s wd=%s mean_val=%.4f' % (
                ds_name, lr, wd, rec['mean_val']), flush=True)
    best = max(grid, key=lambda z: z['mean_val'])
    out = dict(dataset=ds_name, lr=best['lr'], wd=best['wd'],
               mean_val=best['mean_val'], grid=grid,
               rule=('shared (lr, wd) maximizing mean seed-42 validation AUC '
                     'across GCN, VGAE, GraphSAGE, GAT, GATv2, RA-GAT'))
    _dump(path, out)
    print('  [hp] %s SELECTED lr=%s wd=%s mean_val=%.4f' % (
        ds_name, out['lr'], out['wd'], out['mean_val']), flush=True)
    return out


def run_table(ds_name, data, models, seeds, lr, wd, sample=False):
    table, raw = {}, []
    for m in models:
        rows = [cached_run(ds_name, data, m, s, lr, wd) for s in seeds]
        table[m] = summarize(rows, sample=sample)
        raw.extend(rows)
    return table, raw


def paired_ci(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    tcrit = 2.776  # df=4, 95%
    if n == 3:
        tcrit = 4.303
    return dict(mean=mean, lo=mean - tcrit * se, hi=mean + tcrit * se, n=n)


def main():
    print('=== fair-protocol re-run ===', flush=True)
    t0 = time.time()
    audit = {}
    for ds in SYN_DS + PUB_DS:
        d = load_dataset(ds, seed=42) if ds in SYN_DS else load_dataset(ds)
        audit[ds] = assert_exclusive_negatives(d, seed=42)
        print('  exclusive-neg OK %s %s' % (ds, audit[ds]), flush=True)
    _dump(os.path.join(FIG, 'neg_exclusivity_audit.json'), audit)

    # --- synthetic: shared frozen defaults, exclusive negs, identity init ---
    syn_table, syn_raw = {}, []
    for ds in SYN_DS:
        print('=== synthetic %s ===' % ds, flush=True)
        d = load_dataset(ds, seed=42)
        tab, raw = run_table(ds, d, MODELS, SYN_SEEDS,
                             SHARED_DEFAULT['lr'], SHARED_DEFAULT['wd'], sample=False)
        syn_table[ds] = tab
        syn_raw.extend(raw)
    _dump(os.path.join(FIG, 'main_results.json'), syn_table)
    _dump(os.path.join(FIG, 'main_results_raw.json'), syn_raw)
    print('wrote main_results.json', flush=True)

    # --- synthetic component + channel ablations ---
    abl = {}
    for ds in SYN_DS:
        print('=== ablation %s ===' % ds, flush=True)
        d = load_dataset(ds, seed=42)
        abl[ds] = {}
        for ab in COMP_VARS + CHANNEL_VARS:
            rows = [cached_run(ds, d, 'RA-GAT', s, SHARED_DEFAULT['lr'],
                               SHARED_DEFAULT['wd'], ablation=ab) for s in SYN_SEEDS]
            name = 'full' if ab is None else ab
            abl[ds][name] = dict(
                auc_mean=float(np.mean([r['auc'] for r in rows])),
                auc_std=float(np.std([r['auc'] for r in rows])),
                ap_mean=float(np.mean([r['ap'] for r in rows])),
            )
            print('    %s %s AUC=%.4f' % (ds, name, abl[ds][name]['auc_mean']), flush=True)
    _dump(os.path.join(FIG, 'ablation_results.json'), abl)

    # --- public: shared HP search then 5-seed eval including GCN/VGAE ---
    pub_cfg, pub_table, pub_raw, pub_stats = {}, {}, [], {}
    from run_real import graph_stats
    for ds in PUB_DS:
        print('=== public %s ===' % ds, flush=True)
        d = load_dataset(ds)
        pub_stats[ds] = graph_stats(d)
        hp = select_shared_config(ds, d)
        pub_cfg[ds] = dict(lr=hp['lr'], wd=hp['wd'], mean_val=hp['mean_val'])
        tab, raw = run_table(ds, d, MODELS, PUB_SEEDS, hp['lr'], hp['wd'], sample=True)
        pub_table[ds] = tab
        pub_raw.extend(raw)
        # channel ablation on public Cora only, 3 seeds, frozen public config
        if ds == 'Cora':
            ch = {}
            for ab in CHANNEL_VARS + COMP_VARS:
                rows = [cached_run(ds, d, 'RA-GAT', s, hp['lr'], hp['wd'], ablation=ab)
                        for s in SYN_SEEDS]
                name = 'full' if ab is None else ab
                ch[name] = dict(
                    auc_mean=float(np.mean([r['auc'] for r in rows])),
                    auc_std=float(np.std([r['auc'] for r in rows], ddof=1)),
                    ap_mean=float(np.mean([r['ap'] for r in rows])),
                )
            _dump(os.path.join(FIG, 'public_channel_ablation.json'), {ds: ch})

    paired = {}
    for ds in PUB_DS:
        paired[ds] = {}
        by_m = {}
        for m in MODELS:
            by_m[m] = {int(r['seed']): r for r in pub_raw
                       if r['dataset'] == ds and r['model'] == m}
        seeds = PUB_SEEDS
        for other in ['GAT', 'GATv2', 'GCN', 'VGAE', 'GraphSAGE']:
            for metric, key in [('AUC', 'auc'), ('AP', 'ap'), ('Pooled Hits@20', 'hits20')]:
                a = [by_m['RA-GAT'][s][key] for s in seeds]
                b = [by_m[other][s][key] for s in seeds]
                paired[ds]['RA-GAT - %s / %s' % (other, metric)] = paired_ci(a, b)

    rec = dict(
        stats=pub_stats, table=pub_table, raw=pub_raw, hp=pub_cfg, paired=paired,
        protocol=dict(
            negatives='mutually exclusive train/val/test',
            identity_init='RA-GAT extra operators identity-initialized on every graph',
            density_offset='removed from the executed encoder',
            hp_rule=('one shared (lr, wd) per public dataset, selected by mean '
                     'seed-42 validation AUC across all six encoders; frozen for '
                     'seeds 42-46; test scored once after val checkpoint'),
            search_space={'lr': LR_GRID, 'wd': WD_GRID},
            synthetic_hp=SHARED_DEFAULT,
        ),
    )
    _dump(os.path.join(FIG, 'real_citation_results.json'), rec)
    print('wrote real_citation_results.json', flush=True)
    print('=== done in %.1fs ===' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
