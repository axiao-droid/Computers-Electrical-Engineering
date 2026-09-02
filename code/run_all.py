# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

# Resumable orchestrator: caches every (analysis-cell) result to figures/_cache/
# so an interrupted run continues instead of recomputing. Wraps the existing
# science code in main.py / main_part2.py without changing any of its logic.
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn.functional as F
from data_gen import DATASETS, load_dataset
from train import run_once
from utils import link_metrics, hits_at_k
import main as M
import main_part2 as P2

FIG = os.path.join(os.path.dirname(_HERE), 'figures')
CACHE = os.path.join(FIG, '_cache')
os.makedirs(CACHE, exist_ok=True)

MODELS = M.MODELS
SEEDS = M.SEEDS
DS = M.DS


def _cpath(key):
    return os.path.join(CACHE, key + '.json')


def cached_run(ds, model, seed, ablation=None):
    """run_once with disk cache keyed on (ds,model,seed,ablation). Returns
    the JSON-serializable (underscore-stripped) result dict."""
    tag = ablation if ablation else 'full' if model == 'RA-GAT' else 'na'
    key = f'run__{ds}__{model}__s{seed}__{tag}'
    p = _cpath(key)
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    d = load_dataset(ds, seed=42)
    r = run_once(d, model, seed=seed, ablation=ablation)
    clean = {k: v for k, v in r.items() if not k.startswith('_')}
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(clean, f)
    print(f'    cached {key} AUC={clean["auc"]:.4f}', flush=True)
    return clean


def main_comparison():
    print('[2/8] main comparison (5 models x 3 datasets x 3 seeds)', flush=True)
    table, raw = {}, []
    for ds in DS:
        table[ds] = {}
        for m in MODELS:
            aucs, aps, hits, ntimes, nparams = [], [], [], [], []
            for s in SEEDS:
                r = cached_run(ds, m, s)
                aucs.append(r['auc']); aps.append(r['ap']); hits.append(r['hits20'])
                ntimes.append(r['train_time']); nparams.append(r['n_params'])
                raw.append(r)
            table[ds][m] = dict(
                auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                ap_mean=float(np.mean(aps)), ap_std=float(np.std(aps)),
                hits20_mean=float(np.mean(hits)), hits20_std=float(np.std(hits)),
                train_time_mean=float(np.mean(ntimes)), n_params=int(np.mean(nparams)))
    M.save('main_results.json', table)
    M.save('main_results_raw.json', raw)
    return table


def ablation_study():
    print('[3/8] ablation (RA-GAT components)', flush=True)
    variants = {'full': None, 'no_gate': 'no_gate', 'no_temp': 'no_temp',
                'no_density': 'no_density'}
    out = {}
    for ds in DS:
        out[ds] = {}
        for vname, ab in variants.items():
            aucs, aps = [], []
            for s in SEEDS:
                r = cached_run(ds, 'RA-GAT', s, ablation=ab)
                aucs.append(r['auc']); aps.append(r['ap'])
            out[ds][vname] = dict(auc_mean=float(np.mean(aucs)),
                                  auc_std=float(np.std(aucs)),
                                  ap_mean=float(np.mean(aps)))
            print(f'    {ds:14} {vname:12} AUC={np.mean(aucs):.4f}', flush=True)
    M.save('ablation_results.json', out)
    return out


def sensitivity_cached():
    """Same grid as main_part2.sensitivity but cell-cached (validation AUC).

    Old `sens__*` cache files store only test AUC. This function reads
    `sensval__*` cells and never falls back to that test-only field.
    """
    out_json = os.path.join(FIG, 'sensitivity_results.json')
    if os.path.exists(out_json):
        try:
            rec = json.load(open(out_json, encoding='utf-8'))
            if rec.get('metric') == 'best_validation_auc':
                print('[4/8] sensitivity (cached validation AUC, skip)', flush=True)
                return
        except Exception:
            pass
        print('[4/8] sensitivity JSON is not validation AUC; rebuilding', flush=True)
    else:
        print('[4/8] sensitivity sweep', flush=True)
    hidden_grid = [16, 32, 64, 96, 128]
    lr_grid = [0.001, 0.005, 0.01, 0.02, 0.05]
    surface = np.zeros((len(hidden_grid), len(lr_grid)))
    for a, hd in enumerate(hidden_grid):
        for b, lr in enumerate(lr_grid):
            key = 'sensval__h%s__lr%s' % (hd, lr)
            p = _cpath(key)
            if os.path.exists(p):
                rec = json.load(open(p, encoding='utf-8'))
                if 'val_auc' not in rec:
                    raise RuntimeError('cache %s lacks val_auc' % key)
                auc = float(rec['val_auc'])
            else:
                d = load_dataset('Cora-syn', seed=42)
                r = run_once(d, 'RA-GAT', hidden=hd, lr=lr, seed=42, epochs=120)
                auc = float(r['best_val_auc'])
                json.dump({
                    'val_auc': auc,
                    'test_auc': float(r['auc']),
                    'hidden': hd,
                    'lr': lr,
                    'seed': 42,
                    'epochs': 120,
                    'metric': 'best_validation_auc',
                }, open(p, 'w', encoding='utf-8'))
            surface[a, b] = auc
        print(f'    hidden={hd} done', flush=True)
    M.save('sensitivity_results.json', dict(
        hidden_grid=hidden_grid, lr_grid=lr_grid, auc_surface=surface.tolist(),
        metric='best_validation_auc', dataset='Cora-syn', seed=42, epochs=120,
        note=('Post-hoc exploratory surface. Cells report the highest '
              'validation AUC observed during a 120-epoch run. The sweep '
              'was not used for model selection or main conclusions.')))


def _stage(fn_json, fn, *args):
    """Run a part2 stage only if its output JSON is missing (stage-level resume)."""
    p = os.path.join(FIG, fn_json)
    if os.path.exists(p):
        print(f'    skip (cached): {fn_json}', flush=True); return
    fn(*args)


if __name__ == '__main__':
    print('=== RA-GAT experiment suite (resumable) ===', flush=True)
    t0 = time.time()
    if not os.path.exists(os.path.join(FIG, 'dataset_stats.json')):
        M.dataset_stats()
    else:
        print('[1/8] dataset statistics (cached, skip)', flush=True)
    main_tab = main_comparison()
    ablation_study()
    sensitivity_cached()
    _stage('attention_results.json', P2.attention_by_region, M.save, load_dataset, run_once)
    _stage('embedding_results.json', P2.embeddings, M.save, load_dataset, run_once)
    _stage('degree_gain_results.json', P2.degree_gain, M.save, load_dataset, run_once, SEEDS)
    _stage('training_curves_results.json', P2.training_curves, M.save, load_dataset, run_once)
    _stage('efficiency_results.json', P2.efficiency, M.save, main_tab, MODELS, DS)
    print(f'=== done in {time.time()-t0:.1f}s ===', flush=True)
