# Retrain RA-GAT on public Cora and CiteSeer with identity-init (3 seeds).
# Configs selected by seed-42 validation AUC, not test AUC.
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from data_gen import load_dataset
from train import run_once

FIG = os.path.join(os.path.dirname(_HERE), 'figures')
SEEDS = [42, 43, 44]
# seed-42 val-AUC winners
CFGS = {
    'Cora': dict(identity_init=True, extra_lr=None, lr=0.01, hidden=64, epochs=150),
    'CiteSeer': dict(identity_init=True, extra_lr=None, lr=0.005, hidden=64, epochs=150),
}


def main():
    src = os.path.join(FIG, 'real_citation_results.json')
    with open(src, 'r', encoding='utf-8') as f:
        rec = json.load(f)
    table = rec['table']
    raw = [r for r in rec['raw'] if r.get('model') != 'RA-GAT']
    for ds, cfg in CFGS.items():
        print('=== retrain RA-GAT %s ===' % ds, flush=True)
        d = load_dataset(ds)
        aucs, aps, hits, ntimes, nparams = [], [], [], [], []
        for s in SEEDS:
            t0 = time.time()
            r = run_once(d, 'RA-GAT', seed=s, **cfg)
            print('  seed=%d AUC=%.4f AP=%.4f Hits=%.4f val=%.4f (%.1fs)' % (
                s, r['auc'], r['ap'], r['hits20'], r['best_val_auc'], time.time() - t0),
                flush=True)
            aucs.append(r['auc']); aps.append(r['ap']); hits.append(r['hits20'])
            ntimes.append(r['train_time']); nparams.append(r['n_params'])
            raw.append({k: v for k, v in r.items() if not k.startswith('_')})
            raw[-1]['dataset'] = ds
            raw[-1]['identity_init'] = True
            raw[-1]['lr'] = cfg['lr']
        table[ds]['RA-GAT'] = dict(
            auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
            ap_mean=float(np.mean(aps)), ap_std=float(np.std(aps)),
            hits20_mean=float(np.mean(hits)), hits20_std=float(np.std(hits)),
            train_time_mean=float(np.mean(ntimes)), n_params=int(np.mean(nparams)),
            identity_init=True, lr=cfg['lr'])
        print('  -> mean AUC=%.4f AP=%.4f Hits=%.4f' % (
            table[ds]['RA-GAT']['auc_mean'], table[ds]['RA-GAT']['ap_mean'],
            table[ds]['RA-GAT']['hits20_mean']), flush=True)
    rec['table'] = table
    rec['raw'] = raw
    rec['ragat_protocol'] = {
        'identity_init': True,
        'selection': 'seed-42 validation AUC',
        'Cora_lr': CFGS['Cora']['lr'],
        'CiteSeer_lr': CFGS['CiteSeer']['lr'],
    }
    with open(src, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2)
    print('wrote', src, flush=True)
    for ds in CFGS:
        print('\n[%s]' % ds)
        for m in ['GraphSAGE', 'GAT', 'GATv2', 'RA-GAT']:
            t = table[ds][m]
            print('  %-10s AUC %.3f±%.3f  AP %.3f±%.3f  Hits %.3f±%.3f' % (
                m, t['auc_mean'], t['auc_std'], t['ap_mean'], t['ap_std'],
                t['hits20_mean'], t['hits20_std']))


if __name__ == '__main__':
    main()
