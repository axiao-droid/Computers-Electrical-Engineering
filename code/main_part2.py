# Deprecated: not used for reported paper results. See rerun_fair_protocol.py
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

# Second half of the experiment suite: analyses that need model internals.
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from utils import link_metrics


def sensitivity(save, load_dataset, run_once):
    """AUC over (hidden dim x learning rate) grid on Cora-syn -> response surface."""
    print('[4/8] sensitivity sweep (gate dim x temperature init proxy)')
    hidden_grid = [16, 32, 64, 96, 128]
    lr_grid = [0.001, 0.005, 0.01, 0.02, 0.05]
    surface = np.zeros((len(hidden_grid), len(lr_grid)))
    for a, hd in enumerate(hidden_grid):
        for b, lr in enumerate(lr_grid):
            d = load_dataset('Cora-syn', seed=42)
            r = run_once(d, 'RA-GAT', hidden=hd, lr=lr, seed=42, epochs=120)
            surface[a, b] = r['best_val_auc']
        print(f'    hidden={hd} done')
    save('sensitivity_results.json', dict(
        hidden_grid=hidden_grid, lr_grid=lr_grid, auc_surface=surface.tolist(),
        metric='best_validation_auc'))


def attention_by_region(save, load_dataset, run_once):
    """Extract layer-1 attention entropy stratified by node-degree bucket.
    Higher entropy = more uniform aggregation; lower = sharper focus."""
    print('[5/8] attention pattern by degree region')
    d = load_dataset('Cora-syn', seed=42)
    r = run_once(d, 'RA-GAT', seed=42)
    model, split, region, deg = r['_model'], r['_split'], r['_region'], r['_deg']
    from models import add_self_loops
    x, mp = d['x'], split['mp_edge_index']
    model.eval()
    with torch.no_grad():
        n = x.size(0)
        ei = add_self_loops(mp, n)
        row, col = ei[0], ei[1]
        h = model.c1.lin(x).view(n, model.c1.heads, model.c1.o)
        raw = (h * model.c1.att_dst).sum(-1)[row] + (h * model.c1.att_src).sum(-1)[col]
        raw = F.leaky_relu(raw, 0.2)
        tau = F.softplus(model.c1.temp(region)) + 0.5
        raw = raw * tau[row]
        from utils import scatter_softmax
        alpha = scatter_softmax(raw, row, n).mean(1)  # avg over heads [E]
        # entropy per target node
        ent = torch.zeros(n)
        contrib = -(alpha * torch.log(alpha + 1e-12))
        ent.index_add_(0, row, contrib)
        gate = model.c1.gate(region).mean(1).numpy()   # avg gate per node
        tau_mean = tau.mean(1).numpy()

    degn = deg.numpy()
    buckets = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 10000)]
    labels = ['1-2', '3-4', '5-8', '9-16', '17+']
    rows = []
    for (lo, hi), lab in zip(buckets, labels):
        m = (degn >= lo) & (degn <= hi)
        if m.sum() == 0: continue
        rows.append(dict(bucket=lab, n_nodes=int(m.sum()),
                         mean_attn_entropy=float(ent.numpy()[m].mean()),
                         mean_gate=float(gate[m].mean()),
                         mean_temperature=float(tau_mean[m].mean())))
    save('attention_results.json', dict(strata=rows, heads=int(model.c1.heads)))


def embeddings(save, load_dataset, run_once):
    """t-SNE of learned embeddings; separability of linked vs non-linked pairs."""
    print('[6/8] embedding t-SNE + pair separability')
    d = load_dataset('Cora-syn', seed=42)
    r = run_once(d, 'RA-GAT', seed=42)
    z, split, dec = r['_z'], r['_split'], r['_dec']
    y = d['y'].numpy()
    # subsample nodes for t-SNE
    n = z.size(0)
    idx = np.random.default_rng(0).choice(n, size=min(800, n), replace=False)
    emb = TSNE(n_components=2, init='pca', perplexity=30, random_state=0
               ).fit_transform(z.numpy()[idx])
    # pair separability: score distribution positive vs negative test edges
    with torch.no_grad():
        pos = torch.sigmoid(dec(z, split['test_pos'])).numpy()
        neg = torch.sigmoid(dec(z, split['test_neg'])).numpy()
    save('embedding_results.json', dict(
        tsne_x=emb[:, 0].tolist(), tsne_y=emb[:, 1].tolist(),
        labels=y[idx].tolist(),
        pos_scores=pos.tolist(), neg_scores=neg.tolist(),
        pos_mean=float(pos.mean()), neg_mean=float(neg.mean())))


def degree_gain(save, load_dataset, run_once, SEEDS):
    """AUC gain of RA-GAT over GAT stratified by node-degree bucket (test edges)."""
    print('[7/8] degree-stratified gain vs GAT')
    d = load_dataset('Cora-syn', seed=42)
    buckets = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 10000)]
    labels = ['1-2', '3-4', '5-8', '9-16', '17+']

    def stratified_auc(model_name):
        per_bucket = {lab: [] for lab in labels}
        for s in SEEDS:
            r = run_once(d, model_name, seed=s)
            z, split, deg, dec = r['_z'], r['_split'], r['_deg'], r['_dec']
            degn = deg.numpy()
            with torch.no_grad():
                pos_s = torch.sigmoid(dec(z, split['test_pos'])).numpy()
                neg_s = torch.sigmoid(dec(z, split['test_neg'])).numpy()
            tp = split['test_pos'].numpy(); tn = split['test_neg'].numpy()
            # assign each edge to bucket by min endpoint degree (the sparse side)
            pdeg = np.minimum(degn[tp[0]], degn[tp[1]])
            ndeg = np.minimum(degn[tn[0]], degn[tn[1]])
            from sklearn.metrics import roc_auc_score
            for (lo, hi), lab in zip(buckets, labels):
                pm = (pdeg >= lo) & (pdeg <= hi)
                nm = (ndeg >= lo) & (ndeg <= hi)
                if pm.sum() < 3 or nm.sum() < 3: continue
                yv = np.concatenate([np.ones(pm.sum()), np.zeros(nm.sum())])
                sv = np.concatenate([pos_s[pm], neg_s[nm]])
                try:
                    per_bucket[lab].append(roc_auc_score(yv, sv))
                except Exception:
                    pass
        return per_bucket

    ra = stratified_auc('RA-GAT'); ga = stratified_auc('GAT')
    rows = []
    for lab in labels:
        if not ra[lab] or not ga[lab]: continue
        ra_m, ga_m = np.mean(ra[lab]), np.mean(ga[lab])
        rows.append(dict(bucket=lab, ragat_auc=float(ra_m), gat_auc=float(ga_m),
                         gain=float(ra_m - ga_m),
                         ragat_std=float(np.std(ra[lab])), gat_std=float(np.std(ga[lab]))))
        print(f'    deg {lab:6} RA-GAT={ra_m:.4f} GAT={ga_m:.4f} gain={ra_m-ga_m:+.4f}')
    save('degree_gain_results.json', dict(strata=rows))


def training_curves(save, load_dataset, run_once):
    """Loss + val-AUC curves for RA-GAT vs GAT vs GCN on Cora-syn."""
    print('[8/8] training curves')
    d = load_dataset('Cora-syn', seed=42)
    out = {}
    for m in ['GCN', 'GAT', 'RA-GAT']:
        r = run_once(d, m, seed=42, record_curve=True)
        out[m] = r['curve']
    save('training_curves_results.json', out)


def efficiency(save, main_tab, MODELS, DS):
    """Params / train-time / relative cost vs GAT from the main run."""
    print('[extra] efficiency table')
    ref_ds = DS[0]
    gat_p = main_tab[ref_ds]['GAT']['n_params']
    gat_t = main_tab[ref_ds]['GAT']['train_time_mean']
    rows = {}
    for m in MODELS:
        p = main_tab[ref_ds][m]['n_params']
        t = main_tab[ref_ds][m]['train_time_mean']
        rows[m] = dict(n_params=int(p), train_time_s=float(t),
                       params_vs_gat=float(p / gat_p),
                       time_vs_gat=float(t / gat_t))
    save('efficiency_results.json', rows)


def run_part2(save, load_dataset, run_once, DS, SEEDS, MODELS, main_tab):
    sensitivity(save, load_dataset, run_once)
    attention_by_region(save, load_dataset, run_once)
    embeddings(save, load_dataset, run_once)
    degree_gain(save, load_dataset, run_once, SEEDS)
    training_curves(save, load_dataset, run_once)
    efficiency(save, main_tab, MODELS, DS)
