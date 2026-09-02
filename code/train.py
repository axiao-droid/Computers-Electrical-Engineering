# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

# Link-prediction protocol: edge split, negative sampling, region features, train loop.
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import time
import numpy as np
import torch
import torch.nn.functional as F
from utils import set_seed, link_metrics, hits_at_k, count_params
from models import (GCNEncoder, SAGEEncoder, GATEncoder, GATv2Encoder,
                    RAGATEncoder, VGAE, InnerProductDecoder)


def edge_split(edge_index, num_nodes, val_frac=0.05, test_frac=0.10, seed=42):
    """Split undirected edges into train/val/test positive sets + sampled negatives.
    Message-passing graph uses only training positives (transductive LP)."""
    rng = np.random.default_rng(seed)
    ei = edge_index.numpy()
    # keep one direction (u<v) as canonical positive set
    mask = ei[0] < ei[1]
    pos = ei[:, mask].T  # [P,2]
    P = pos.shape[0]
    perm = rng.permutation(P)
    pos = pos[perm]
    n_val, n_test = int(P * val_frac), int(P * test_frac)
    test_pos = pos[:n_test]
    val_pos = pos[n_test:n_test + n_val]
    train_pos = pos[n_test + n_val:]

    # Negative sampling: reject self-pairs, all observed positives, and
    # negatives already assigned to another split so train/val/test
    # positives AND negatives are mutually exclusive.
    forbidden = set(map(tuple, pos.tolist()))
    def sample_neg(k):
        negs = set()
        while len(negs) < k:
            u = rng.integers(num_nodes); v = rng.integers(num_nodes)
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in forbidden or (a, b) in negs:
                continue
            negs.add((a, b))
        return np.array(list(negs), dtype=np.int64)

    train_neg = sample_neg(len(train_pos))
    forbidden |= set(map(tuple, train_neg.tolist()))
    val_neg = sample_neg(len(val_pos))
    forbidden |= set(map(tuple, val_neg.tolist()))
    test_neg = sample_neg(len(test_pos))

    # message-passing edge_index = train positives, both directions
    mp = np.concatenate([train_pos, train_pos[:, ::-1]], axis=0).T
    to_t = lambda a: torch.from_numpy(a.T.copy())  # [2,K]
    return {
        'mp_edge_index': torch.from_numpy(mp),
        'train_pos': to_t(train_pos), 'train_neg': to_t(train_neg),
        'val_pos': to_t(val_pos), 'val_neg': to_t(val_neg),
        'test_pos': to_t(test_pos), 'test_neg': to_t(test_neg),
    }


def compute_region(mp_edge_index, x, y, num_nodes, channel_mask=None):
    """Region descriptor r_i = [std log-degree, std mean neighbour degree,
    std mean cosine feature agreement]. Computed from the TRAINING
    message-passing graph only. Class / community labels y are unused:
    they are accepted only for call-site compatibility. Feature agreement
    is the mean cosine similarity of neighbour features, not class labels."""
    del y
    row, col = mp_edge_index[0], mp_edge_index[1]
    deg = torch.zeros(num_nodes).index_add_(0, row, torch.ones(row.size(0)))
    logdeg = torch.log1p(deg)
    neigh_deg_sum = torch.zeros(num_nodes).index_add_(0, row, deg[col])
    dens = neigh_deg_sum / deg.clamp(min=1)
    xn = F.normalize(x, dim=1)
    sim = (xn[row] * xn[col]).sum(1).clamp(-1, 1)
    homo_sum = torch.zeros(num_nodes).index_add_(0, row, sim)
    homo = homo_sum / deg.clamp(min=1)

    def std(t):
        return (t - t.mean()) / (t.std() + 1e-6)
    region = torch.stack([std(logdeg), std(dens), std(homo)], dim=1).float()
    if channel_mask is not None:
        mask = torch.as_tensor(channel_mask, dtype=region.dtype)
        region = region * mask.view(1, -1)
    return region, deg


# Descriptor-channel masks: 0=log-degree, 1=mean neighbour degree,
# 2=mean cosine feature agreement. Used only for channel ablations.
CHANNEL_MASKS = {
    'ch_deg':    [1, 0, 0],
    'ch_neigh':  [0, 1, 0],
    'ch_feat':   [0, 0, 1],
    'loo_deg':   [0, 1, 1],
    'loo_neigh': [1, 0, 1],
    'loo_feat':  [1, 1, 0],
}


def build_model(name, i, h, o, heads, ablation=None, identity_init=True):
    if name == 'GCN':      return GCNEncoder(i, h, o), 'enc'
    if name == 'GraphSAGE':return SAGEEncoder(i, h, o), 'enc'
    if name == 'GAT':      return GATEncoder(i, h, o, heads=heads), 'enc'
    if name == 'GATv2':    return GATv2Encoder(i, h, o, heads=heads), 'enc'
    if name == 'VGAE':     return VGAE(i, h, o), 'vgae'
    if name == 'RA-GAT':
        flags = dict(use_gate=True, use_temp=True, use_density=False,
                     identity_init=identity_init)
        if ablation == 'no_gate':
            flags['use_gate'] = False
        if ablation == 'no_temp':
            flags['use_temp'] = False
        return RAGATEncoder(i, h, o, heads=heads, **flags), 'ragat'
    raise ValueError(name)


def run_once(data, model_name, hidden=64, out=32, heads=4, epochs=150, lr=0.01,
             wd=5e-4, seed=42, ablation=None, record_curve=False,
             identity_init=True, extra_lr=None):
    set_seed(seed)
    i = data['num_feat']
    split = edge_split(data['edge_index'], data['num_nodes'], seed=seed)
    channel_mask = CHANNEL_MASKS.get(ablation)
    region, deg = compute_region(split['mp_edge_index'], data['x'], data['y'],
                                 data['num_nodes'], channel_mask=channel_mask)
    model, kind = build_model(model_name, i, hidden, out, heads, ablation,
                              identity_init=identity_init)
    dec = InnerProductDecoder()
    if kind == 'ragat' and extra_lr is not None:
        opt = torch.optim.Adam([
            {'params': model.backbone_parameters(), 'lr': lr},
            {'params': model.extra_parameters(), 'lr': extra_lr},
        ], weight_decay=wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    x, mp = data['x'], split['mp_edge_index']

    def encode():
        if kind == 'ragat':
            return model(x, mp, region=region)
        if kind == 'vgae':
            return model.encode(x, mp)
        return model(x, mp)

    curve = {'epoch': [], 'loss': [], 'val_loss': [], 'val_auc': []}
    best_val, best_state = -1, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z = encode()
        pos = dec(z, split['train_pos'])
        neg = dec(z, split['train_neg'])
        logits = torch.cat([pos, neg])
        target = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = F.binary_cross_entropy_with_logits(logits, target)
        if kind == 'vgae':
            loss = loss + (1.0 / data['num_nodes']) * model.kl_loss()
        loss.backward()
        opt.step()

        if ep % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                z = encode()
                vpos = dec(z, split['val_pos'])
                vneg = dec(z, split['val_neg'])
                va, _ = link_metrics(vpos, vneg)
                if record_curve:
                    vlogits = torch.cat([vpos, vneg])
                    vtarget = torch.cat([torch.ones_like(vpos), torch.zeros_like(vneg)])
                    vloss = float(F.binary_cross_entropy_with_logits(vlogits, vtarget))
            if va > best_val:
                best_val = va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if record_curve:
                curve['epoch'].append(ep)
                curve['loss'].append(float(loss))
                curve['val_loss'].append(vloss)
                curve['val_auc'].append(va)
    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = encode()
        auc, ap = link_metrics(dec(z, split['test_pos']), dec(z, split['test_neg']))
        h20 = hits_at_k(dec(z, split['test_pos']), dec(z, split['test_neg']), 20)

    res = dict(model=model_name, seed=seed, auc=auc, ap=ap, hits20=h20,
               n_params=count_params(model), train_time=train_time,
               best_val_auc=best_val)
    if record_curve:
        res['curve'] = curve
    # attach objects for downstream analysis (embeddings/region/attention)
    res['_z'] = z.detach()
    res['_split'] = split
    res['_region'] = region
    res['_deg'] = deg
    res['_model'] = model
    res['_dec'] = dec
    return res
