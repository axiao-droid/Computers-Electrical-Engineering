# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

# Synthetic citation-network generator.
# Design goals (see data_quality rules): realistic degree skew (hubs + long tail),
# community structure (topic clusters), and homophilous node features so that
# link prediction is non-trivial but learnable. NO real benchmark is bundled in
# this offline environment, so we simulate graphs that mimic Cora/CiteSeer/PubMed
# scale and heterogeneity. Seeded for reproducibility.
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch


def _preferential_attachment(n, m, rng):
    """Barabasi-Albert-style growth -> power-law degree (hub authors / seminal papers)."""
    edges = set()
    targets = list(range(m))
    repeated = list(range(m))
    for src in range(m, n):
        chosen = set()
        while len(chosen) < min(m, len(set(repeated))):
            chosen.add(repeated[rng.integers(len(repeated))])
        for t in chosen:
            edges.add((min(src, t), max(src, t)))
        repeated.extend(chosen)
        repeated.extend([src] * m)
    return edges


def generate_citation_graph(name, n_nodes, n_feat, n_comm, m_attach,
                            homophily=0.85, feat_noise=1.0, seed=42):
    """Return dict with edge_index, x (features), y (community labels), meta."""
    rng = np.random.default_rng(seed)

    # 1) community assignment (topic clusters), skewed sizes
    sizes = rng.dirichlet(np.ones(n_comm) * 2.0) * n_nodes
    sizes = np.maximum(sizes.astype(int), 10)
    diff = n_nodes - sizes.sum()
    sizes[0] += diff
    labels = np.concatenate([np.full(s, c) for c, s in enumerate(sizes)])
    rng.shuffle(labels)

    # 2) backbone edges via preferential attachment (degree skew)
    pa_edges = _preferential_attachment(n_nodes, m_attach, rng)

    # 3) rewire fraction to respect community homophily (citations mostly within topic)
    edges = set()
    for (u, v) in pa_edges:
        if rng.random() < homophily and labels[u] != labels[v]:
            # rewire v to a node in u's community
            same = np.where(labels == labels[u])[0]
            v = int(same[rng.integers(len(same))])
        if u != v:
            edges.add((min(u, v), max(u, v)))

    # 4) add a few extra intra-community edges to lift density realistically
    n_extra = int(len(edges) * 0.15)
    for _ in range(n_extra):
        c = rng.integers(n_comm)
        pool = np.where(labels == c)[0]
        if len(pool) < 2:
            continue
        u, v = rng.choice(pool, 2, replace=False)
        edges.add((int(min(u, v)), int(max(u, v))))

    edge_arr = np.array(sorted(edges), dtype=np.int64)
    # undirected -> both directions
    ei = np.concatenate([edge_arr, edge_arr[:, ::-1]], axis=0).T  # [2, 2E]

    # 5) homophilous features: community centroid + Gaussian noise (bag-of-topics style)
    centroids = rng.normal(0, 1.0, size=(n_comm, n_feat))
    x = centroids[labels] + rng.normal(0, feat_noise, size=(n_nodes, n_feat))
    x = x.astype(np.float32)

    return {
        'name': name,
        'edge_index': torch.from_numpy(ei),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(labels.astype(np.int64)),
        'num_nodes': n_nodes,
        'num_feat': n_feat,
        'num_comm': n_comm,
    }


# Three benchmark-scale synthetic datasets mirroring Cora/CiteSeer/PubMed heterogeneity.
DATASETS = {
    'Cora-syn':     dict(n_nodes=2708, n_feat=128, n_comm=7,  m_attach=2, homophily=0.83),
    'CiteSeer-syn': dict(n_nodes=3312, n_feat=128, n_comm=6,  m_attach=2, homophily=0.74),
    'PubMed-syn':   dict(n_nodes=4000, n_feat=128, n_comm=3,  m_attach=3, homophily=0.80),
}


def load_dataset(name, seed=42):
    if name in REAL_DATASETS:
        return REAL_DATASETS[name]()
    cfg = DATASETS[name]
    return generate_citation_graph(name, seed=seed, **cfg)


def _planetoid_dir(name):
    here = os.path.join(_HERE, 'data', name)
    parent = os.path.abspath(os.path.join(_HERE, '..', 'code', 'data', name))
    if os.path.isfile(os.path.join(here, name + '.content')):
        return here
    return parent


def load_planetoid(name):
    """Public Planetoid citation graph (Sen et al. / LINQS), undirected for this protocol."""
    root = _planetoid_dir(name.lower())
    content_path = os.path.join(root, name.lower() + '.content')
    cites_path = os.path.join(root, name.lower() + '.cites')
    ids, feats, labels = [], [], []
    class_to_i = {}
    with open(content_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            ids.append(parts[0])
            feats.append([float(x) for x in parts[1:-1]])
            lab = parts[-1]
            if lab not in class_to_i:
                class_to_i[lab] = len(class_to_i)
            labels.append(class_to_i[lab])
    id_to_i = {pid: i for i, pid in enumerate(ids)}
    x = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    n = x.shape[0]
    edges = set()
    with open(cites_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if parts[0] not in id_to_i or parts[1] not in id_to_i:
                continue
            u, v = id_to_i[parts[0]], id_to_i[parts[1]]
            if u == v:
                continue
            edges.add((min(u, v), max(u, v)))
    edge_arr = np.array(sorted(edges), dtype=np.int64)
    ei = np.concatenate([edge_arr, edge_arr[:, ::-1]], axis=0).T
    return {
        'name': name,
        'edge_index': torch.from_numpy(ei),
        'x': torch.from_numpy(x),
        'y': torch.from_numpy(y),
        'num_nodes': n,
        'num_feat': int(x.shape[1]),
        'num_comm': int(y.max()) + 1,
    }


def load_cora():
    return load_planetoid('Cora')


def load_citeseer():
    return load_planetoid('CiteSeer')


REAL_DATASETS = {
    'Cora': load_cora,
    'CiteSeer': load_citeseer,
}
