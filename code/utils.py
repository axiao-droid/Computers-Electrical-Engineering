# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

# Shared utilities: reproducibility, sparse scatter ops, LP metrics.
# ⛔ 自举模块路径（让 sibling import 不依赖调用方式）
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scatter_sum(src, index, dim_size):
    """Sum src rows into buckets given by index. src:[E,H], index:[E] -> [dim_size,H]."""
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def scatter_softmax(scores, index, dim_size):
    """Softmax of scores grouped by target index (per-node attention normalization).
    scores:[E,H] logits, index:[E] target node of each edge -> normalized [E,H]."""
    # numerical stability: subtract per-group max
    max_per = torch.full((dim_size, scores.size(1)), float('-inf'), device=scores.device)
    max_per = max_per.index_reduce(0, index, scores, 'amax', include_self=True)
    max_gathered = max_per[index]
    max_gathered = torch.where(torch.isinf(max_gathered), torch.zeros_like(max_gathered), max_gathered)
    exp = torch.exp(scores - max_gathered)
    denom = scatter_sum(exp, index, dim_size)[index] + 1e-16
    return exp / denom


def link_metrics(pos_score, neg_score):
    """AUC + AP from positive/negative edge scores (1-D tensors)."""
    pos = pos_score.detach().cpu().numpy().ravel()
    neg = neg_score.detach().cpu().numpy().ravel()
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s = np.concatenate([pos, neg])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def hits_at_k(pos_score, neg_score, k):
    """Hits@K: fraction of positive edges ranked above the K-th largest negative score."""
    pos = pos_score.detach().cpu().numpy().ravel()
    neg = neg_score.detach().cpu().numpy().ravel()
    if len(neg) < k:
        thresh = np.min(neg)
    else:
        thresh = np.sort(neg)[-k]
    return float(np.mean(pos > thresh))


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
