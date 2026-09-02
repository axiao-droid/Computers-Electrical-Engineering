# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

# GNN encoders implemented from scratch (no torch_geometric).
# All use edge-index message passing with sparse scatter ops for CPU efficiency.
# Encoders: GCN, GraphSAGE(mean), GAT, GATv2, VGAE(GCN-based), and RA-GAT (ours).
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import scatter_sum, scatter_softmax


def add_self_loops(edge_index, n):
    loops = torch.arange(n, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, loops], dim=1)


def gcn_norm(edge_index, n):
    """Symmetric normalization coefficients D^-1/2 A D^-1/2 (with self-loops)."""
    ei = add_self_loops(edge_index, n)
    row, col = ei[0], ei[1]
    deg = torch.zeros(n, device=edge_index.device).index_add_(
        0, row, torch.ones(row.size(0), device=edge_index.device))
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0
    w = dinv[row] * dinv[col]
    return ei, w


class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.lin = nn.Linear(i, o)

    def forward(self, x, edge_index):
        n = x.size(0)
        ei, w = gcn_norm(edge_index, n)
        x = self.lin(x)
        msg = x[ei[1]] * w.unsqueeze(1)
        return scatter_sum(msg, ei[0], n)


class SAGELayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.lin_self = nn.Linear(i, o)
        self.lin_neigh = nn.Linear(i, o)

    def forward(self, x, edge_index):
        n = x.size(0)
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(n, device=x.device).index_add_(
            0, row, torch.ones(row.size(0), device=x.device)).clamp(min=1)
        agg = scatter_sum(x[col], row, n) / deg.unsqueeze(1)
        return self.lin_self(x) + self.lin_neigh(agg)


class GATLayer(nn.Module):
    def __init__(self, i, o, heads=4, concat=True):
        super().__init__()
        self.heads, self.o, self.concat = heads, o, concat
        self.lin = nn.Linear(i, heads * o, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, o))
        self.att_dst = nn.Parameter(torch.empty(1, heads, o))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index):
        n = x.size(0)
        ei = add_self_loops(edge_index, n)
        row, col = ei[0], ei[1]  # row=target, col=source
        h = self.lin(x).view(n, self.heads, self.o)
        alpha = (h * self.att_dst).sum(-1)[row] + (h * self.att_src).sum(-1)[col]
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = scatter_softmax(alpha, row, n)  # [E, heads]
        msg = h[col] * alpha.unsqueeze(-1)
        out = torch.zeros(n, self.heads, self.o, device=x.device)
        out.index_add_(0, row, msg)
        if self.concat:
            return out.reshape(n, self.heads * self.o)
        return out.mean(1)


class GATv2Layer(nn.Module):
    """Dynamic attention (Brody et al.): LeakyReLU before the attention vector."""
    def __init__(self, i, o, heads=4, concat=True):
        super().__init__()
        self.heads, self.o, self.concat = heads, o, concat
        self.lin_l = nn.Linear(i, heads * o, bias=False)
        self.lin_r = nn.Linear(i, heads * o, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, o))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        n = x.size(0)
        ei = add_self_loops(edge_index, n)
        row, col = ei[0], ei[1]
        h_l = self.lin_l(x).view(n, self.heads, self.o)
        h_r = self.lin_r(x).view(n, self.heads, self.o)
        alpha = (F.leaky_relu(h_l[row] + h_r[col], 0.2) * self.att).sum(-1)
        alpha = scatter_softmax(alpha, row, n)
        msg = h_r[col] * alpha.unsqueeze(-1)
        out = torch.zeros(n, self.heads, self.o, device=x.device)
        out.index_add_(0, row, msg)
        if self.concat:
            return out.reshape(n, self.heads * self.o)
        return out.mean(1)


class RAGATLayer(nn.Module):
    """Region-Adaptive GAT layer (ours).
    Extends GAT with (a) a region gate g_i from local structural context r_i
    and (b) an adaptive per-node temperature tau_i modulating attention sharpness.
    The former additive density offset is omitted: it is constant over neighbours
    of a fixed target and therefore idle under softmax. Region descriptor r_i is
    supplied externally (log-degree, mean neighbour degree, mean cosine agreement)."""
    def __init__(self, i, o, heads=4, concat=True, r_dim=3):
        super().__init__()
        self.heads, self.o, self.concat = heads, o, concat
        self.lin = nn.Linear(i, heads * o, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, o))
        self.att_dst = nn.Parameter(torch.empty(1, heads, o))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.gate = nn.Sequential(nn.Linear(r_dim, heads), nn.Sigmoid())
        self.temp = nn.Linear(r_dim, heads)

    def identity_init_(self):
        """Initialize extra operators so the layer coincides with GAT at step 0.
        Gate uses 2*sigmoid, so zero logits give multiplier 1. Temperature
        softplus(bias)+0.5 equals 1."""
        import math
        lin = self.gate[0]
        nn.init.zeros_(lin.weight)
        nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.temp.weight)
        nn.init.constant_(self.temp.bias, math.log(math.expm1(0.5)))

    def forward(self, x, edge_index, region, use_gate=True, use_temp=True,
                identity_gate=False, **_kw):
        n = x.size(0)
        ei = add_self_loops(edge_index, n)
        row, col = ei[0], ei[1]
        h = self.lin(x).view(n, self.heads, self.o)
        raw = (h * self.att_dst).sum(-1)[row] + (h * self.att_src).sum(-1)[col]
        raw = F.leaky_relu(raw, 0.2)
        if use_temp:
            tau = F.softplus(self.temp(region)) + 0.5
            raw = raw * tau[row]
        alpha = scatter_softmax(raw, row, n)
        msg = h[col] * alpha.unsqueeze(-1)
        out = torch.zeros(n, self.heads, self.o, device=x.device)
        out.index_add_(0, row, msg)
        if use_gate:
            g = self.gate(region).unsqueeze(-1)
            if identity_gate:
                g = 2.0 * g
            out = out * g
        if self.concat:
            return out.reshape(n, self.heads * self.o)
        return out.mean(1)


# ---------------- Encoders (2-layer) ----------------
class GCNEncoder(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.c1, self.c2 = GCNLayer(i, h), GCNLayer(h, o)

    def forward(self, x, ei, **kw):
        x = F.relu(self.c1(x, ei))
        return self.c2(x, ei)


class SAGEEncoder(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.c1, self.c2 = SAGELayer(i, h), SAGELayer(h, o)

    def forward(self, x, ei, **kw):
        x = F.relu(self.c1(x, ei))
        return self.c2(x, ei)


class GATEncoder(nn.Module):
    def __init__(self, i, h, o, heads=4):
        super().__init__()
        self.c1 = GATLayer(i, h, heads=heads, concat=True)
        self.c2 = GATLayer(h * heads, o, heads=1, concat=False)

    def forward(self, x, ei, **kw):
        x = F.elu(self.c1(x, ei))
        return self.c2(x, ei)


class GATv2Encoder(nn.Module):
    def __init__(self, i, h, o, heads=4):
        super().__init__()
        self.c1 = GATv2Layer(i, h, heads=heads, concat=True)
        self.c2 = GATv2Layer(h * heads, o, heads=1, concat=False)

    def forward(self, x, ei, **kw):
        x = F.elu(self.c1(x, ei))
        return self.c2(x, ei)


class RAGATEncoder(nn.Module):
    def __init__(self, i, h, o, heads=4, r_dim=3,
                 use_gate=True, use_temp=True, use_density=False,
                 identity_init=True):
        super().__init__()
        self.c1 = RAGATLayer(i, h, heads=heads, concat=True, r_dim=r_dim)
        self.c2 = RAGATLayer(h * heads, o, heads=1, concat=False, r_dim=r_dim)
        self.flags = dict(use_gate=use_gate, use_temp=use_temp,
                          identity_gate=bool(identity_init))
        if identity_init:
            self.c1.identity_init_()
            self.c2.identity_init_()

    def extra_parameters(self):
        named = []
        for layer in (self.c1, self.c2):
            named.extend(list(layer.gate.parameters()))
            named.extend(list(layer.temp.parameters()))
        return named

    def backbone_parameters(self):
        extra_ids = {id(p) for p in self.extra_parameters()}
        return [p for p in self.parameters() if id(p) not in extra_ids]

    def forward(self, x, ei, region=None, **kw):
        x = F.elu(self.c1(x, ei, region, **self.flags))
        return self.c2(x, ei, region, **self.flags)


class InnerProductDecoder(nn.Module):
    def forward(self, z, edge_pairs):
        # edge_pairs: [2, P] -> logits
        return (z[edge_pairs[0]] * z[edge_pairs[1]]).sum(-1)


class VGAE(nn.Module):
    """Variational Graph Auto-Encoder (Kipf & Welling) with GCN encoder."""
    def __init__(self, i, h, o):
        super().__init__()
        self.shared = GCNLayer(i, h)
        self.mu = GCNLayer(h, o)
        self.logstd = GCNLayer(h, o)
        self.dec = InnerProductDecoder()

    def encode(self, x, ei, **kw):
        hid = F.relu(self.shared(x, ei))
        mu = self.mu(hid, ei)
        logstd = self.logstd(hid, ei).clamp(max=10)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(logstd)
        else:
            z = mu
        self._mu, self._logstd = mu, logstd
        return z

    def kl_loss(self):
        return -0.5 * torch.mean(torch.sum(
            1 + 2 * self._logstd - self._mu ** 2 - torch.exp(2 * self._logstd), dim=1))
