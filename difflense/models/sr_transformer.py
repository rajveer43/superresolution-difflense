"""Graph super-resolution transformer that predicts the flow-matching vector field v_theta(x_t, t | LR).

Follows the paper (figure 3): every LR cell is split into fu x fv HR nodes, each node carries the copied
LR energy (condition) stacked with its current flow state x_t, a transformer gives a neighbourhood-aware
representation, and an MLP head predicts v_t per node.

Changes for scale, all standard in diffusion transformers:
* attention over a local neighbourhood (the (2w+1)^2 cells around a node plus the same cell in the other
  planes) with a learned bias per relative offset, plus a per-event mean-pooled global context. The paper
  uses full attention on small graphs and notes that limited connectivity is the way to bigger ones; a
  2048 x 2048 graph has ~1e5-1e6 nodes, out of reach for full attention.
* adaLN-zero conditioning on t and on whether the parents are noisy measurements (DiT).
* QK-normalisation for stable attention logits at lr 1e-3.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.graphs import scatter_mean


def timestep_embedding(t, dim, max_period=1000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LocalAttention(nn.Module):
    def __init__(self, dim, heads, n_offsets, dropout=0.0, chunk=65536):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(self.dh)
        self.k_norm = nn.LayerNorm(self.dh)
        self.bias = nn.Parameter(torch.zeros(n_offsets, heads))
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.chunk = chunk

    def forward(self, x, nbr):
        N = x.shape[0]
        q, k, v = self.qkv(x).view(N, 3, self.h, self.dh).unbind(1)
        q, k = self.q_norm(q), self.k_norm(k)
        out = torch.empty_like(q)
        chunk = N if self.training else self.chunk
        for s in range(0, N, chunk):
            nb = nbr[s:s + chunk]
            valid = nb >= 0
            idx = nb.clamp_min(0)
            kn, vn = k[idx], v[idx]                                  # (n, W, h, dh)
            logits = torch.einsum("nhd,nwhd->nwh", q[s:s + chunk], kn).float() / math.sqrt(self.dh)
            logits = logits + self.bias[None].float()
            logits = logits.masked_fill(~valid[..., None], float("-inf"))
            att = self.drop(torch.softmax(logits, dim=1)).to(vn.dtype)
            out[s:s + chunk] = torch.einsum("nwh,nwhd->nhd", att, vn)
        return self.proj(out.reshape(N, -1))


class Block(nn.Module):
    def __init__(self, dim, heads, n_offsets, mlp_ratio=4.0, dropout=0.0, chunk=65536):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = LocalAttention(dim, heads, n_offsets, dropout, chunk)
        self.ng = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.glob = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
                                 nn.Linear(hid, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 8 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond, nbr, ev, B):
        sa, ca, ga, sg, gg, sm, cm, gm = self.ada(cond).chunk(8, -1)
        x = x + ga * self.attn(modulate(self.n1(x), sa, ca), nbr)
        g = scatter_mean(self.ng(x) + sg, ev, B)                      # per-event context
        x = x + gg * self.glob(g)[ev]
        x = x + gm * self.mlp(modulate(self.n2(x), sm, cm))
        return x


class GraphSRTransformer(nn.Module):
    def __init__(self, n_planes, pos_dim, n_offsets, dim=256, depth=4, heads=8, mlp_ratio=4.0, dropout=0.0,
                 attn_chunk=65536):
        super().__init__()
        n_in = 1 + 1 + 2 + pos_dim                                    # x_t, c, (du, dv), position
        self.inp = nn.Sequential(nn.Linear(n_in, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.plane = nn.Embedding(n_planes, dim)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.flag = nn.Embedding(2, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, n_offsets, mlp_ratio, dropout, attn_chunk)
                                    for _ in range(depth))
        self.n_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.dim = dim

    def forward(self, x_t, t, g):
        """x_t: (N,) flow state per node; t: (B,) time per event; g: graph batch (see data.datasets)."""
        ev, B = g["ev"], g["B"]
        feats = torch.cat([x_t[:, None], g["c"][:, None], g["du"][:, None], g["dv"][:, None], g["pos"]], -1)
        h = self.inp(feats.to(self.inp[0].weight.dtype)) + self.plane(g["pl"])
        cond_ev = self.t_mlp(timestep_embedding(t, self.dim)) + self.flag(g["flag"])
        cond = cond_ev[ev]
        for blk in self.blocks:
            h = blk(h, cond, g["nbr"], ev, B)
        shift, scale = self.ada_out(cond).chunk(2, -1)
        return self.head(modulate(self.n_out(h), shift, scale)).squeeze(-1).float()


def build_sr_model(cfg, geom, n_offsets):
    m = cfg["sr_model"]
    return GraphSRTransformer(geom.P, geom.pos_dim(), n_offsets, m["dim"], m["depth"], m["heads"],
                              m["mlp_ratio"], m["dropout"], m["attn_chunk"])
