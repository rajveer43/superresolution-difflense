"""Sparse cell graphs (torch). A batch of cells is a set of 1-D tensors of equal length:

    ev  event index in the batch      pl  plane
    iu, iv  cell indices on the grid  nu, nv  grid size (per node, so a batch may mix grids)

Only non-empty parents are split, so a graph has as many nodes as the shower has hits, not as the
image has pixels. That is what lets the ladder reach 2048 x 2048.
Neighbours are found by hashing (ev, pl, iu, iv) into one int64 key and binary search, O(N W log N),
without ever building a dense grid.
"""
import torch

KEY_U = 1 << 13  # grids up to 8192 cells per axis


def dense_to_nodes(img, min_value=0.0):
    """(B, P, U, V) -> dict of the cells with value > min_value."""
    B, P, U, V = img.shape
    ev, pl, iu, iv = (img > min_value).nonzero(as_tuple=True)
    n = len(ev)
    return {"ev": ev, "pl": pl, "iu": iu, "iv": iv, "E": img[ev, pl, iu, iv],
            "nu": torch.full((n,), U, dtype=torch.long, device=img.device),
            "nv": torch.full((n,), V, dtype=torch.long, device=img.device)}


def nodes_to_dense(nodes, B, P, shape, value="E"):
    out = torch.zeros(B, P, *shape, dtype=nodes[value].dtype, device=nodes[value].device)
    out.index_put_((nodes["ev"], nodes["pl"], nodes["iu"], nodes["iv"]), nodes[value], accumulate=True)
    return out


def split_parents(parents, factor):
    """Every parent -> fu*fv children. Returns children with
    parent (index into parents), du/dv (child centre relative to the parent centre, in parent widths)."""
    fu, fv = int(factor[0]), int(factor[1])
    K = fu * fv
    dev = parents["ev"].device
    a = torch.arange(fu, device=dev).repeat_interleave(fv)
    b = torch.arange(fv, device=dev).repeat(fu)
    n = len(parents["ev"])
    rep = lambda t: t.repeat_interleave(K)
    ch = {"ev": rep(parents["ev"]), "pl": rep(parents["pl"]),
          "iu": rep(parents["iu"]) * fu + a.repeat(n), "iv": rep(parents["iv"]) * fv + b.repeat(n),
          "nu": rep(parents["nu"]) * fu, "nv": rep(parents["nv"]) * fv,
          "parent": torch.arange(n, device=dev).repeat_interleave(K),
          "du": ((a.float() + 0.5) / fu - 0.5).repeat(n), "dv": ((b.float() + 0.5) / fv - 0.5).repeat(n)}
    return ch


def neighbour_offsets(window, plane_offsets):
    """(W, 3) integer offsets (du, dv, dp): the in-plane (2w+1)^2 window, then the same cell in other planes.
    Row 0 is the node itself, so every node has at least one valid neighbour."""
    offs = [(0, 0, 0)]
    for du in range(-window, window + 1):
        for dv in range(-window, window + 1):
            if (du, dv) != (0, 0):
                offs.append((du, dv, 0))
    offs += [(0, 0, int(p)) for p in plane_offsets if p != 0]
    return torch.tensor(offs, dtype=torch.long)


def _keys(ev, pl, iu, iv, P):
    return ((ev * P + pl) * KEY_U + iu) * KEY_U + iv


def neighbour_index(nodes, offsets, P, circular_u=False):
    """(N, W) index of each node's neighbour at each offset, -1 where there is no such cell.
    Nodes of different grids never match (their keys differ via nu)."""
    ev, pl, iu, iv, nu, nv = (nodes[k] for k in ("ev", "pl", "iu", "iv", "nu", "nv"))
    N = len(ev)
    if N == 0:
        return torch.zeros(0, len(offsets), dtype=torch.long, device=ev.device)
    grid_id = nu * KEY_U + nv  # separate grids inside a batch through the event slot
    _, gid = torch.unique(grid_id, return_inverse=True)
    slot = ev * (int(gid.max()) + 1) + gid
    keys = _keys(slot, pl, iu, iv, P)
    sk, order = keys.sort()
    offsets = offsets.to(ev.device)
    du, dv, dp = offsets[:, 0], offsets[:, 1], offsets[:, 2]
    u = iu[:, None] + du[None]
    v = iv[:, None] + dv[None]
    p = pl[:, None] + dp[None]
    if circular_u:
        u = torch.remainder(u, nu[:, None])
    ok = (u >= 0) & (u < nu[:, None]) & (v >= 0) & (v < nv[:, None]) & (p >= 0) & (p < P)
    q = _keys(slot[:, None], p.clamp(0, P - 1), u.clamp(min=0), v.clamp(min=0), P)
    pos = torch.searchsorted(sk, q).clamp(max=N - 1)
    found = ok & (sk[pos] == q)
    return torch.where(found, order[pos], torch.full_like(pos, -1))


def scatter_sum(src, index, n):
    out = torch.zeros((n,) + src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


def scatter_mean(src, index, n):
    s = scatter_sum(src, index, n)
    c = torch.bincount(index, minlength=n).clamp_min(1).to(src.dtype)
    return s / c.reshape((-1,) + (1,) * (src.dim() - 1))


def select(nodes, mask):
    return {k: v[mask] for k, v in nodes.items()}


def cat_nodes(parts):
    keys = parts[0].keys()
    return {k: torch.cat([p[k] for p in parts]) for k in keys}
