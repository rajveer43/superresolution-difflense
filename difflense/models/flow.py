"""Conditional flow matching (Lipman et al. 2023, the paper's eq. 3-4) and the stage-wise sampler.

    x_t = (1 - (1 - s) t) x0 + t x1,     u_t = x1 - (1 - s) x0,     x0 ~ N(0, 1),  s = sigma_min = 1e-4
    L   = E || v_theta(x_t, t | LR) - u_t ||^2

One SR stage (inference): split the parent cells, integrate dx/dt = v_theta from t = 0 to 1 for an
ensemble of x0 draws (paper section 3.3), map each member back to energies and average in energy space.
"""
import torch

from ..data.datasets import denormalise
from ..data.graphs import scatter_sum, split_parents


def cfm_loss(model, g, sigma_min=1e-4):
    x1 = g["x1"]
    t = torch.rand(g["B"], device=x1.device)
    tn = t[g["ev"]]
    x0 = torch.randn_like(x1)
    xt = (1 - (1 - sigma_min) * tn) * x0 + tn * x1
    ut = x1 - (1 - sigma_min) * x0
    v = model(xt, t, g)
    return ((v - ut) ** 2).mean()


@torch.no_grad()
def integrate(model, g, steps=25, solver="midpoint", generator=None):
    N, B = len(g["ev"]), g["B"]
    dev = g["c"].device
    x = torch.randn(N, device=dev, generator=generator)
    if solver == "dopri5":
        from torchdiffeq import odeint

        def f(t, y):
            return model(y, t.expand(B), g)
        return odeint(f, x, torch.tensor([0.0, 1.0], device=dev), method="dopri5", rtol=1e-4, atol=1e-4)[-1]
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt, device=dev)
        v = model(x, t, g)
        if solver == "euler":
            x = x + dt * v
        else:  # midpoint (2nd order)
            x = x + dt * model(x + 0.5 * dt * v, t + 0.5 * dt, g)
    return x


def project_to_parents(E, E_raw, parent, E_par, n_parents):
    """Rescale children so they sum exactly to their parent (energy closure). E, E_raw, parent: per child;
    E_par: per parent. A parent whose children were all zero-suppressed gives its energy to the child with
    the largest raw prediction."""
    S = scatter_sum(E, parent, n_parents)
    Sp, Ep = S[parent], E_par[parent]
    out = torch.where(Sp > 0, E * Ep / Sp.clamp_min(1e-30), torch.zeros_like(E))
    empty = (S <= 0) & (E_par > 0)
    if empty.any():
        best = torch.full((n_parents,), -1.0, device=E.device, dtype=E_raw.dtype)
        best = best.scatter_reduce(0, parent, E_raw, reduce="amax")
        win = empty[parent] & (E_raw >= best[parent])
        idx = torch.arange(len(E), device=E.device)
        first = torch.full((n_parents,), len(E), device=E.device, dtype=torch.long)
        first = first.scatter_reduce(0, parent[win], idx[win], reduce="amin")  # one winner per parent
        out = torch.where(win & (first[parent] == idx), Ep, out)
    return out


@torch.no_grad()
def sr_stage(model, feat, parents, factor, noisy, B, cfg, ensemble, steps, solver, threshold, project,
             generator=None):
    """parents: node dict with E (> 0). Returns children with E (ensemble mean), E_std (ensemble spread),
    E_members_sum (B, M, P) per-event per-plane totals of each member, for the ensemble studies."""
    K = int(factor[0]) * int(factor[1])
    ch = split_parents(parents, factor)
    E_par_child = parents["E"][ch["parent"]]
    ch = feat(ch, E_par_child, float(K))
    ch["flag"] = torch.full((B,), int(noisy), dtype=torch.long, device=ch["c"].device)
    ch["B"] = B
    acc = torch.zeros_like(ch["c"])
    acc2 = torch.zeros_like(ch["c"])
    P = feat.geom.P
    member_sums = []
    thr = threshold.to(ch["c"].device)[ch["pl"]] * ch["area"]  # zero suppression scales with the cell area
    n_par = len(parents["E"])
    for _ in range(ensemble):
        x1 = integrate(model, ch, steps, solver, generator)
        E_raw = denormalise(x1 + ch["c"], ch["scale"], ch["area"])
        E = torch.where(E_raw > thr, E_raw, torch.zeros_like(E_raw))
        if project:
            E = project_to_parents(E, E_raw, ch["parent"], parents["E"], n_par)
        acc += E
        acc2 += E * E
        tot = torch.zeros(B * P, device=E.device).index_add_(0, ch["ev"] * P + ch["pl"], E)
        member_sums.append(tot.reshape(B, P))
    mean = acc / ensemble
    if project:  # the mean of projected members is itself projected; repeat for float round-off
        mean = project_to_parents(mean, mean, ch["parent"], parents["E"], n_par)
    ch["E"] = mean
    ch["E_std"] = (acc2 / ensemble - mean ** 2).clamp_min(0).sqrt()
    ch["E_members_sum"] = torch.stack(member_sums, 1)
    return ch
