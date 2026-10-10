import numpy as np
import pytest
import torch

from difflense.config import load_config
from difflense.data.graphs import (dense_to_nodes, neighbour_index, neighbour_offsets, nodes_to_dense,
                                   split_parents)
from difflense.geometry import Geometry, grid_ladder, train_transitions
from difflense.models.flow import project_to_parents
from difflense.physics.observables import dense_to_cells, observables
from difflense.physics.ops import measure_lr, pool_to, sum_pool, uniform_upsample
from difflense.sr.progressive import refine_select, render


@pytest.fixture
def qg():
    return load_config("configs/smoke_qg.yaml")


@pytest.fixture
def calo():
    return load_config("configs/smoke_calo.yaml")


def test_sum_pool_conserves_energy():
    x = np.random.rand(2, 3, 16, 12).astype(np.float32)
    for f in [(2, 2), (4, 3), (16, 12)]:
        assert np.allclose(sum_pool(x, f).sum((2, 3)), x.sum((2, 3)), rtol=1e-5)
        assert np.allclose(uniform_upsample(sum_pool(x, f), f).sum((2, 3)), x.sum((2, 3)), rtol=1e-5)


def test_measure_lr_noise_free_is_pooling(qg):
    hr = torch.rand(2, 3, 32, 32)
    lr = measure_lr(hr, (4, 4), np.zeros(3, np.float32), qg["noise"]["topo"])
    assert torch.allclose(lr, sum_pool(hr, (4, 4)))
    noisy = measure_lr(hr, (4, 4), np.full(3, 0.5, np.float32), qg["noise"]["topo"])
    assert (noisy >= 0).all()


def test_split_and_dense_roundtrip():
    img = torch.zeros(2, 3, 4, 4)
    img[0, 1, 2, 3] = 5.0
    img[1, 0, 0, 0] = 2.0
    nodes = dense_to_nodes(img)
    assert torch.equal(nodes_to_dense(nodes, 2, 3, (4, 4)), img)
    ch = split_parents(nodes, (2, 3))
    assert len(ch["ev"]) == 2 * 6
    ch["E"] = nodes["E"][ch["parent"]] / 6
    fine = nodes_to_dense(ch, 2, 3, (8, 12))
    assert torch.allclose(sum_pool(fine, (2, 3)), img)
    assert torch.allclose(ch["du"].abs().max(), torch.tensor(0.25))


def test_neighbours_and_periodic_axis():
    img = torch.zeros(1, 2, 4, 3)
    img[0, 0, 0, 1] = img[0, 0, 3, 1] = img[0, 1, 0, 1] = 1.0
    nodes = dense_to_nodes(img)
    offs = neighbour_offsets(1, [-1, 1])
    nb_flat = neighbour_index(nodes, offs, 2, circular_u=False)
    nb_circ = neighbour_index(nodes, offs, 2, circular_u=True)
    i0 = int(((nodes["pl"] == 0) & (nodes["iu"] == 0)).nonzero())
    i3 = int(((nodes["pl"] == 0) & (nodes["iu"] == 3)).nonzero())
    up = int(((offs[:, 0] == -1) & (offs[:, 1] == 0) & (offs[:, 2] == 0)).nonzero())
    assert nb_flat[i0, up] == -1 and nb_circ[i0, up] == i3     # u = -1 wraps to u = 3
    assert (nb_circ[:, 0] == torch.arange(len(nodes["ev"]))).all()  # offset 0 is the node itself
    plane = int(((offs[:, 2] == 1)).nonzero())
    assert nodes["pl"][nb_circ[i0, plane]] == 1                 # same cell, next layer


def test_projection_closure():
    parent = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    E_par = torch.tensor([4.0, 2.0])
    E = torch.tensor([1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    E_raw = torch.tensor([1.0, 0.1, 3.0, 1.0, 0.01, 0.3, 0.02, 0.0])
    out = project_to_parents(E, E_raw, parent, E_par, 2)
    assert torch.allclose(torch.zeros(2).index_add_(0, parent, out), E_par)
    assert out[5] == 2.0 and out[4:].sum() == 2.0                 # empty parent -> its largest raw child


def test_refine_and_render_keep_energy():
    img = torch.rand(2, 1, 8, 8) * (torch.rand(2, 1, 8, 8) > 0.5)
    nodes = dense_to_nodes(img)
    act, leaves = refine_select(nodes, 5, 0.0, torch.ones(1), 64.0)
    assert (torch.bincount(act["ev"], minlength=2) <= 5).all()
    ch = split_parents(act, (2, 2))
    ch["E"] = act["E"][ch["parent"]] / 4
    full = render([leaves, {k: ch[k] for k in ("ev", "pl", "iu", "iv", "nu", "nv", "E")}], 2, 1, (16, 16))
    assert torch.allclose(full.sum((1, 2, 3)), img.sum((1, 2, 3)), rtol=1e-5)
    assert torch.allclose(sum_pool(full, (2, 2)), img, atol=1e-6)


def test_ladders(qg, calo):
    gq, gc = Geometry(qg), Geometry(calo)
    assert grid_ladder(gq, qg) == [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
    assert [t[:2] for t in train_transitions(gq, qg)] == [((4, 4), (8, 8)), ((8, 8), (16, 16)), ((16, 16), (32, 32))]
    assert grid_ladder(gc, calo)[-1] == (64, 36)
    assert np.allclose(gc.r_edges(36)[::4], gc.r_edges_native)
    assert np.allclose(gc.r_edges(3), gc.r_edges_native[::3])


@pytest.mark.parametrize("name", ["qg", "calo"])
def test_observables_grid_consistent(name, qg, calo):
    """A uniform refinement changes no grid-independent observable much; pooled energies are exact."""
    cfg = qg if name == "qg" else calo
    g = Geometry(cfg)
    rng = np.random.default_rng(0)
    img = (rng.random((4, g.P, *g.native)) * (rng.random((4, g.P, *g.native)) > 0.7)).astype(np.float64)
    thr = np.full(g.P, 1e-6)
    tgt = img.sum((1, 2, 3))
    a = observables(dense_to_cells(img), 4, g, thr, tgt)
    f = (2, 2)
    fine = uniform_upsample(img, f)
    b = observables(dense_to_cells(fine), 4, g, thr, tgt)
    key = "sum_all" if name == "qg" else "e_total"
    assert np.allclose(a[key], b[key])
    for o in (["girth", "jet_mass"] if name == "qg" else ["r_centroid", "depth_centroid", "x_centroid"]):
        assert np.allclose(a[o], b[o], rtol=0.05, atol=1e-3), o
    coarse = pool_to(img, (g.native[0] // 2, g.native[1] // (3 if name == "calo" else 2)))
    c = observables(dense_to_cells(coarse), 4, g, thr, tgt)
    assert np.allclose(a[key], c[key])


def test_model_shapes_and_loss(qg):
    from difflense.data.datasets import Collate, Featurizer
    from difflense.models.flow import cfm_loss, integrate
    from difflense.models.sr_transformer import build_sr_model
    g = Geometry(qg)
    feat = Featurizer(g, np.ones(3, np.float32), 1)
    hr = torch.rand(3, 32, 32) * (torch.rand(3, 32, 32) > 0.8)
    items = [{"parent": sum_pool(hr, (4, 4)), "child": sum_pool(hr, (2, 2)), "factor": (2, 2), "noisy": 1},
             {"parent": sum_pool(hr, (2, 2)), "child": hr, "factor": (2, 2), "noisy": 0}]
    b = Collate(feat)(items)
    model = build_sr_model(qg, g, len(feat.offsets))
    loss = cfm_loss(model, b)
    loss.backward()
    assert torch.isfinite(loss)
    x = integrate(model.eval(), b, steps=2)
    assert x.shape == b["c"].shape
    # the target reconstructs the truth exactly
    from difflense.data.datasets import denormalise
    assert torch.allclose(denormalise(b["x1"] + b["c"], b["scale"], b["area"]), b["E_true"], atol=1e-5)
