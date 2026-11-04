"""Checks, written first (TDD): every operator equals its formula, the mixture and the discretisation do what DARTS
says, the written equation is the trained DAG (same predictions, exactly), the baselines are their closed forms, and
both learners beat chance from 10 images per class.

Run with `pytest problems/ai/operator_search_mnist/test.py` (CPU, seeded).
"""

import math

import numpy as np
import pytest
import torch

from . import equation as eq
from . import solve
from . import tensors as T

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def data():
    return T.load(CPU)


def corr2d(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Slow reference: zero-padded 2D cross-correlation, out[i, j] = Σ_ab k[a, b] img[i + a − r, j + b − r]."""
    r = k.shape[0] // 2
    pad = np.pad(img, r)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (k * pad[i : i + k.shape[0], j : j + k.shape[1]]).sum()
    return out


# ---------------------------------------------------------------- data
def test_bundled_subset_is_balanced_mnist():
    with np.load(T.DATA) as d:
        x, y = d["images"], d["labels"]
    assert x.shape == (3000, 28, 28) and x.dtype == np.uint8 and y.shape == (3000,) and y.dtype == np.uint8
    assert np.all(np.bincount(y[: T.N_POOL]) == 100) and np.all(np.bincount(y[T.N_POOL :]) == 200)
    assert x.max() == 255 and x.min() == 0


def test_small_set_is_balanced_and_split(data):
    for n in T.SIZES:
        tr, va = T.small_set(data["y_pool"], n, seed=0)
        assert torch.all(torch.bincount(data["y_pool"][tr], minlength=10) == math.ceil(n / 2))
        if n > 1:
            assert torch.all(torch.bincount(data["y_pool"][va], minlength=10) == n // 2)
            assert len(set(tr.tolist()) & set(va.tolist())) == 0
        else:
            assert torch.equal(tr, va)


# ---------------------------------------------------------------- every operator equals its formula
def test_cell_operators_match_their_formulas():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(2, 1, 6, 6, generator=g)
    K = torch.randn(3, 3, generator=g)
    zn = z.numpy()
    ref = {"identity": zn, "square": zn**2, "abs": np.abs(zn), "relu": np.maximum(zn, 0), "tanh": np.tanh(zn),
           "shrink": np.sign(zn) * np.maximum(np.abs(zn) - T.LAMBDA, 0), "zero": 0 * zn}  # fmt: skip
    bank = T.fixed_kernels().numpy()
    for name, k in [("conv3", K.numpy())] + [(f, bank[i]) for i, f in enumerate(eq.FIXED)]:
        ref[name] = np.stack([corr2d(zn[b, 0], k) for b in range(2)])[:, None]
    assert set(ref) == set(eq.CELL_OPS)
    for name, r in ref.items():
        np.testing.assert_allclose(eq.cell_op(name, z, K).numpy(), r, atol=1e-5, err_msg=name)


def test_fixed_kernels_are_the_textbook_ones():
    bank = T.fixed_kernels()
    assert torch.equal(bank[0, 1:4, 1:4], torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    assert torch.equal(bank[1, 1:4, 1:4], bank[0, 1:4, 1:4].T)
    assert torch.equal(bank[2, 1:4, 1:4], torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    assert abs(float(bank[3].sum())) < 1e-6  # zero-mean Gabor: blind to a constant background
    ramp = torch.arange(8.0)[None, None, None].expand(1, 1, 8, 8)  # brightness grows along columns
    assert torch.allclose(eq.cell_op("sobel_x", ramp)[0, 0, 2:-2, 2:-2], torch.full((4, 4), 8.0))
    assert torch.allclose(eq.cell_op("laplace", ramp)[0, 0, 2:-2, 2:-2], torch.zeros(4, 4), atol=1e-6)


def test_pool_and_reduce_operators_match_their_formulas():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(2, 3, 4, 4, generator=g)
    zn = z.numpy()
    avg = zn.reshape(2, 3, 2, 2, 2, 2).mean((3, 5))
    mx = zn.reshape(2, 3, 2, 2, 2, 2).max((3, 5))
    np.testing.assert_allclose(eq.pool_op("avgpool", z).numpy(), avg, atol=1e-6)
    np.testing.assert_allclose(eq.pool_op("maxpool", z).numpy(), mx, atol=1e-6)

    Z = torch.randn(5, 4, 7, 7, generator=g)
    Zn = Z.numpy()
    st = {"mu": torch.randn(10, 4, 7, 7, generator=g), "U": torch.randn(4, 49, T.Q_PCA, generator=g)}
    u = np.linspace(-1, 1, 7)
    U, V = np.meshgrid(u, u, indexing="ij")
    basis = [np.ones_like(U), U, V, U**2, V**2, U * V]
    ref = {
        "vec": Zn.reshape(5, -1),
        "rowsum": np.array([[Zn[b, c, h, :].sum() for c in range(4) for h in range(7)] for b in range(5)]),
        "colsum": np.array([[Zn[b, c, :, w].sum() for c in range(4) for w in range(7)] for b in range(5)]),
        "fft": np.abs(np.fft.fft2(Zn) / 7)[..., :4, :4].reshape(5, -1),
        "template": np.einsum("bchw,kchw->bck", Zn, st["mu"].numpy()).reshape(5, -1),
        "pca": np.einsum("bcp,cpq->bcq", Zn.reshape(5, 4, -1), st["U"].numpy()).reshape(5, -1),
        "moments": np.array([[(Zn[b, c] * m).mean() for c in range(4) for m in basis] for b in range(5)]),
    }
    assert set(ref) == set(eq.RED_OPS)
    for name, r in ref.items():
        out = eq.reduce_op(name, Z, st)
        assert out.shape[1] == eq.reduce_dims()[name], name
        np.testing.assert_allclose(out.numpy(), r, rtol=1e-4, atol=1e-4, err_msg=name)


def test_heads_match_their_formulas():
    g = torch.Generator().manual_seed(2)
    fh = torch.randn(4, 6, generator=g)
    th = {"W_vec": torch.randn(10, 6, generator=g), "b_vec": torch.randn(10, generator=g),
          "P_vec": torch.randn(10, 6, generator=g), "tau_vec": torch.tensor(0.3)}  # fmt: skip
    lin = np.array([[sum(th["W_vec"][k, j] * fh[b, j] for j in range(6)) + th["b_vec"][k] for k in range(10)]
                    for b in range(4)])  # fmt: skip
    np.testing.assert_allclose(eq.head_op("linear", fh, th, "vec").numpy(), lin, atol=1e-5)
    d2 = ((fh[:, None] - th["P_vec"][None]) ** 2).sum(-1)
    pr = eq.head_op("proto", fh, th, "vec")
    np.testing.assert_allclose(pr.numpy(), (-math.exp(0.3) * d2 / 6).numpy(), atol=1e-5)
    assert torch.equal(pr.argmax(1), d2.argmin(1))  # the decision is the nearest prototype


# ---------------------------------------------------------------- mixture, typing rule, discretisation
def test_masked_softmax_respects_the_typing_rule():
    M = eq.family_mask()
    assert M.shape == (9, 23) and M[:6].sum(1).tolist() == [12] * 6 and M[6:].sum(1).tolist() == [2, 7, 2]
    alpha = torch.randn(9, 23, generator=torch.Generator().manual_seed(3))
    w = eq.mixture_weights(alpha, M)
    assert torch.all(w[~M] == 0) and torch.allclose(w.sum(1), torch.ones(9))


def test_one_hot_mixture_is_the_single_operator():
    g = torch.Generator().manual_seed(4)
    z = torch.randn(3, 1, 8, 8, generator=g)
    K = torch.randn(3, 3, generator=g)
    for o, name in enumerate(eq.CELL_OPS):
        row = torch.zeros(23)
        row[o] = 1.0
        assert torch.equal(eq.mixed_edge(z, row, K), eq.cell_op(name, z, K)), name
    row = torch.zeros(23)
    row[[1, 4]] = torch.tensor([0.25, 0.75])  # a real mixture is the weighted sum
    assert torch.allclose(eq.mixed_edge(z, row, K), 0.25 * z + 0.75 * z.clamp(min=0))


def test_one_hot_supernet_is_the_discrete_dag(data):
    arch = {"edges": [(0, "sobel_x"), (1, "conv3"), (2, "abs"), (3, "relu"), (5, "square")], "pool": "maxpool",
            "reduce": "colsum", "head": "linear"}  # fmt: skip
    W = solve.onehot(arch)
    x = data["x_pool"][:20]
    th = T.init_supernet(6, eq.reduce_dims(), 0, CPU)
    z1 = eq.cell_op("sobel_x", x)
    z2 = eq.cell_op("conv3", x, th["K"][1]) + eq.cell_op("abs", z1)
    z3 = eq.cell_op("relu", x) + eq.cell_op("square", z2)
    with torch.no_grad():
        assert torch.equal(eq.pooled(x, W, th), eq.pool_op("maxpool", torch.cat([x, z1, z2, z3], 1)))


def test_discretisation_picks_the_argmax():
    alpha = torch.randn(9, 23, generator=torch.Generator().manual_seed(5))
    w = eq.mixture_weights(alpha, eq.family_mask())
    w[:6, 0] = 0
    w[1, 0] = 1.0  # 'zero' wins edge z0→z2: the edge is removed (z_2 still has z1→z2)
    arch = solve.discretise(w)
    kept = dict(arch["edges"])
    assert 1 not in kept
    for e in (0, 2, 3, 4, 5):
        assert kept[e] == eq.OPS[int(w[e].argmax())]
    for name in ("pool", "reduce", "head"):
        assert arch[name] == eq.OPS[int(w[eq.EDGES.index(name)].argmax())]
    assert torch.equal(arch["onehot"].sum(1), torch.tensor([1.0, 0, 1, 1, 1, 1, 1, 1, 1]))
    w[0, 0], w[2, 0] = 1.0, 1.0  # now z_1 and z_2 lose every input: each keeps its strongest non-zero edge
    arch = solve.discretise(w)
    nz = [eq.OPS.index(o) for o in eq.CELL_OPS[1:]]
    assert {e for e, _ in arch["edges"]} >= {0} and sum(e in (1, 2) for e, _ in arch["edges"]) == 1
    e2 = max((1, 2), key=lambda e: float(w[e, nz].max()))
    assert dict(arch["edges"])[e2] == eq.OPS[nz[int(w[e2, nz].argmax())]]


def test_mixed_logits_are_the_weighted_sum_of_all_heads(data):
    """The batched head mixture equals Σ_r Σ_h a_r b_h head_{r,h}(f̂_r) computed one head at a time."""
    g = torch.Generator().manual_seed(6)
    w = eq.mixture_weights(torch.randn(9, 23, generator=g), eq.family_mask())
    th = T.init_supernet(6, eq.reduce_dims(), 0, CPU)
    x, y = data["x_pool"][:60], data["y_pool"][:60]
    st = solve.statistics(x, y, w, th, list(eq.RED_OPS))
    with torch.no_grad():
        Z = eq.pooled(x, w, th)
        ref = sum(w[7, eq.OPS.index(r)] * w[8, eq.OPS.index(h)]
                  * eq.head_op(h, (eq.reduce_op(r, Z, st) - st[f"m_{r}"]) / st[f"s_{r}"], th, r)
                  for r in eq.RED_OPS for h in eq.HEAD_OPS)  # fmt: skip
        assert torch.allclose(eq.logits(Z, w, th, st), ref, atol=1e-4)


# ---------------------------------------------------------------- the equation writer
def random_arch(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    alpha = 3 * torch.randn(9, 23, generator=g)
    return solve.discretise(eq.mixture_weights(alpha, eq.family_mask()))


def test_written_equation_reads_back_to_the_same_dag():
    for seed in range(40):
        arch = random_arch(seed)
        back = solve.read_equation(solve.write_equation(arch))
        assert back == {k: arch[k] for k in ("edges", "pool", "reduce", "head")}


def test_written_equation_reproduces_the_dag_predictions_exactly(data):
    x, y = data["x_pool"][::4], data["y_pool"][::4]
    seen = set()
    for seed in range(30):
        arch = random_arch(seed)
        seen |= {arch["reduce"], arch["head"]}
        th = T.init_supernet(6, eq.reduce_dims(), seed, CPU)
        W = arch["onehot"]
        st = solve.statistics(x, y, W, th, [arch["reduce"]])
        lines = solve.write_equation(arch)
        assert torch.equal(solve.evaluate_equation(lines, x, th, st), solve.predict(x, W, th, st)), lines
    assert seen >= {"linear", "proto"} and len(seen) >= 6


def test_equation_of_a_known_dag():
    arch = {"edges": [(0, "conv3"), (1, "relu"), (2, "abs"), (3, "identity"), (4, "tanh")], "pool": "avgpool",
            "reduce": "template", "head": "linear"}  # fmt: skip
    lines = solve.write_equation(arch)
    assert lines[0] == r"z_{1} = K_{1} \ast x"
    assert lines[1] == r"z_{2} = \mathrm{ReLU}(x) + |z_{1}|"
    assert lines[2] == r"z_{3} = x + \tanh(z_{1})"
    assert lines[3] == r"f = \langle \mathrm{avgpool}_{2}(z), \mu_{k} \rangle"


# ---------------------------------------------------------------- baselines
def test_nearest_class_mean_is_the_direct_computation(data):
    x, y, q = data["x_pool"][:200], data["y_pool"][:200], data["x_test"][:50]
    X = x.reshape(200, -1).numpy().astype(np.float64)  # (200, 196) half-resolution pixels
    mu = np.stack([X[y.numpy() == c].mean(0) for c in range(10)])
    d = ((q.reshape(50, -1).numpy()[:, None, :] - mu[None]) ** 2).sum(-1)
    assert np.array_equal(eq.nearest_class_mean(x, y, q).numpy(), d.argmin(1))


def test_pca_lda_with_one_image_per_class_is_nearest_class_mean(data):
    tr, _ = T.small_set(data["y_pool"], 1, seed=0)
    x, y, q = data["x_pool"][tr], data["y_pool"][tr], data["x_test"]
    # S_w = 0 → Σ = I; the 9 PCA axes span the 10 class means, so distances differ from pixel space by a constant
    assert torch.equal(eq.pca_lda(x, y, q), eq.nearest_class_mean(x, y, q))


def test_parameter_counts():
    p = T.init_cnn(0, CPU)
    assert sum(t.numel() for t in p.values()) == T.cnn_param_count() == 2698
    arch = {"edges": [(0, "conv3"), (1, "conv3"), (2, "abs")], "pool": "avgpool", "reduce": "pca", "head": "proto"}
    assert solve.write_equation(arch)[2] == "z_{3} = 0"  # no edge into z_3
    assert solve.read_equation(solve.write_equation(arch)) == {k: arch[k] for k in arch}
    assert solve.n_trainable(arch) == 2 * 9 + 10 * 32 + 1


def test_data_needed_interpolates_in_log_n():
    sizes = [1, 10, 100]
    assert solve.n_to_reach(sizes, [0.5, 0.7, 0.9], 0.8) == pytest.approx(math.sqrt(10 * 100))
    assert solve.n_to_reach(sizes, [0.95, 0.96, 0.97], 0.9) == 1.0
    assert math.isnan(solve.n_to_reach(sizes, [0.5, 0.6, 0.7], 0.9))


# ---------------------------------------------------------------- learning from 10 images per class
def test_cnn_beats_chance_at_n10(data):
    torch.manual_seed(0)
    tr, va = T.small_set(data["y_pool"], 10, seed=0)
    r = solve.run_cnn(data, tr, va, {"cnn_steps": 120, "batch": 128}, seed=0)
    assert r["acc"] > 0.4  # chance is 0.1


def test_searched_dag_beats_chance_at_n10(data):
    torch.manual_seed(0)
    tr, va = T.small_set(data["y_pool"], 10, seed=0)
    r = solve.run_tensor(data, tr, va, {"search_steps": 40, "retrain_steps": 60, "batch": 128}, seed=0)
    assert r["acc"] > 0.4
    assert len(r["equation"]) == 6
    pred = solve.evaluate_equation(r["equation"], data["x_test"], r["theta"], r["stats"])
    assert torch.equal(pred, r["pred"])  # the written equation of the trained DAG gives its predictions


@pytest.mark.slow
def test_full_study_cnn_and_dag(data):
    """All sizes, 2 seeds: both learners improve with data; the parameter-free baselines are reproduced."""
    cfg = {"search_steps": 150, "retrain_steps": 150, "cnn_steps": 300, "batch": 128}
    acc = {"cnn": [], "dag": []}
    for n in (1, 10, 100):
        a_c, a_d = [], []
        for seed in range(2):
            tr, va = T.small_set(data["y_pool"], n, seed)
            a_c.append(solve.run_cnn(data, tr, va, cfg, seed)["acc"])
            a_d.append(solve.run_tensor(data, tr, va, cfg, seed)["acc"])
        acc["cnn"].append(np.mean(a_c))
        acc["dag"].append(np.mean(a_d))
    for k in acc:
        assert acc[k][0] < acc[k][1] < acc[k][2], (k, acc[k])
        assert acc[k][2] > 0.8, (k, acc[k])
