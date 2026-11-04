"""Checks written first (TDD) for stitching a separately trained CNN and LM through a DAG edge.

Run with `pytest problems/ai/stitch_llm_cnn_dag/test.py`. Exact checks: the training rule graph is acyclic with
train_cnn and train_lm on level 0; the data (sentences, masks, results mod 10) are right; the M stacked models are
independent (zero cross-gradient, batched training = per-model training); the connector stage changes only the
connector; the MoE input layer with the fixed modality router is the DAG-wired model (outputs and gradients).
Learning checks: the CNN alone and the LM alone reach their targets, the stitched model beats chance, the global
fine-tune does not lower the accuracy, and the full run (slow).
"""

import pytest
import torch

from problems.medical.cow_workflow_dag import equation as dag

from . import equation as eq
from . import solve
from . import tensors as T

CPU = torch.device("cpu")


def small_cfg(**kw) -> solve.Config:
    base = dict(seeds=[0], sizes=[3000], batch=64, steps_cnn=150, steps_lm=500, steps_con=150, steps_ft=60,
                steps_router=0, n_test=300)  # fmt: skip
    base.update(kw)
    return solve.Config(**base)


# ---------------------------------------------------------------- the rule graph
def test_training_graph_is_a_dag_with_the_expected_levels():
    d = solve.build_dag(solve.Config())
    lev = dict(zip(d["names"], d["levels"].tolist()))
    assert lev["train_cnn"] == lev["train_lm"] == 0  # pretrained separately, side by side
    assert lev["train_monolith"] == 0
    assert lev["train_connector"] == 1
    assert lev["global_finetune"] == lev["train_router"] == lev["train_router_soft"] == 2
    assert lev["evaluate"] == 3
    assert dag.is_nilpotent(d["A"]) and bool((dag.walk_traces(d["A_rule"]) == 0).all())
    S, N = 2, 4
    assert len(d["A"]) == 2 * S + 5 * S * N + 1  # cnn, lm (per seed); monolith, connector, fine-tune, 2 routers; eval
    # connector (seed, n) reads cnn (seed): the broadcast block I_S ⊗ 1_N, so each connector job has 1 CNN parent
    o, n = d["off"], d["names"]
    blk = d["A"][o[n.index("train_connector")] : o[n.index("train_connector") + 1], o[0] : o[1]]
    assert torch.equal(blk, torch.kron(torch.eye(S), torch.ones(N, 1)))


def test_a_cycle_is_refused():
    text = T.TRAINING_GRAPH.read_text(encoding="utf-8").replace(
        'input: "data/mnist_subset.npz"', 'input: "models/stitched_{seed}_{n}.pt"'
    )
    d = solve.parse_snakefile(text)
    A_rule, _ = solve.rule_graph(d)
    assert not dag.is_nilpotent(A_rule)
    with pytest.raises(ValueError):
        dag.levels(A_rule)


def test_critical_path_is_the_longest_chain():
    d = solve.build_dag(solve.Config())
    sec = {"train_cnn": 1.0, "train_lm": 3.0, "train_monolith": 5.0, "train_connector": 2.0,
           "global_finetune": 1.5, "train_router": 1.0, "train_router_soft": 1.2, "evaluate": 0.5}  # fmt: skip
    s = solve.schedule(d, sec)
    assert s["path"] == ["train_lm", "train_connector", "global_finetune", "evaluate"]
    assert s["makespan"] == pytest.approx(7.0) and s["serial"] == pytest.approx(15.2)


# ---------------------------------------------------------------- data
def test_sentences_and_masks():
    tab = T.sentence_table()
    assert T.sentence(3, 0, 2) == "three+two=five.".ljust(T.CTX, T.PAD)
    assert T.sentence(1, 1, 4).startswith("  one-four=seven.")  # 1 − 4 = −3 ≡ 7 (mod 10)
    assert T.sentence(7, 2, 8).startswith("seven*eight=six.")  # 56 ≡ 6
    assert max(len(T.sentence(a, o, b).rstrip(T.PAD)) for a in range(10) for o in range(3) for b in range(10)) == T.CTX
    a, op, b = torch.meshgrid(torch.arange(10), torch.arange(3), torch.arange(10), indexing="ij")
    r = T.result(a, op, b)
    for i in torch.randperm(300, generator=torch.Generator().manual_seed(0))[:40].tolist():
        ai, oi, bi = a.flatten()[i], op.flatten()[i], b.flatten()[i]
        s = T.decode(tab["tokens"][ai, oi, bi])
        tgt = s[1:]
        assert "".join(c for c, m in zip(tgt, tab["answer_mask"][ai, oi, bi]) if m) == T.WORDS[r.flatten()[i]] + "."
        assert "".join(c for c, m in zip(tgt, tab["text_mask"][ai, oi, bi]) if m) == s[1 : s.index(".") + 1]
    assert T.IMG not in "".join(T.sentence(x, 0, 0) for x in range(10))


def test_mnist_subset_is_balanced():
    data = T.load_mnist(CPU)
    assert data["x_train"].shape == (3000, 28, 28) and data["x_test"].shape == (1000, 28, 28)
    assert torch.equal(torch.bincount(data["y_train"]), torch.full((10,), 300))
    assert torch.equal(torch.bincount(data["y_test"]), torch.full((10,), 100))


def test_operand_never_a_target():
    """The joint task's targets start after the soft prefix: the hidden operand word is never trained on."""
    st = solve.make_store(small_cfg(), CPU)
    bt = solve.joint_draw(st, torch.arange(32)[None])
    assert not bt["mask"][..., : T.K - 1].any()
    assert bt["ans"].sum(-1).ge(4).all()  # at least 'two.' / 'one.'


# ---------------------------------------------------------------- independence on the model axis
def _models(seeds, d=32):
    return (T.init_cnn(seeds, 1, CPU), T.init_connector(seeds, 3, CPU, d), T.init_lm(seeds, 2, CPU, d, 2, 1))


def test_zero_cross_gradient():
    w_cnn, w_con, w_lm = _models([0, 1, 2])
    st = solve.make_store(small_cfg(seeds=[0, 1, 2], sizes=[3000]), CPU)
    loss = solve.joint_loss(st, torch.randint(3000, (3, 8), generator=torch.Generator().manual_seed(0)),
                            w_cnn, w_con, w_lm)  # fmt: skip
    loss[1].backward()  # only model 1's loss
    for p in T.leaves(w_cnn) + T.leaves(w_con) + T.leaves(w_lm):
        if p.grad is not None:
            assert p.grad[0].abs().max() == 0 and p.grad[2].abs().max() == 0


def test_batched_training_equals_per_seed_training():
    """Two seeds trained in one stack = each seed trained alone (Adam is elementwise, clipping is per model)."""
    cfg = small_cfg(steps_cnn=5, steps_lm=5)
    st2 = solve.make_store(solve.Config(seeds=[0, 1], sizes=[3000], steps_cnn=5, steps_lm=5, d=32, layers=1), CPU)
    both = solve.rule_train_lm(st2)
    for i, s in enumerate([0, 1]):
        st1 = solve.make_store(solve.Config(seeds=[s], sizes=[3000], steps_cnn=5, steps_lm=5, d=32, layers=1), CPU)
        one = solve.rule_train_lm(st1)
        # float32 reductions batched vs alone differ in summation order (and across BLAS builds): ~1e-5 relative
        torch.testing.assert_close(both["loss"][:, i], one["loss"][:, 0], rtol=1e-4, atol=1e-4)
        for a, b in zip(T.leaves(both["w"]), T.leaves(one["w"])):
            # Adam divides by √v: a float-rounding difference in a near-zero gradient can move a weight by up to
            # ~lr, so weights are compared at a fraction of the learning rate (the losses above agree to 1e-5)
            torch.testing.assert_close(a[i], b[0], rtol=0, atol=0.2 * st1["cfg"].lr_lm)
        c2, c1 = solve.rule_train_cnn(st2), solve.rule_train_cnn(st1)
        torch.testing.assert_close(c2["loss"][:, i], c1["loss"][:, 0], rtol=1e-4, atol=1e-4)
    c = cfg.macs()
    assert cfg.steps_mono == round((5 * c["cnn"] + 5 * c["lm"] + 210 * c["joint"]) / c["joint"])


# ---------------------------------------------------------------- the MoE view of the DAG edge
def test_fixed_router_moe_equals_the_dag_edge():
    torch.manual_seed(0)
    w_cnn, w_con, w_lm = _models([0, 1])
    st = solve.make_store(small_cfg(seeds=[0, 1]), CPU)
    bt = solve.joint_draw(st, torch.randint(3000, (2, 16), generator=torch.Generator().manual_seed(1)))
    blank = st["data"]["blank"]
    z_dag = eq.stitched_logits(bt["x"], bt["inp"], w_cnn, w_con, w_lm)
    z_moe, G, _ = solve.moe_forward(bt, w_cnn, w_con, w_lm, None, blank)
    assert torch.equal(G[0, 0, :, 0], (torch.arange(T.CTX - 1) < T.K).float())
    torch.testing.assert_close(z_moe, z_dag, rtol=0, atol=1e-6)
    params = T.leaves(w_cnn) + T.leaves(w_con) + T.leaves(w_lm)
    g_dag = torch.autograd.grad(z_dag.square().mean(), params, allow_unused=True)
    g_moe = torch.autograd.grad(z_moe.square().mean(), params, allow_unused=True)
    for a, b in zip(g_dag, g_moe):
        if a is None:
            assert b is None or b.abs().max() == 0
        else:
            torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-7)


def test_balance_loss_limits():
    G = torch.zeros(1, 4, 10, 2)
    G[..., 0] = 1
    assert eq.balance_loss(G, G).item() == pytest.approx(2.0)  # all to one expert: X · 1 · 1
    p = torch.full((1, 4, 10, 2), 0.5)
    Gh = torch.zeros(1, 4, 10, 2)
    Gh[:, :, :5, 0], Gh[:, :, 5:, 1] = 0.5, 0.5
    assert eq.balance_loss(Gh, p).item() == pytest.approx(1.0)  # even split


def test_dispatch_packs_each_group():
    mask = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.bool)
    idx, valid = eq.dispatch(mask)
    assert idx.shape == (2, 2) and torch.equal(idx[0], torch.tensor([0, 2])) and int(idx[1, 0]) == 3
    assert torch.equal(valid, torch.tensor([[True, True], [True, False]]))


# ---------------------------------------------------------------- learning (seeded; about a minute on a CPU)
@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    cfg = small_cfg()
    st = solve.make_store(cfg, CPU)
    for name in ("train_cnn", "train_lm", "train_connector", "global_finetune"):
        st[name] = solve.RULES[name](st)
    return st


def test_connector_stage_changes_only_the_connector():
    cfg = small_cfg(seeds=[0, 1], sizes=[100, 3000], steps_cnn=3, steps_lm=3, steps_con=3, d=32, layers=1)
    st = solve.make_store(cfg, CPU)
    for name in ("train_cnn", "train_lm"):
        st[name] = solve.RULES[name](st)
    c = solve.rule_train_connector(st)
    for a, b in zip(T.leaves(c["cnn"]), T.leaves(st["train_cnn"]["w"])):  # seed → (seed, n): I ⊗ 1, untouched
        assert torch.equal(a, b.detach().repeat_interleave(2, 0))
    for a, b in zip(T.leaves(c["lm"]), T.leaves(st["train_lm"]["w"])):
        assert torch.equal(a, b.detach().repeat_interleave(2, 0))
    assert all(not torch.equal(a, b) for a, b in zip(T.leaves(c["con"]), T.leaves(c["con_init"])))


def _acc(st, stage, n=300):
    data = st["data"]
    test = T.joint_test(n, CPU)
    sl = torch.arange(n)
    bt = solve.joint_batch(data, sl[None], test["op"][None], test["b"][None], train=False)
    s = st[stage]
    with torch.no_grad():
        z = eq.stitched_logits(bt["x"], bt["inp"], s["cnn"], s["con"], s["lm"])
    return eq.exact(z, bt["tgt"], bt["ans"]).float().mean().item()


@pytest.mark.slow
def test_cnn_alone_reads_digits(trained):
    w, data = trained["train_cnn"]["w"], trained["data"]
    with torch.no_grad():
        pred = eq.cnn_logits(eq.cnn_features(data["x_test"][None, :500], w), w).argmax(-1)[0]
    assert (pred == data["y_test"][:500]).float().mean() > 0.9


@pytest.mark.slow
def test_lm_alone_completes_text_prompts(trained):
    w, tab = trained["train_lm"]["w"], trained["data"]["table"]
    tok, ans = tab["tokens"].reshape(1, 300, T.CTX), tab["answer_mask"].reshape(1, 300, T.CTX - 1)
    with torch.no_grad():
        z = eq.lm_logits(eq.embed(tok[..., :-1], w), w)
    assert eq.exact(z, tok[..., 1:], ans).float().mean() > 0.9


@pytest.mark.slow
def test_stitched_beats_chance_and_finetune_keeps_it(trained):
    a_con = _acc(trained, "train_connector")
    a_ft = _acc(trained, "global_finetune")
    assert a_con > 0.5  # chance is 0.1 (ten result words)
    assert a_ft >= a_con - 0.02


@pytest.mark.slow
def test_full_run(tmp_path):
    from pathlib import Path

    from problems._common.runtime import load_meta, resolve_params

    from . import main

    out = main.run(resolve_params(load_meta(Path(__file__).parent)), tmp_path)
    assert out.exists() and (tmp_path / "stages.csv").exists() and (tmp_path / "timings.csv").exists()
