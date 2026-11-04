"""Checks written first (TDD): a workflow DAG with a fixed router is a top-1 mixture of experts, exactly.

Run with `pytest problems/ai/moe_dag_pipeline/test.py`. Exact checks: the task's answer table, the text parser (the
router of the workflow), the Snakefile's rule graph and its routed job DAG (acyclic, levels, only the N chosen fuse
jobs needed), the DAG engine against the Switch-style dispatch → experts → combine layer with G fixed to the rule
choice (identical outputs), the einsum dispatch/combine against a loop, one-hot router rows and the load-balancing
loss on known cases, the distillation loss and int8 fake quantisation. Learning checks (slow, one shared training
run): split-trained experts reach their sub-task accuracy, end-to-end fine-tuning does not lower the accuracy, the
student stays near its teacher, and the int8 student is ≥ 3.5× smaller at almost the same accuracy.
"""

import itertools
import math

import pytest
import torch

from . import equation, solve, tensors

E, V = len(tensors.OPS), len(tensors.VOCAB)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _weights(M=2, seed=0, dtype=torch.float64):
    return tensors.cast(tensors.init_weights(M, tensors.Config(), _gen(seed)), dtype)


# ---------------------------------------------------------------- data and the router of the workflow
def test_answer_table():
    d, op, k = torch.meshgrid(torch.arange(10), torch.arange(E), torch.arange(10), indexing="ij")
    y = tensors.answer(d.flatten(), op.flatten(), k.flatten())
    for di, oi, ki, yi in zip(d.flatten().tolist(), op.flatten().tolist(), k.flatten().tolist(), y.tolist()):
        ref = {0: di + ki, 1: di - ki, 2: di * ki}[oi] % 10  # Python's % is never negative here
        assert yi == ref


def test_texts_have_one_op_one_number_one_filler_and_parse_back():
    tok, op, k = tensors.make_texts(500, _gen(1))
    assert tok.shape == (500, tensors.L)
    is_op, is_num = tok < E, (tok >= E) & (tok < E + 10)
    assert torch.equal(is_op.sum(1), torch.ones(500, dtype=torch.long))
    assert torch.equal(is_num.sum(1), torch.ones(500, dtype=torch.long))
    assert torch.equal(tensors.parse_op(tok), op)  # the rule-based parser = Snakemake's input function
    assert torch.equal(tensors.parse_number(tok), k)
    G = tensors.router_matrix(tok)
    assert torch.equal(G.sum(1), torch.ones(500)) and torch.equal(G.argmax(1), op)
    assert len(set(is_op.float().argmax(1).tolist())) == tensors.L  # the operation word appears at every position


def test_mnist_subset_is_balanced_uint8():
    d = tensors.load_mnist()
    assert d["x_train"].shape == (3000, 784) and d["x_test"].shape == (1000, 784)
    assert torch.equal(torch.bincount(d["y_train"]), torch.full((10,), 300))
    assert torch.equal(torch.bincount(d["y_test"]), torch.full((10,), 100))
    assert 0.0 <= float(d["x_train"].min()) and float(d["x_train"].max()) <= 1.0


# ---------------------------------------------------------------- the workflow DAG
def test_snakefile_rule_graph_and_levels():
    dag = solve.build_dag(tensors.SNAKEFILE.read_text(encoding="utf-8"), n_samples=5)
    names = [r["name"] for r in dag["rules"]]
    assert set(names) == {"collect", "encode_image", "encode_text", "fuse", "classify"}
    idx = {n: i for i, n in enumerate(names)}
    edges = {(names[s], names[r]) for s, r in dag["A_rule"].nonzero().tolist()}
    assert edges == {("fuse", "encode_image"), ("fuse", "encode_text"), ("classify", "fuse"), ("collect", "classify")}
    assert dag["conditional"] == {(idx["classify"], idx["fuse"]): "op"}  # chosen per sample by `routed`
    lev = dict(zip(names, dag["levels"].tolist()))
    assert lev == {"encode_image": 0, "encode_text": 0, "fuse": 1, "classify": 2, "collect": 3}
    assert dag["axes"][idx["fuse"]] == ("sample", "op") and dag["axes"][idx["collect"]] == ()


def test_routed_job_dag_keeps_only_the_chosen_fuse_jobs():
    N = 7
    tok, op, _ = tensors.make_texts(N, _gen(2))
    G = tensors.router_matrix(tok)
    dag = solve.build_dag(tensors.SNAKEFILE.read_text(encoding="utf-8"), n_samples=N)
    A, off = solve.job_adjacency_routed(dag, G)
    J = N * (1 + 1 + E + 1) + 1
    assert A.shape == (J, J)
    f, c = dag["names"].index("fuse"), dag["names"].index("classify")
    block = A[off[c] : off[c + 1], off[f] : off[f + 1]]  # classify job n reads fuse job (n', e)
    ref = torch.zeros(N, N * E)
    for n in range(N):
        ref[n, n * E + int(op[n])] = 1.0
    assert torch.equal(block, ref)  # = (I ⊗ 𝟙ᵀ) ⊙ vec(G)ᵀ: the router sits inside the adjacency matrix
    from problems.medical.cow_workflow_dag import equation as dag_eq

    assert dag_eq.is_nilpotent(A) and bool((dag_eq.walk_traces(A) == 0).all())  # acyclic
    needed = solve.needed_jobs(dag, G)  # block-wise, never builds the (J, J) matrix
    R = dag_eq.closure(A)
    target = torch.zeros(J, dtype=torch.bool)  # the job of the target rule `collect`
    k = dag["names"].index("collect")
    target[off[k] : off[k + 1]] = True
    ref_needed = target | ((R.double().T @ target.double()) > 0)  # every job the target depends on
    assert torch.equal(torch.cat([needed[n] for n in dag["names"]]), ref_needed)
    assert torch.equal(needed["fuse"].reshape(N, E), G.bool())
    assert int(ref_needed.sum()) == N * 4 + 1  # N of the N·E fuse jobs, plus every other job


# ---------------------------------------------------------------- DAG ≡ fixed-router MoE
def test_dag_engine_equals_fixed_router_moe_exactly():
    data = tensors.load_mnist()
    N = 300
    x = data["x_test"][:N].double()
    tok, _, _ = tensors.make_texts(N, _gen(3))
    w = _weights(M=2, seed=4)
    dag = solve.build_dag(tensors.SNAKEFILE.read_text(encoding="utf-8"), n_samples=N)
    z_dag, _ = solve.run_dag(w, x, tok, dag)
    G = tensors.router_matrix(tok).double().expand(2, N, E)
    z_moe = solve.moe_forward(w, x, tok, G, combine=G)
    assert z_dag.shape == z_moe.shape == (2, N, 10)
    assert float((z_dag - z_moe).abs().max()) <= 1e-12  # float64: identical up to summation order
    assert torch.equal(z_dag.argmax(-1), z_moe.argmax(-1))


def test_dispatch_combine_matches_loop():
    M, N, C_in, H = 2, 40, 6, 5
    g = _gen(5)
    u = torch.randn(M, N, C_in, generator=g, dtype=torch.float64)
    probs = torch.softmax(torch.randn(M, N, E, generator=g, dtype=torch.float64), -1)
    G = equation.top1(probs)
    W = torch.randn(M, E, C_in, H, generator=g, dtype=torch.float64)
    C = int(G.sum(1).max())
    D = equation.dispatch(G, C)  # (M, N, E, C)
    U = torch.einsum("mnec,mni->meci", D, u)
    Y = torch.einsum("meci,meih->mech", U, W)
    out = equation.combine(D, G * probs, Y)
    ref = torch.zeros(M, N, H, dtype=torch.float64)
    for m in range(M):
        for n in range(N):
            e = int(probs[m, n].argmax())
            ref[m, n] = probs[m, n, e] * (u[m, n] @ W[m, e])
    assert torch.allclose(out, ref, rtol=0, atol=1e-12)
    assert torch.equal(D.sum((2, 3)), torch.ones(M, N, dtype=torch.float64))  # every token in exactly one slot
    assert float(D.sum(1).max()) == 1.0  # every slot holds at most one token


def test_router_rows_are_one_hot_and_load_balance_loss():
    probs = torch.softmax(torch.randn(3, 64, E, generator=_gen(6)), -1)
    G = equation.top1(probs)
    assert torch.equal(G.sum(-1), torch.ones(3, 64)) and set(G.unique().tolist()) <= {0.0, 1.0}
    # perfectly balanced, uniform router: E · Σ_e (1/E)(1/E) = 1
    Gb = torch.eye(E).repeat(1, 4, 1)  # (1, 12, E) 4 tokens per expert
    assert torch.allclose(equation.load_balance_loss(torch.full((1, 12, E), 1 / E), Gb), torch.ones(1))
    # collapsed: all tokens and all probability on expert 0 → E
    Gc = torch.zeros(1, 12, E)
    Gc[..., 0] = 1
    assert torch.allclose(equation.load_balance_loss(Gc.clone(), Gc), torch.tensor([float(E)]))
    # a known mixed case: f = (1/2, 1/2, 0), P = (0.6, 0.3, 0.1) → 3 (0.3 + 0.15) = 1.35
    P = torch.tensor([[[0.6, 0.3, 0.1]]]).expand(1, 4, E)
    Gm = torch.tensor([[[1.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]])
    assert torch.allclose(equation.load_balance_loss(P, Gm), torch.tensor([1.35]))


def test_distillation_loss():
    t = torch.randn(2, 30, 10, generator=_gen(7))
    assert torch.allclose(equation.kd_loss(t, t, 4.0), torch.zeros(2), atol=1e-6)
    s = t + torch.randn(2, 30, 10, generator=_gen(8))
    assert bool((equation.kd_loss(s, t, 4.0) > 0).all())


def test_int8_fake_quant_error_and_size():
    W = torch.randn(2, 50, 20, generator=_gen(9))
    Wq, scale = equation.fake_quant(W)  # per output channel, symmetric
    assert scale.shape == (2, 1, 20)
    q = Wq / scale
    assert torch.allclose(q, q.round(), atol=1e-4) and float(q.abs().max()) <= 127.0 + 1e-4
    assert bool(((Wq - W).abs() <= scale / 2 + 1e-7).all())
    w = tensors.init_student(1, tensors.Config(), _gen(10))
    assert solve.int8_bytes(w) * 3.5 <= solve.float_bytes(w)


# ---------------------------------------------------------------- learning (one shared run, the default settings)
@pytest.fixture(scope="module")
def trained():
    cfg = tensors.Config()  # the default run (about a minute on a CPU)
    return solve.run_stages(cfg, n_checkpoints=3)


@pytest.mark.slow
def test_split_experts_reach_their_subtask_accuracy(trained):
    sub = trained["subtask_acc"]  # (M,) each
    assert bool((sub["image"] >= 0.85).all())  # MNIST digits, 3000 training images, an MLP
    assert bool((sub["text"] >= 0.99).all())  # the number in a three-word text
    assert bool((sub["fuse"] >= 0.99).all())  # the 300-entry arithmetic table from one-hot inputs


@pytest.mark.slow
def test_global_finetune_does_not_lower_accuracy(trained):
    acc = trained["acc"]
    assert bool((acc["global"] >= acc["split"] - 0.01).all())  # no harm, within 10 of 1000 test samples per seed


@pytest.mark.slow
def test_student_near_teacher_and_int8_small(trained):
    acc = trained["acc"]
    assert bool((acc["student"] >= acc["global"] - 0.10).all())  # still learning after 3000 steps
    assert bool((trained["agree_teacher"] >= 0.80).all())
    assert bool(((acc["int8"] - acc["student"]).abs() <= 0.01).all())
    assert trained["bytes"]["int8"] * 3.5 <= trained["bytes"]["student"]
    assert trained["params"]["student"] < trained["params"]["global"]


@pytest.mark.slow
def test_learned_router_is_a_valid_assignment(trained):
    conf = trained["router_confusion"]  # (M, E, E) true operation × chosen expert, fractions
    assert torch.allclose(conf.sum((1, 2)), torch.ones(conf.shape[0]), atol=1e-6)
    rec = trained["router_recovery"]
    best = torch.stack([conf[:, list(range(E)), list(p)].sum(1) for p in itertools.permutations(range(E))]).max(0)
    assert torch.allclose(rec, best.values, atol=1e-6)


@pytest.mark.slow
def test_full_run(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORIZER_FAST", "1")
    from . import main

    out = main.run(
        {"seeds": 2, "split_steps": 1000, "router_steps": 800, "global_steps": 400, "student_steps": 3000,
         "student_hidden": 64, "temperature": 4.0, "balance_weight": 0.01},
        tmp_path,
    )  # fmt: skip
    assert out.exists() and out.suffix == ".gif"
    assert (tmp_path / "stages.csv").exists() and (tmp_path / "equivalence.csv").exists()
    assert math.isfinite(float((tmp_path / "stages.csv").read_text().splitlines()[1].split(",")[2]))


@pytest.mark.slow
def test_gate_in_front_of_the_workflow_experts_rediscovers_the_rule(trained):
    M = trained["M"]
    assert bool((trained["router_identity"][:M] >= 0.75).all())  # operation e → expert e for most samples
