"""Checks written first (TDD) for the chest X-ray report toy: a CNN and a tiny report LM trained apart, stitched by a
DAG edge, then fine-tuned on real figure captions.

Run with `pytest problems/ai/cxr_report_dag/test.py`. Exact checks: the template generator and the report parser
round-trip (every label set, every phrasing), the parser handles negation, the NIH labels are parsed and the bundled
subsets have the documented shapes, the training rule graph has the planned levels, the MoE input layer with the
fixed modality router equals the DAG wiring (outputs and gradients), the connector stage changes only the
connector, the caption stage leaves the CNN alone, models on the stacked axis are independent (zero cross-gradient,
batched over seeds = each seed alone). Learning checks: the CNN beats chance, the LM reproduces the templates, and
the full run (slow, thresholds with headroom: float reductions differ across BLAS builds).
"""

import pytest
import torch

from problems.medical.cow_workflow_dag import equation as dag

from . import equation as eq
from . import solve
from . import tensors as T

CPU = torch.device("cpu")


def small_cfg(**kw) -> solve.Config:
    base = dict(seeds=[0], steps_cnn=4, steps_lm=4, steps_con=3, steps_ft=2, steps_router=2, steps_cap=2,
                d=32, layers=1)  # fmt: skip
    base.update(kw)
    return solve.Config(**base)


# ---------------------------------------------------------------- templates and parser
def test_every_template_round_trips_through_the_parser():
    sets = T.label_sets()
    assert len(sets) == 1 + 6 + 15  # no finding, one finding, two findings
    n = 0
    for y in sets:
        for v in range(T.N_VARIANTS):
            text = T.report(y, v)
            assert len(text) <= T.MAX_TXT
            assert all(c in T.STOI for c in text)
            assert T.parse_report(text) == tuple(y)  # back to the same labels
            assert T.template_labels(text) == tuple(y)  # and it is a well-formed template
            n += 1
    assert n == 22 * T.N_VARIANTS
    # the phrasings really differ (the LM must learn a distribution, not one string per label set)
    assert len({T.report(sets[7], v) for v in range(T.N_VARIANTS)}) >= 3


def test_parser_reads_negation_and_rejects_free_text():
    assert T.parse_report("enlarged heart. no pneumothorax.") == (1, 0, 0, 0, 0, 0)
    assert T.parse_report("small pneumothorax.") == (0, 0, 0, 0, 1, 0)
    assert T.parse_report("normal chest.") == (0,) * 6
    assert T.parse_report("pleural effusion and a nodule") == (0, 1, 0, 0, 0, 1)  # keywords, even off-template
    assert T.template_labels("pleural effusion and a nodule") is None  # but not a template
    assert T.template_labels("effusion. cardiomegaly.") is None  # wrong order
    assert T.template_labels("small pneumothorax. no pneumothorax.") is None  # contradiction
    assert T.template_labels("normal chest. nodule.") is None


def test_sequence_layout():
    y = (1, 1, 0, 0, 0, 0)
    s = T.sequence(y, 0)
    assert len(s) == T.CTX and s[: T.K] == "ce----" and s[T.K] == T.BOS
    assert s[T.K + 1 :].rstrip(T.PAD).endswith(T.EOS)
    tab = T.template_table()
    assert tab["tokens"].shape[1] == T.CTX
    # targets: from the first report letter (predicted at the BOS position) up to and including EOS, never the header
    m = tab["mask"][0]
    assert not m[: T.K].any() and m[T.K]
    row = T.decode(tab["tokens"][0])
    assert "".join(c for c, k in zip(row[1:], m) if k) == row[T.K + 1 : row.index(T.EOS) + 1]


def test_stage2_text_cleaning_keeps_whole_sentences():
    raw = (
        "Figure 2. (A) Chest X-ray showing right-sided pleural effusion; arrows mark the fluid level. "
        "A chest radiograph (Fig. 2b) showed blunting of the right costophrenic angle. " + "More text here. " * 40
    )
    c, cut = T.clean_text(raw)
    assert not cut and len(c) <= T.MAX_CAP and all(ch in T.STOI for ch in c)
    assert c.startswith("chest x-ray showing right-sided pleural effusion")
    assert "fig" not in c and c.endswith(".")  # figure references removed, ends on a full sentence
    assert "a chest radiograph showed blunting of the right costophrenic angle." in c
    long_one, cut = T.clean_text("word " * 200)
    assert cut and len(long_one) <= T.MAX_CAP  # only an over-long first sentence is cut at a word


# ---------------------------------------------------------------- data
def test_nih_label_parsing():
    assert T.parse_nih("Cardiomegaly|Effusion") == (1, 1, 0, 0, 0, 0)
    assert T.parse_nih("No Finding") == (0,) * 6
    assert T.parse_nih("Nodule") == T.parse_nih("Mass") == (0, 0, 0, 0, 0, 1)
    assert T.parse_nih("Mass|Nodule") == (0, 0, 0, 0, 0, 1)
    with pytest.raises(KeyError):
        T.parse_nih("Hernia")  # not one of the six findings kept


def test_bundled_subsets():
    d = T.load_nih(CPU)
    n_tr, n_te = len(d["y_train"]), len(d["y_test"])
    assert d["x_train"].shape == (n_tr, T.IMG_SIZE, T.IMG_SIZE) and n_tr > 300 and n_te > 100
    assert d["y_train"].shape == (n_tr, 6) and int(d["y_train"].sum(1).max()) <= 2
    assert (d["y_train"].sum(0) >= 20).all()  # every finding present in training (pneumothorax is the rarest)
    assert d["sets_train"].max() < len(T.label_sets())
    torch.testing.assert_close(T.SET_Y[d["sets_train"]], d["y_train"])
    p = T.load_pmc(CPU)
    assert p["x_train"].shape[1:] == (T.IMG_SIZE, T.IMG_SIZE) and len(p["cap_train"]) == len(p["x_train"]) > 150
    assert len(p["cap_test"]) == len(p["x_test"]) > 30
    assert all(0 < len(c) <= T.MAX_CAP for c in p["cap_train"] + p["cap_test"])
    words = sorted(len(c.split()) for c in p["cap_train"])
    assert words[len(words) // 2] >= 25  # caption + the article's sentences: far longer than a caption


# ---------------------------------------------------------------- the rule graph
def test_training_graph_levels_and_critical_path():
    d = solve.build_dag(solve.Config())
    lev = dict(zip(d["names"], d["levels"].tolist()))
    assert lev["train_cnn"] == lev["train_lm"] == lev["train_monolith"] == 0
    assert lev["train_connector"] == 1
    assert lev["global_finetune"] == lev["train_router"] == 2
    assert lev["caption_finetune"] == 3 and lev["evaluate"] == 4
    assert dag.is_nilpotent(d["A"]) and bool((dag.walk_traces(d["A_rule"]) == 0).all())
    S = 2
    assert len(d["A"]) == 7 * S + 1  # every rule per seed, evaluate once
    sec = dict.fromkeys(d["names"], 1.0) | {"train_monolith": 2.5, "train_lm": 3.0}
    s = solve.schedule(d, sec)
    assert s["path"] == ["train_lm", "train_connector", "global_finetune", "caption_finetune", "evaluate"]
    assert s["makespan"] == pytest.approx(7.0) and s["serial"] == pytest.approx(sum(sec.values()))


# ---------------------------------------------------------------- the DAG edge as an MoE layer
def _models(seeds, d=32):
    return (T.init_cnn(seeds, 1, CPU), T.init_connector(seeds, 3, CPU, d), T.init_lm(seeds, 2, CPU, d, 2, 1))


def test_fixed_router_moe_equals_the_dag_edge():
    torch.manual_seed(0)
    w_cnn, w_con, w_lm = _models([0, 1])
    data = T.load_nih(CPU)
    tab = T.template_table()
    idx = torch.arange(12)
    x = data["x_train"][idx][None].expand(2, -1, -1, -1)
    tok = tab["tokens"][T.row_of(data["sets_train"][idx], idx % T.N_VARIANTS)][None].expand(2, -1, -1)
    z_dag = eq.stitched_logits(x, tok[..., :-1], w_cnn, w_con, w_lm)
    ids, is_img = eq.position_items(tok[..., :-1])
    G = eq.route_fixed(2, len(idx), is_img)
    assert torch.equal(G[0, 0, :, 0], (torch.arange(T.CTX - 1) < T.K).float())
    z_moe = eq.lm_logits(eq.moe_inputs(x, ids, G, w_cnn, w_con, w_lm, data["blank"]), w_lm)
    torch.testing.assert_close(z_moe, z_dag, rtol=0, atol=1e-6)
    params = T.leaves(w_cnn) + T.leaves(w_con) + T.leaves(w_lm)
    g_dag = torch.autograd.grad(z_dag.square().mean(), params, allow_unused=True)
    g_moe = torch.autograd.grad(z_moe.square().mean(), params, allow_unused=True)
    for a, b in zip(g_dag, g_moe):
        if a is None:
            assert b is None or b.abs().max() == 0
        else:
            torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)


def test_f1_and_text_metrics():
    true = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=torch.bool)
    pred = torch.tensor([[[1, 0], [1, 0], [1, 0], [0, 1]]], dtype=torch.bool)  # (M=1, N, F)
    micro, macro = eq.f1(pred, true)
    # class 0: tp 2, fp 1, fn 0 → 0.8; class 1: tp 1, fp 0, fn 1 → 2/3; micro tp 3 fp 1 fn 1 → 0.75
    assert micro.item() == pytest.approx(0.75) and macro.item() == pytest.approx((0.8 + 2 / 3) / 2)
    assert eq.bleu1("a b c d", "a b c d") == pytest.approx(1.0)
    assert eq.rouge_l("a b c d", "a x c d") == pytest.approx(0.75)
    assert eq.rouge_l("", "a b") == 0.0
    s = torch.tensor([0.9, 0.8, 0.3, 0.1])
    assert eq.auroc(s, torch.tensor([1, 1, 0, 0])).item() == pytest.approx(1.0)
    assert eq.auroc(s, torch.tensor([0, 1, 1, 0])).item() == pytest.approx(0.5)


# ---------------------------------------------------------------- independence on the model axis
def test_zero_cross_gradient():
    w_cnn, w_con, w_lm = _models([0, 1, 2])
    st = solve.make_store(small_cfg(seeds=[0, 1, 2]), CPU)
    rows = torch.randint(len(st["data"]["y_train"]), (3, 8), generator=torch.Generator().manual_seed(0))
    loss = solve.joint_loss(st, rows, w_cnn, w_con, w_lm)
    loss[1].backward()
    for p in T.leaves(w_cnn) + T.leaves(w_con) + T.leaves(w_lm):
        if p.grad is not None:
            assert p.grad[0].abs().max() == 0 and p.grad[2].abs().max() == 0


def test_batched_over_seeds_equals_per_seed():
    """Two seeds trained in one stack = each seed trained alone (Adam is elementwise, clipping is per model).
    Float32 reductions batched vs alone differ in summation order (and across BLAS builds): rtol 1e-4."""
    st2 = solve.make_store(small_cfg(seeds=[0, 1], steps_lm=5, steps_cnn=5), CPU)
    lm2, cnn2 = solve.rule_train_lm(st2), solve.rule_train_cnn(st2)
    for i, s in enumerate([0, 1]):
        st1 = solve.make_store(small_cfg(seeds=[s], steps_lm=5, steps_cnn=5), CPU)
        lm1, cnn1 = solve.rule_train_lm(st1), solve.rule_train_cnn(st1)
        torch.testing.assert_close(lm2["loss"][:, i], lm1["loss"][:, 0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(cnn2["loss"][:, i], cnn1["loss"][:, 0], rtol=1e-4, atol=1e-4)
        for a, b in zip(T.leaves(lm2["w"]), T.leaves(lm1["w"])):
            # Adam divides by √v: a rounding difference in a near-zero gradient can move a weight by up to ~lr
            torch.testing.assert_close(a[i], b[0], rtol=0, atol=0.2 * st1["cfg"].lr_lm)
    c = st2["cfg"]
    m = c.macs()
    assert c.steps_mono == round((5 * m["cnn"] + 5 * m["lm"] + 5 * m["joint"]) / m["joint"])


def test_connector_stage_changes_only_the_connector_and_captions_leave_the_cnn():
    st = solve.make_store(small_cfg(seeds=[0, 1]), CPU)
    for name in ("train_cnn", "train_lm", "train_connector", "global_finetune", "caption_finetune"):
        st[name] = solve.RULES[name](st)
    c = st["train_connector"]
    for a, b in zip(T.leaves(c["cnn"]), T.leaves(st["train_cnn"]["w"])):
        assert torch.equal(a, b.detach())
    for a, b in zip(T.leaves(c["lm"]), T.leaves(st["train_lm"]["w"])):
        assert torch.equal(a, b.detach())
    assert all(not torch.equal(a, b) for a, b in zip(T.leaves(c["con"]), T.leaves(c["con_init"])))
    cap, ft = st["caption_finetune"], st["global_finetune"]
    for a, b in zip(T.leaves(cap["cnn"]), T.leaves(ft["cnn"])):
        assert torch.equal(a, b.detach())  # stage 2 trains the connector and the LM only
    assert any(not torch.equal(a, b) for a, b in zip(T.leaves(cap["lm"]), T.leaves(ft["lm"])))


def test_greedy_writes_what_teacher_forcing_prefers():
    """Greedy decoding: each written letter is the argmax of the logits given the letters written so far."""
    st = solve.make_store(small_cfg(), CPU)
    w_cnn, w_con, w_lm = _models([0])
    x = st["data"]["x_test"][:3][None]
    out = solve.generate(x, w_cnn, w_con, w_lm, n_new=8)  # (1, 3, CTX) ids
    z = eq.stitched_logits(x, out[..., :-1], w_cnn, w_con, w_lm)
    pos = torch.arange(T.K, T.K + 8)
    written = out[..., pos + 1]
    ok = (z.argmax(-1)[..., pos] == written) | (written == T.STOI[T.PAD])  # padding only after an EOS
    assert bool(ok.all())


# ---------------------------------------------------------------- learning (seeded)
@pytest.fixture(scope="module")
def trained():
    cfg = solve.Config(seeds=[0], steps_router=0, steps_cap=0)
    st = solve.make_store(cfg, CPU)
    for name in ("train_cnn", "train_lm"):
        st[name] = solve.RULES[name](st)
    return st


@pytest.mark.slow
def test_cnn_beats_chance(trained):
    w, d = trained["train_cnn"]["w"], trained["data"]
    with torch.no_grad():
        p = eq.cnn_output(d["x_test"][None], w)[0]
    auc = torch.stack([eq.auroc(p[:, f], d["y_test"][:, f]) for f in range(6)])
    assert auc.mean() > 0.6  # chance 0.5; per-class AUROC averaged over the six findings


@pytest.mark.slow
def test_lm_reproduces_templates(trained):
    """Given each label header, the LM alone writes a well-formed template of exactly those labels."""
    w = trained["train_lm"]["w"]
    sets = torch.arange(len(T.label_sets()))
    tok = T.template_table()["tokens"][T.row_of(sets, torch.zeros_like(sets))][None]
    out = solve.generate_text(tok, w)
    texts = [T.report_text(r) for r in out[0]]
    ok = [T.template_labels(t) == tuple(T.SET_Y[i].int().tolist()) for i, t in enumerate(texts)]
    assert sum(ok) >= 0.9 * len(ok)


@pytest.mark.slow
def test_full_run(tmp_path):
    from pathlib import Path

    from problems._common.runtime import load_meta, resolve_params

    from . import main

    out = main.run(resolve_params(load_meta(Path(__file__).parent)), tmp_path)
    assert out.exists() and (tmp_path / "stages.csv").exists() and (tmp_path / "captions.csv").exists()
    res = main.LAST
    f1 = {k: float(v.mean()) for k, v in res["f1_micro"].items()}
    assert f1["cnn"] > 0.22 and f1["symbolic"] > 0.22  # random guessing at the training prevalence: about 0.16
    assert f1["finetune"] > 0.1 and f1["finetune"] >= f1["monolith"]  # the DAG writes findings, the monolith none
    assert float(res["wellformed"]["finetune"].mean()) > 0.8
    assert 0.0 < float(res["cap"]["caption"]["rouge_l"].mean()) < 1.0  # stage 2: weak but not empty
