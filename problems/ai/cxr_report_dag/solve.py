"""The training process as a tensor DAG: every rule of data/training.smk is ONE batched call over the seeds.

L0  train_cnn ∥ train_lm ∥ train_monolith   (separate pretraining on NIH labels and on template reports; the
                                             monolith baseline is just another rule)
L1  train_connector                          (CNN and LM frozen, only the connector learns, NIH image → report)
L2  global_finetune ∥ train_router           (everything end to end; or a learned router on the frozen experts)
L3  caption_finetune                         (stage 2: connector + LM on real PMC figure captions, CNN frozen)
L4  evaluate

The DAG algebra (Kronecker job graph, boolean levels, max-plus finish times, critical path) is imported from the
Circle of Willis workflow problem, the optimiser loop from problems/ai/stitch_llm_cnn_dag.
"""

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from problems.ai.stitch_llm_cnn_dag.solve import fit, rule_graph
from problems.medical.cow_workflow_dag import equation as dag
from problems.medical.cow_workflow_dag.solve import parse_snakefile

from . import equation as eq
from . import tensors as T


@dataclass
class Config:
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    batch: int = 32
    steps_cnn: int = 250
    steps_lm: int = 200
    steps_con: int = 120
    steps_ft: int = 60
    steps_router: int = 40
    steps_cap: int = 150
    d: int = 32
    heads: int = 4
    layers: int = 2
    lr: float = 3e-3  # CNN and router
    lr_lm: float = 1e-2
    lr_con: float = 3e-2
    lr_ft: float = 3e-4
    lr_cap: float = 1e-2
    batch_cap: int = 16  # stage-2 sequences are about six times longer
    lr_mono: float = 2e-3
    shift: int = 3  # random translation of the training images (pixels)
    n_test: int | None = None  # evaluate on the first n_test test images only (fast mode)

    def macs(self) -> dict:
        """Multiply-adds of one forward pass per example (backward ≈ 2× forward for every one, so ratios hold)."""
        d, t, v = self.d, T.CTX - 1, len(T.VOCAB)
        cnn = 60 * 60 * 8 * 25 + 28 * 28 * 16 * 72 + 12 * 12 * 16 * 144 + 576 * 64 + 64 * 6
        lm = t * (self.layers * (12 * d * d + 2 * t * d) + d * v)
        con = T.K * (6 * 64 + 64 * d)
        return {"cnn": cnn, "lm": lm, "joint": cnn + con + lm}

    @property
    def steps_mono(self) -> int:
        """Matched compute: the monolith's joint steps cost as many multiply-adds as the DAG pipeline's CNN, LM,
        connector and fine-tune steps together (the caption stage is not part of the stage-1 comparison)."""
        c = self.macs()
        work = self.steps_cnn * c["cnn"] + self.steps_lm * c["lm"] + (self.steps_con + self.steps_ft) * c["joint"]
        return round(work / c["joint"])


# ---------------------------------------------------------------- the rule graph (parsed with the CoW parser)
def build_dag(cfg: Config, text: str | None = None) -> dict:
    """Rules, A_rule, rule levels and the (J, J) job adjacency A = Σ A_rule ⊗ (Kronecker blocks over the seeds)."""
    rules = parse_snakefile(text if text is not None else T.TRAINING_GRAPH.read_text(encoding="utf-8"))
    A_rule, axes = rule_graph(rules)
    sizes = {"seed": len(cfg.seeds)}
    A, off = dag.job_adjacency(A_rule, axes, sizes)
    return {"rules": rules, "names": [r["name"] for r in rules], "A_rule": A_rule, "axes": axes, "sizes": sizes,
            "levels": dag.levels(A_rule), "A": A, "off": off}  # fmt: skip


def schedule(d: dict, seconds: dict) -> dict:
    """Max-plus finish times with the measured rule durations: with one worker per rule the makespan is the
    critical path; one worker runs the sum."""
    dur = torch.tensor([seconds[n] for n in d["names"]], dtype=torch.float64)
    t = dag.finish_times(d["A_rule"].double(), dur)
    path = dag.critical_path(d["A_rule"].double(), t, dur)
    return {"finish": t, "duration": dur, "path": [d["names"][i] for i in path], "makespan": float(t.max()),
            "serial": float(dur.sum())}  # fmt: skip


# ---------------------------------------------------------------- batches
def shifted(x: torch.Tensor, seeds: list[int], step: int, salt: int, shift: int) -> torch.Tensor:
    """Random integer translations of each image (M, B, H, W), zero padded, drawn per model from its own generator
    (so a model's augmentation never depends on which other models share the stack)."""
    if shift == 0:
        return x
    m, b, h, w = x.shape
    off = torch.stack([torch.randint(-shift, shift + 1, (b, 2), generator=torch.Generator().manual_seed(
        s * 104_729 + step * 31 + salt)) for s in seeds]).to(x.device)  # fmt: skip
    pad = F.pad(x, (shift, shift, shift, shift))
    iy = (torch.arange(h, device=x.device) + shift)[None, None, :] - off[..., 0:1]  # (M, B, H)
    ix = (torch.arange(w, device=x.device) + shift)[None, None, :] - off[..., 1:2]
    rows = pad.gather(2, iy[..., None].expand(m, b, h, w + 2 * shift))
    return rows.gather(3, ix[:, :, None, :].expand(m, b, h, w))


def joint_batch(st: dict, rows: torch.Tensor, variants: torch.Tensor | None = None, train: bool = True) -> dict:
    """NIH images rows (M, B) with their template reports (phrasing variant per example) → x, tokens, mask."""
    d, tab = st["data"], st["table"]
    sets = (d["sets_train"] if train else d["sets_test"])[rows]
    v = rows % T.N_VARIANTS if variants is None else variants
    tok = tab["tokens"][T.row_of(sets, v)]
    return {"x": (d["x_train"] if train else d["x_test"])[rows], "tok": tok, "mask": tab["mask"][T.row_of(sets, v)]}


def joint_loss(st: dict, rows, w_cnn, w_con, w_lm, variants=None, step: int = 0, salt: int = 0) -> torch.Tensor:
    """Mean −log p of the report letters given the image (the header is the soft prefix).  → (M,) nats"""
    bt = joint_batch(st, rows, variants)
    x = shifted(bt["x"], st["cfg"].seeds, step, salt, st["cfg"].shift) if salt else bt["x"]
    z = eq.stitched_logits(x, bt["tok"][..., :-1], w_cnn, w_con, w_lm)
    return eq.masked_ce(z, bt["tok"][..., 1:], bt["mask"])


def moe_logits(x, tok, w_cnn, w_con, w_lm, w_router, blank, soft: bool = False, hard: bool = False, q_img=None):
    """LM logits through the MoE input layer: fixed router (w_router None) or a learned one; hard = route then run
    (G one-hot, the chosen expert's output not scaled by p), as used for writing."""
    ids, is_img = eq.position_items(tok)
    m, n, _ = ids.shape
    if w_router is None:
        G = p = eq.route_fixed(m, n, is_img)
    else:
        G, p = eq.route_learned(eq.router_features(x, ids, is_img, blank), w_router, soft)
        G = (G > 0).to(p.dtype) if hard else G
    return eq.lm_logits(eq.moe_inputs(x, ids, G, w_cnn, w_con, w_lm, blank, q_img), w_lm), G, p


# ---------------------------------------------------------------- the rules: one batched call each
def rule_train_cnn(st: dict) -> dict:
    cfg, d, S = st["cfg"], st["data"], len(st["cfg"].seeds)
    w = T.init_cnn(cfg.seeds, 1, st["dev"])
    rows = T.draws(cfg.seeds, [len(d["y_train"])] * S, cfg.steps_cnn, cfg.batch, 1, st["dev"])
    pos = d["y_train"].mean(0)
    pw = ((1 - pos) / pos).clamp(max=8.0)

    def loss(t):
        x = shifted(d["x_train"][rows[t]], cfg.seeds, t, 1, cfg.shift)
        return eq.bce(eq.cnn_logits(eq.cnn_features(x, w), w), d["y_train"][rows[t]], pw)

    return {"w": w, "loss": fit(T.leaves(w), loss, cfg.steps_cnn, cfg.lr, "train_cnn")}


def rule_train_lm(st: dict) -> dict:
    """The LM alone on every template report, the true label header written as text in front."""
    cfg, tab, S = st["cfg"], st["table"], len(st["cfg"].seeds)
    w = T.init_lm(cfg.seeds, 2, st["dev"], cfg.d, cfg.heads, cfg.layers)
    rows = T.draws(cfg.seeds, [len(tab["tokens"])] * S, cfg.steps_lm, cfg.batch, 2, st["dev"])

    def loss(t):
        x = tab["tokens"][rows[t]]
        return eq.masked_ce(eq.lm_logits(eq.embed(x[..., :-1], w), w), x[..., 1:], tab["mask"][rows[t]])

    return {"w": w, "loss": fit(T.leaves(w), loss, cfg.steps_lm, cfg.lr_lm, "train_lm")}


def _joint_draws(st: dict, steps: int, salt: int):
    cfg, n = st["cfg"], len(st["data"]["y_train"])
    S = len(cfg.seeds)
    rows = T.draws(cfg.seeds, [n] * S, steps, cfg.batch, salt, st["dev"])
    var = T.draws(cfg.seeds, [T.N_VARIANTS] * S, steps, cfg.batch, salt + 100, st["dev"])
    return rows, var


def rule_train_monolith(st: dict) -> dict:
    """The non-DAG baseline: the same CNN + connector + LM, random init, trained end to end on image → report only."""
    cfg, dev = st["cfg"], st["dev"]
    w_cnn, w_con = T.init_cnn(cfg.seeds, 11, dev), T.init_connector(cfg.seeds, 13, dev, cfg.d)
    w_lm = T.init_lm(cfg.seeds, 12, dev, cfg.d, cfg.heads, cfg.layers)
    rows, var = _joint_draws(st, cfg.steps_mono, 5)
    params = T.leaves(w_cnn) + T.leaves(w_con) + T.leaves(w_lm)
    hist = fit(params, lambda t: joint_loss(st, rows[t], w_cnn, w_con, w_lm, var[t], t, 5), cfg.steps_mono,
               cfg.lr_mono, "train_monolith")  # fmt: skip
    return {"cnn": w_cnn, "con": w_con, "lm": w_lm, "loss": hist}


def rule_train_connector(st: dict) -> dict:
    """Freeze the pretrained CNN and LM; only the connector learns (NIH image → template report)."""
    cfg, dev = st["cfg"], st["dev"]
    w_cnn, w_lm = T.clone(st["train_cnn"]["w"], False), T.clone(st["train_lm"]["w"], False)
    w_con = T.init_connector(cfg.seeds, 3, dev, cfg.d)
    init = T.clone(w_con, False)
    rows, var = _joint_draws(st, cfg.steps_con, 3)
    hist = fit(T.leaves(w_con), lambda t: joint_loss(st, rows[t], w_cnn, w_con, w_lm, var[t], t, 3), cfg.steps_con,
               cfg.lr_con, "train_connector")  # fmt: skip
    return {"cnn": w_cnn, "con": w_con, "lm": w_lm, "con_init": init, "loss": hist}


def rule_global_finetune(st: dict) -> dict:
    """Unfreeze everything and train end to end through the DAG edge, gently."""
    cfg, c = st["cfg"], st["train_connector"]
    w_cnn, w_con, w_lm = (T.clone(c[k], True) for k in ("cnn", "con", "lm"))
    rows, var = _joint_draws(st, cfg.steps_ft, 4)
    params = T.leaves(w_cnn) + T.leaves(w_con) + T.leaves(w_lm)
    hist = fit(params, lambda t: joint_loss(st, rows[t], w_cnn, w_con, w_lm, var[t], t, 4), cfg.steps_ft, cfg.lr_ft,
               "global_finetune")  # fmt: skip
    return {"cnn": w_cnn, "con": w_con, "lm": w_lm, "loss": hist}


def rule_train_router(st: dict) -> dict:
    """Replace the fixed modality routing by a learned linear router (experts frozen), trained as a dense mixture
    G = p (both experts everywhere, so the gradient compares them) and used top-1: does it find the routing again?"""
    cfg, c, blank = st["cfg"], st["train_connector"], st["data"]["blank"]
    w_r = T.init_router(cfg.seeds, 4, st["dev"])
    rows, var = _joint_draws(st, cfg.steps_router, 6)

    def loss(t):
        bt = joint_batch(st, rows[t], var[t])
        z, _, _ = moe_logits(bt["x"], bt["tok"][..., :-1], c["cnn"], c["con"], c["lm"], w_r, blank, soft=True)
        return eq.masked_ce(z, bt["tok"][..., 1:], bt["mask"])

    return {"router": w_r, "loss": fit(T.leaves(w_r), loss, cfg.steps_router, cfg.lr, "train_router")}


def rule_caption_finetune(st: dict) -> dict:
    """Stage 2: the stitched model (after the global fine-tune) learns real figure captions; CNN frozen."""
    cfg, ft, p = st["cfg"], st["global_finetune"], st["pmc"]
    w_cnn = T.clone(ft["cnn"], False)
    w_con, w_lm = T.clone(ft["con"], True), T.clone(ft["lm"], True)
    rows = T.draws(cfg.seeds, [len(p["cap_train"])] * len(cfg.seeds), cfg.steps_cap, cfg.batch_cap, 8, st["dev"])
    tok, mask = p["tok_train"]["tokens"], p["tok_train"]["mask"]

    def loss(t):
        x = shifted(p["x_train"][rows[t]], cfg.seeds, t, 8, cfg.shift)
        z = eq.stitched_logits(x, tok[rows[t]][..., :-1], w_cnn, w_con, w_lm)
        return eq.masked_ce(z, tok[rows[t]][..., 1:], mask[rows[t]])

    hist = fit(T.leaves(w_con) + T.leaves(w_lm), loss, cfg.steps_cap, cfg.lr_cap, "caption_finetune")
    return {"cnn": w_cnn, "con": w_con, "lm": w_lm, "loss": hist}


# ---------------------------------------------------------------- greedy decoding
def blank_prompt(m: int, n: int, headers: list[list[str]] | None = None, device=None) -> torch.Tensor:
    """(M, N, CTX) ids: a header ('------', or headers[m][n]) + BOS, then padding to be written over."""
    hs = headers or [["-" * T.K] * n] * m
    return torch.tensor([[T.encode((h + T.BOS).ljust(T.CTX, T.PAD)) for h in row] for row in hs], device=device)


@torch.no_grad()
def _greedy(logits_fn, tok: torch.Tensor, n_new: int) -> torch.Tensor:
    """Write n_new letters after BOS, each the argmax given what is written so far (causal: only the prefix is fed);
    stop early once every row has written EOS (later letters stay padding)."""
    done = torch.zeros(tok.shape[:-1], dtype=torch.bool, device=tok.device)
    for k in range(n_new):
        p = T.K + 1 + k
        nxt = logits_fn(tok[..., :p])[..., p - 1, :].argmax(-1)
        tok[..., p] = torch.where(done, torch.full_like(nxt, T.STOI[T.PAD]), nxt)
        done |= nxt == T.STOI[T.EOS]
        if bool(done.all()):
            break
    return tok


def _prompt_ids(m: int, n: int, device) -> torch.Tensor:
    return torch.tensor(T.encode("-" * T.K + T.BOS), device=device).expand(m, n, -1)


def generate(x, w_cnn, w_con, w_lm, n_new: int = T.MAX_TXT + 1) -> torch.Tensor:
    """The stitched model writes from the image alone (soft header from the CNN, then BOS), with a key/value cache.
    x (M, N, 64, 64) → ids (M, N, K + 1 + n_new)."""
    m, n = x.shape[:2]
    bos = eq.embed(torch.full((m, n, 1), T.STOI[T.BOS], device=x.device), w_lm)
    gen = eq.greedy_cached(torch.cat([eq.soft_prefix(x, w_cnn, w_con), bos], 2), w_lm, n_new)
    return torch.cat([_prompt_ids(m, n, x.device), gen], -1)


def generate_moe(x, w_cnn, w_con, w_lm, w_router, blank, n_new: int = T.MAX_TXT + 1) -> torch.Tensor:
    """Through the MoE input layer (learned router, route then run): every letter re-routes the whole prefix, so no
    cache here."""
    tok = blank_prompt(x.shape[0], x.shape[1], device=x.device)
    q = eq.cnn_output(x, w_cnn)
    return _greedy(lambda t: moe_logits(x, t, w_cnn, w_con, w_lm, w_router, blank, hard=True, q_img=q)[0], tok, n_new)


def generate_text(tok: torch.Tensor, w_lm, n_new: int = T.MAX_TXT + 1) -> torch.Tensor:
    """The LM alone, given a written header + BOS (the first K + 1 ids of tok).  → ids (M, N, K + 1 + n_new)."""
    gen = eq.greedy_cached(eq.embed(tok[..., : T.K + 1], w_lm), w_lm, n_new)
    return torch.cat([tok[..., : T.K + 1], gen], -1)


def texts(tok: torch.Tensor) -> list[list[str]]:
    return [[T.report_text(r) for r in m] for m in tok]


# ---------------------------------------------------------------- evaluation
def report_scores(txt: list[list[str]], y_true: torch.Tensor, sets_true: torch.Tensor) -> dict:
    """Parse generated reports back into labels: micro and macro F1, share of well-formed templates, and share of
    exact templates of the true label set.  Per model (M,)."""
    pred = torch.tensor([[T.parse_report(s) for s in m] for m in txt], dtype=torch.bool)
    micro, macro = eq.f1(pred, y_true.bool().cpu())
    lab = [[T.template_labels(s) for s in m] for m in txt]
    true = [T.SETS[int(i)] for i in sets_true]
    wf = torch.tensor([[a is not None for a in m] for m in lab]).float().mean(1)
    ex = torch.tensor([[a == b for a, b in zip(m, true)] for m in lab]).float().mean(1)
    return {"micro": micro, "macro": macro, "wellformed": wf, "exact": ex, "pred": pred}


def caption_scores(gen: list[list[str]], ref: list[str]) -> dict:
    """BLEU-1, ROUGE-L and finding-keyword recall/precision of generated captions against the true ones (M,)."""
    out = {k: [] for k in ("bleu1", "rouge_l", "kw_recall", "kw_precision")}
    yt = torch.tensor([T.parse_report(r) for r in ref], dtype=torch.float32)
    for m in gen:
        out["bleu1"].append(sum(eq.bleu1(h, r) for h, r in zip(m, ref)) / len(ref))
        out["rouge_l"].append(sum(eq.rouge_l(h, r) for h, r in zip(m, ref)) / len(ref))
        yp = torch.tensor([T.parse_report(h) for h in m], dtype=torch.float32)
        tp = (yp * yt).sum()
        out["kw_recall"].append(float(tp / yt.sum().clamp(min=1)))
        out["kw_precision"].append(float(tp / yp.sum().clamp(min=1)))
    return {k: torch.tensor(v) for k, v in out.items()}


def constant_caption(captions: list[str], n: int = 40) -> str:
    """A blind baseline: of n training texts drawn at random (seed 0), the one with the highest mean ROUGE-L against
    n other training texts."""
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(captions), generator=g)[: 2 * n].tolist()
    cand, ref = [captions[i] for i in idx[:n]], [captions[i] for i in idx[n:]]
    return max(cand, key=lambda c: sum(eq.rouge_l(c, r) for r in ref))


@torch.no_grad()
def rule_evaluate(st: dict) -> dict:
    """Every stage writes a report for every NIH test image; the stage-1 stacks are joined on the model axis and
    decoded in ONE batched call. Then stage 2: captions for the held-out PMC figures."""
    cfg, dev = st["cfg"], st["dev"]
    d, p = dict(st["data"]), dict(st["pmc"])
    if cfg.n_test:
        for k in ("x_test", "y_test", "sets_test"):
            d[k] = d[k][: cfg.n_test]
        p["x_test"], p["cap_test"] = p["x_test"][: cfg.n_test], p["cap_test"][: cfg.n_test]
    S = len(cfg.seeds)
    con, ft, mono = st["train_connector"], st["global_finetune"], st["train_monolith"]
    stages = {"connector": (con["cnn"], con["con"], con["lm"]), "finetune": (ft["cnn"], ft["con"], ft["lm"]),
              "monolith": (mono["cnn"], mono["con"], mono["lm"])}  # fmt: skip
    w = [T.cat_models([v[i] for v in stages.values()]) for i in range(3)]
    xt = d["x_test"][None].expand(S * len(stages), -1, -1, -1)
    out = generate(xt, *w)
    gen = {k: out[i * S : (i + 1) * S] for i, k in enumerate(stages)}
    gen["router"] = generate_moe(xt[:S], con["cnn"], con["con"], con["lm"], st["train_router"]["router"], d["blank"])
    # unimodal: the CNN alone (threshold 0.5) and its labels written as a text header for the LM alone (symbolic)
    q = eq.cnn_output(xt[:S], st["train_cnn"]["w"])  # (S, N, 6)
    cnn_pred = q > 0.5
    heads = [[T.header(tuple(int(v) for v in r)) for r in m] for m in cnn_pred.tolist()]
    gen["symbolic"] = generate_text(blank_prompt(S, len(d["y_test"]), heads, dev), st["train_lm"]["w"])
    txt = {k: texts(v) for k, v in gen.items()}
    rep = {k: report_scores(v, d["y_test"], d["sets_test"]) for k, v in txt.items()}
    cnn_micro, cnn_macro = eq.f1(cnn_pred.cpu(), d["y_test"].bool().cpu())
    auc = torch.stack([torch.stack([eq.auroc(q[m, :, f].cpu(), d["y_test"][:, f].cpu()) for f in range(6)])
                       for m in range(S)])  # fmt: skip
    # random guessing at the training prevalence: expected counts → expected micro F1
    pt, pi = d["y_train"].mean(0).cpu(), d["y_test"].mean(0).cpu()
    tp, fp, fn = (pt * pi).sum(), (pt * (1 - pi)).sum(), ((1 - pt) * pi).sum()
    chance = float(2 * tp / (2 * tp + fp + fn))
    # the learned router: share of positions sent to the modality's expert, and its routing on the written reports
    tok0 = gen["router"][..., :-1]
    ids, is_img = eq.position_items(tok0)
    _, G, prob = moe_logits(xt[:S], tok0, con["cnn"], con["con"], con["lm"], st["train_router"]["router"], d["blank"])
    route_acc = ((G[..., 0] > 0) == is_img).float().mean((1, 2)).cpu()

    # stage 2: captions of held-out PMC figures (before and after the caption stage; with shuffled images)
    n_p = len(p["cap_test"])
    perm = torch.roll(torch.arange(n_p, device=dev), 1)
    cap = st["caption_finetune"]
    xp = p["x_test"][None].expand(S, -1, -1, -1)
    w2 = [T.cat_models([a[k], b[k], b[k]]) for k, a, b in zip(("cnn", "con", "lm"), (ft,) * 3, (cap,) * 3)]
    out2 = generate(torch.cat([xp, xp, xp[:, perm]]), *w2, n_new=T.MAX_CAP + 1)
    cgen = {k: out2[i * S : (i + 1) * S] for i, k in enumerate(("finetune", "caption", "caption_shuffled"))}
    ctxt = {k: texts(v) for k, v in cgen.items()}
    const = constant_caption(p["cap_train"])
    ctxt["constant"] = [[const] * n_p] * S
    cscore = {k: caption_scores(v, p["cap_test"]) for k, v in ctxt.items()}
    # held-out cross-entropy of the true texts (teacher forcing, nats per letter): does the image help at all?
    tok = st["pmc"]["tok_test"]["tokens"][:n_p][None].expand(3 * S, -1, -1)
    msk = st["pmc"]["tok_test"]["mask"][:n_p][None].expand(3 * S, -1, -1)
    x3, num, den = torch.cat([xp, xp, xp[:, perm]]), 0.0, 0.0
    for lo in range(0, n_p, 8):  # chunks: the (M, N, H, T, T) attention maps of 367 letters are large
        sl = slice(lo, lo + 8)
        n_mask = msk[:, sl].sum((1, 2))
        num = num + n_mask * eq.masked_ce(eq.stitched_logits(x3[:, sl], tok[:, sl, :-1], *w2), tok[:, sl, 1:],
                                          msk[:, sl])  # fmt: skip
        den = den + n_mask
    ce = num / den
    for i, k in enumerate(("finetune", "caption", "caption_shuffled")):
        cscore[k]["nats"] = ce[i * S : (i + 1) * S].cpu()
    cscore["constant"]["nats"] = torch.full((S,), float("nan"))
    return {"txt": txt, "rep": rep, "cnn_micro": cnn_micro, "cnn_macro": cnn_macro, "auc": auc, "q": q.cpu(),
            "chance": chance, "route_acc": route_acc, "route_p": prob[..., 0].cpu(), "cap_txt": ctxt, "cap": cscore,
            "constant": const}  # fmt: skip


RULES = {"train_cnn": rule_train_cnn, "train_lm": rule_train_lm, "train_monolith": rule_train_monolith,
         "train_connector": rule_train_connector, "global_finetune": rule_global_finetune,
         "train_router": rule_train_router, "caption_finetune": rule_caption_finetune,
         "evaluate": rule_evaluate}  # fmt: skip


def make_store(cfg: Config, dev: torch.device) -> dict:
    """Data and tables every rule reads."""
    return {"cfg": cfg, "dev": dev, "data": T.load_nih(dev), "pmc": T.load_pmc(dev),
            "table": {k: v.to(dev) for k, v in T.template_table().items()}}  # fmt: skip


def run_engine(cfg: Config, dev: torch.device, d: dict | None = None) -> tuple[dict, dict, dict]:
    """Run the training DAG: rules level by level (levels of A_rule), each one batched call over the seeds."""
    d = d or build_dag(cfg)
    st = make_store(cfg, dev)
    seconds = {}
    for k in range(int(d["levels"].max()) + 1):
        for r in (d["levels"] == k).nonzero().flatten().tolist():
            name = d["names"][r]
            t0 = time.perf_counter()
            st[name] = RULES[name](st)
            seconds[name] = time.perf_counter() - t0
    return st, seconds, d
