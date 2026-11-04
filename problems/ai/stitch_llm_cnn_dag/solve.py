"""The training process as a tensor DAG: every rule of data/training.smk is ONE batched call over its models.

L0  train_cnn ∥ train_lm ∥ train_monolith   (separate pretraining; the monolith baseline is just another rule)
L1  train_connector                          (CNN and LM frozen, only the connector learns)
L2  global_finetune ∥ train_router ∥ train_router_soft   (all weights end to end; or a learned MoE router on the
                                                          frozen experts, Switch top-1 or trained dense)
L3  evaluate

The DAG algebra (Kronecker job graph, boolean levels, max-plus finish times, critical path) is imported from the
Circle of Willis workflow problem, problems/medical/cow_workflow_dag.
"""

import math
import re
import time
from dataclasses import dataclass, field

import torch

from problems._common.progress import steps as progress_steps
from problems.medical.cow_workflow_dag import equation as dag
from problems.medical.cow_workflow_dag.solve import parse_snakefile

from . import equation as eq
from . import tensors as T


@dataclass
class Config:
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    sizes: list[int] = field(default_factory=lambda: [100, 300, 1000, 3000])  # joint training examples n
    batch: int = 32
    steps_cnn: int = 150
    steps_lm: int = 600
    steps_con: int = 150
    steps_ft: int = 80
    steps_router: int = 60
    d: int = 32
    heads: int = 4
    layers: int = 2
    lr: float = 3e-3  # CNN and router
    lr_lm: float = 1e-2
    lr_con: float = 3e-2
    lr_ft: float = 3e-4  # global fine-tune: a gentle nudge of everything
    lr_mono: float = 2e-3
    alpha: float = 0.01  # Switch load-balancing coefficient
    n_test: int = 1000

    def macs(self) -> dict:
        """Multiply-adds of one forward pass per example (backward ≈ 2× forward for every one, so ratios hold)."""
        d, t, v = self.d, T.CTX - 1, len(T.VOCAB)
        cnn = 6 * 24 * 24 * 25 + 12 * 8 * 8 * 150 + 192 * 32 + 32 * 10
        lm = t * (self.layers * (12 * d * d + 2 * t * d) + d * v)  # Q, K, V, O, MLP 4d; scores and mix; read-out
        con = T.K * (10 * 64 + 64 * d)
        return {"cnn": cnn, "lm": lm, "joint": cnn + con + lm}

    @property
    def steps_mono(self) -> int:
        """Matched compute: the monolith's joint steps (same batch) cost as many multiply-adds as the DAG pipeline's
        CNN, LM, connector and fine-tune steps together, all counted per model."""
        c = self.macs()
        work = self.steps_cnn * c["cnn"] + self.steps_lm * c["lm"] + (self.steps_con + self.steps_ft) * c["joint"]
        return round(work / c["joint"])


# ---------------------------------------------------------------- the rule graph (parsed with the CoW parser)
def rule_graph(rules: list[dict]) -> tuple[torch.Tensor, list[tuple]]:
    """A_rule (R, R): rule s reads a file pattern rule r writes; and each rule's wildcards (seed, n)."""
    A = torch.zeros(len(rules), len(rules))
    for s, rs in enumerate(rules):
        for pat, _ in rs["inputs"]:
            for r, rr in enumerate(rules):
                if pat in rr["outputs"]:
                    A[s, r] = 1
    axes = [tuple(a for a in ("seed", "n") if any("{" + a + "}" in p for p in r["outputs"])) for r in rules]
    return A, axes


def build_dag(cfg: Config, text: str | None = None) -> dict:
    """Rules, A_rule, rule levels, and the (J, J) job adjacency A = Σ A_rule ⊗ (Kronecker blocks over seed, n)."""
    rules = parse_snakefile(text if text is not None else T.TRAINING_GRAPH.read_text(encoding="utf-8"))
    A_rule, axes = rule_graph(rules)
    sizes = {"seed": len(cfg.seeds), "n": len(cfg.sizes)}
    A, off = dag.job_adjacency(A_rule, axes, sizes)
    return {"rules": rules, "names": [r["name"] for r in rules], "A_rule": A_rule, "axes": axes, "sizes": sizes,
            "levels": dag.levels(A_rule), "A": A, "off": off, "job_levels": dag.levels(A)}  # fmt: skip


def schedule(d: dict, seconds: dict) -> dict:
    """Max-plus finish times on the rule graph with the measured rule durations: with one worker per rule, the
    makespan is the critical path; one worker runs the sum."""
    dur = torch.tensor([seconds[n] for n in d["names"]], dtype=torch.float64)
    t = dag.finish_times(d["A_rule"].double(), dur)
    path = dag.critical_path(d["A_rule"].double(), t, dur)
    return {"finish": t, "duration": dur, "path": [d["names"][i] for i in path], "makespan": float(t.max()),
            "serial": float(dur.sum())}  # fmt: skip


# ---------------------------------------------------------------- training helpers
def clip_per_model(params: list[torch.Tensor], max_norm: float) -> None:
    """Scale model m's gradient to norm ≤ max_norm, model by model (a global clip would couple the models)."""
    params = [p for p in params if p.grad is not None]  # e.g. the CNN's digit head is unused in the stitched model
    sq = sum(p.grad.pow(2).flatten(1).sum(1) for p in params)
    scale = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for p in params:
        p.grad.mul_(scale.view(-1, *[1] * (p.dim() - 1)))


def fit(params: list[torch.Tensor], loss_fn, n_steps: int, lr: float, label: str) -> torch.Tensor:
    """AdamW (elementwise, so models stay independent), warm-up + cosine decay, per-model clipping.
    loss_fn(step) → (M,) losses. Returns the loss history (steps, M)."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, min(30, n_steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / max(1, n_steps))))
    )
    hist = []
    for t in progress_steps(n_steps, label):
        loss = loss_fn(t)
        opt.zero_grad()
        loss.sum().backward()  # model m's loss reaches only model m's weights
        clip_per_model(params, 1.0)
        opt.step()
        sched.step()
        hist.append(loss.detach().cpu())
    return torch.stack(hist) if hist else torch.zeros(0, 1)


def joint_batch(data: dict, img_idx: torch.Tensor, op: torch.Tensor, b: torch.Tensor, train: bool = True) -> dict:
    """Joint examples: image idx (M, B) with prompt op, b → images, LM inputs, targets and masks (M, B, …).
    The operand word's letters are in `tokens` but sit under the soft prefix (positions < K) and are never targets."""
    x = (data["x_train"] if train else data["x_test"])[img_idx]
    a = (data["y_train"] if train else data["y_test"])[img_idx]
    tab = data["table"]
    tok = tab["tokens"][a, op, b]
    j = torch.arange(T.CTX - 1, device=tok.device)
    return {"x": x, "inp": tok[..., :-1], "tgt": tok[..., 1:], "mask": tab["text_mask"][a, op, b] & (j >= T.K - 1),
            "ans": tab["answer_mask"][a, op, b], "a": a, "op": op, "b": b}  # fmt: skip


def joint_draw(store: dict, rows: torch.Tensor) -> dict:
    """Rows (M, B) of each model's own joint set → a joint batch."""
    pool, si = store["pool"], store["seed_of"]
    return joint_batch(store["data"], pool["img"][si[:, None], rows], pool["op"][si[:, None], rows],
                       pool["b"][si[:, None], rows])  # fmt: skip


def task_loss(z: torch.Tensor, bt: dict) -> torch.Tensor:
    """Mean −log p over the text targets plus the same over the answer letters (the hard part).  → (M,)"""
    return eq.masked_ce(z, bt["tgt"], bt["mask"]) + eq.masked_ce(z, bt["tgt"], bt["ans"])


def joint_loss(store: dict, rows: torch.Tensor, w_cnn, w_con, w_lm) -> torch.Tensor:
    bt = joint_draw(store, rows)
    return task_loss(eq.stitched_logits(bt["x"], bt["inp"], w_cnn, w_con, w_lm), bt)


def moe_forward(bt: dict, w_cnn, w_con, w_lm, w_router, blank, soft: bool = False):
    """LM logits through the MoE input layer with a learned router (or the fixed one when w_router is None)."""
    ids, is_img = eq.position_items(bt["inp"])
    m, n, _ = ids.shape
    if w_router is None:
        G = eq.route_fixed(m, n, is_img)
        p = G
    else:
        G, p = eq.route_learned(eq.router_features(bt["x"], ids, is_img, blank), w_router, soft)
    return eq.lm_logits(eq.moe_inputs(bt["x"], ids, G, w_cnn, w_con, w_lm, blank), w_lm), G, p


# ---------------------------------------------------------------- the rules: one batched call each
def rule_train_cnn(st: dict) -> dict:
    cfg, data, S = st["cfg"], st["data"], len(st["cfg"].seeds)
    w = T.init_cnn(cfg.seeds, 1, st["dev"])
    rows = T.draws(cfg.seeds, [len(data["y_train"])] * S, cfg.steps_cnn, cfg.batch, 1, st["dev"])

    def loss(t):
        logits = eq.cnn_logits(eq.cnn_features(data["x_train"][rows[t]], w), w)
        return torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), data["y_train"][rows[t]], reduction="none"
        ).mean(1)

    return {"w": w, "loss": fit(T.leaves(w), loss, cfg.steps_cnn, cfg.lr, "train_cnn")}


def rule_train_lm(st: dict) -> dict:
    cfg, tab, S = st["cfg"], st["data"]["table"], len(st["cfg"].seeds)
    w = T.init_lm(cfg.seeds, 2, st["dev"], cfg.d, cfg.heads, cfg.layers)
    tok, msk = tab["tokens"].reshape(300, T.CTX), tab["text_mask"].reshape(300, T.CTX - 1)
    ans = tab["answer_mask"].reshape(300, T.CTX - 1)
    rows = T.draws(cfg.seeds, [300] * S, cfg.steps_lm, cfg.batch, 2, st["dev"])

    def loss(t):
        x = tok[rows[t]]
        return task_loss(eq.lm_logits(eq.embed(x[..., :-1], w), w), {"tgt": x[..., 1:], "mask": msk[rows[t]],
                                                                    "ans": ans[rows[t]]})  # fmt: skip

    return {"w": w, "loss": fit(T.leaves(w), loss, cfg.steps_lm, cfg.lr_lm, "train_lm")}


def rule_train_monolith(st: dict) -> dict:
    """The non-DAG baseline: the same CNN + connector + LM, random init, trained end to end on joint data only."""
    cfg, sm, dev = st["cfg"], st["seeds_mn"], st["dev"]
    w_cnn, w_con = T.init_cnn(sm, 11, dev), T.init_connector(sm, 13, dev, cfg.d)
    w_lm = T.init_lm(sm, 12, dev, cfg.d, cfg.heads, cfg.layers)
    rows = T.draws(sm, st["n_mn"], cfg.steps_mono, cfg.batch, 5, dev)
    params = T.leaves(w_cnn) + T.leaves(w_con) + T.leaves(w_lm)
    hist = fit(params, lambda t: joint_loss(st, rows[t], w_cnn, w_con, w_lm), cfg.steps_mono, cfg.lr_mono,
               "train_monolith")  # fmt: skip
    return {"cnn": w_cnn, "con": w_con, "lm": w_lm, "loss": hist}


def rule_train_connector(st: dict) -> dict:
    """Freeze the pretrained CNN and LM (broadcast seed → (seed, n)); only the connector learns."""
    cfg, sm, dev, N = st["cfg"], st["seeds_mn"], st["dev"], len(st["cfg"].sizes)
    w_cnn = T.clone(T.expand(st["train_cnn"]["w"], N), False)
    w_lm = T.clone(T.expand(st["train_lm"]["w"], N), False)
    w_con = T.init_connector(sm, 3, dev, cfg.d)
    init = T.clone(w_con, False)
    rows = T.draws(sm, st["n_mn"], cfg.steps_con, cfg.batch, 3, dev)
    hist = fit(T.leaves(w_con), lambda t: joint_loss(st, rows[t], w_cnn, w_con, w_lm), cfg.steps_con, cfg.lr_con,
               "train_connector")  # fmt: skip
    return {"cnn": w_cnn, "con": w_con, "lm": w_lm, "con_init": init, "loss": hist}


def rule_global_finetune(st: dict) -> dict:
    """Unfreeze everything and train end to end through the DAG edge, gently."""
    cfg, sm, dev, c = st["cfg"], st["seeds_mn"], st["dev"], st["train_connector"]
    w_cnn, w_con, w_lm = (T.clone(c[k], True) for k in ("cnn", "con", "lm"))
    rows = T.draws(sm, st["n_mn"], cfg.steps_ft, cfg.batch, 4, dev)
    params = T.leaves(w_cnn) + T.leaves(w_con) + T.leaves(w_lm)
    hist = fit(params, lambda t: joint_loss(st, rows[t], w_cnn, w_con, w_lm), cfg.steps_ft, cfg.lr_ft,
               "global_finetune")  # fmt: skip
    return {"cnn": w_cnn, "con": w_con, "lm": w_lm, "loss": hist}


def rule_train_router(st: dict, soft: bool = False) -> dict:
    """Replace the fixed modality routing by a learned router (experts frozen): does it find it again?
    Switch top-1 (default): only the chosen expert runs, its output scaled by p. soft: train with the dense mixture
    G = p (both experts everywhere, so the gradient compares them), evaluate top-1 like the Switch router."""
    cfg, sm, dev, c = st["cfg"], st["seeds_mn"], st["dev"], st["train_connector"]
    w_r = T.init_router(sm, 4, dev)
    rows = T.draws(sm, st["n_mn"], cfg.steps_router, cfg.batch, 7 if soft else 6, dev)
    blank = st["data"]["blank"]

    def loss(t):
        bt = joint_draw(st, rows[t])
        logits, G, p = moe_forward(bt, c["cnn"], c["con"], c["lm"], w_r, blank, soft)
        return task_loss(logits, bt) + cfg.alpha * eq.balance_loss(G, p)

    name = "train_router_soft" if soft else "train_router"
    return {"router": w_r, "loss": fit(T.leaves(w_r), loss, cfg.steps_router, cfg.lr, name)}


@torch.no_grad()
def rule_evaluate(st: dict) -> dict:
    """Every stage on the same 1000 joint test examples, plus the unimodal checks."""
    cfg, data, dev = st["cfg"], st["data"], st["dev"]
    S, N, M = len(cfg.seeds), len(cfg.sizes), len(st["seeds_mn"])
    n_test = min(cfg.n_test, len(data["y_test"]))
    test = T.joint_test(n_test, dev)
    con, ft, mono = st["train_connector"], st["global_finetune"], st["train_monolith"]
    blank = data["blank"]
    routers = {"router": st["train_router"]["router"], "router_soft": st["train_router_soft"]["router"]}
    acc = {k: torch.zeros(M) for k in ("connector_init", "connector", "finetune", "router", "router_soft", "monolith")}
    route_acc = {k: torch.zeros(M) for k in routers}
    symbolic, cnn_acc = torch.zeros(S), torch.zeros(S)
    w_cnn1, w_lm1 = st["train_cnn"]["w"], st["train_lm"]["w"]
    for lo in range(0, n_test, 250):  # chunks of the test set
        sl = torch.arange(lo, min(lo + 250, n_test), device=dev)
        frac = len(sl) / n_test
        bt = joint_batch(data, sl[None].expand(M, -1), test["op"][sl][None].expand(M, -1),
                         test["b"][sl][None].expand(M, -1), train=False)  # fmt: skip

        def score(w_cnn, w_con, w_lm, bt=bt):
            return eq.exact(eq.stitched_logits(bt["x"], bt["inp"], w_cnn, w_con, w_lm), bt["tgt"], bt["ans"])

        acc["connector_init"] += frac * score(con["cnn"], con["con_init"], con["lm"]).float().mean(1).cpu()
        acc["connector"] += frac * score(con["cnn"], con["con"], con["lm"]).float().mean(1).cpu()
        acc["finetune"] += frac * score(ft["cnn"], ft["con"], ft["lm"]).float().mean(1).cpu()
        acc["monolith"] += frac * score(mono["cnn"], mono["con"], mono["lm"]).float().mean(1).cpu()
        is_img = torch.arange(T.CTX - 1, device=dev) < T.K
        for k, w_r in routers.items():  # both evaluated top-1
            logits, G, _ = moe_forward(bt, con["cnn"], con["con"], con["lm"], w_r, blank)
            acc[k] += frac * eq.exact(logits, bt["tgt"], bt["ans"]).float().mean(1).cpu()
            route_acc[k] += frac * ((G[..., 0] > 0) == is_img).float().mean((1, 2)).cpu()
        # unimodal: the CNN alone, and the symbolic pipeline (CNN's argmax digit written as a word, then the LM)
        x1 = data["x_test"][sl][None].expand(S, -1, -1, -1)
        pred = eq.cnn_logits(eq.cnn_features(x1, w_cnn1), w_cnn1).argmax(-1)  # (S, n)
        cnn_acc += frac * (pred == data["y_test"][sl]).float().mean(1).cpu()
        tok = data["table"]["tokens"][pred, test["op"][sl], test["b"][sl]]
        truth = data["table"]["tokens"][data["y_test"][sl], test["op"][sl], test["b"][sl]]
        ans = data["table"]["answer_mask"][data["y_test"][sl], test["op"][sl], test["b"][sl]]
        z = eq.lm_logits(eq.embed(tok[..., :-1], w_lm1), w_lm1)
        symbolic += frac * eq.exact(z, truth[..., 1:].expand(S, -1, -1), ans.expand(S, -1, -1)).float().mean(1).cpu()
    # the LM alone on every text-only prompt of the language
    tok = data["table"]["tokens"].reshape(300, T.CTX)
    ans = data["table"]["answer_mask"].reshape(300, T.CTX - 1)
    z = eq.lm_logits(eq.embed(tok[None, :, :-1].expand(S, -1, -1), w_lm1), w_lm1)
    lm_acc = eq.exact(z, tok[None, :, 1:].expand(S, -1, -1), ans[None].expand(S, -1, -1)).float().mean(1).cpu()
    shape = (S, N)
    return {"acc": {k: v.reshape(shape) for k, v in acc.items()},
            "route_acc": {k: v.reshape(shape) for k, v in route_acc.items()},
            "cnn_acc": cnn_acc, "lm_acc": lm_acc, "symbolic": symbolic, "test": test}  # fmt: skip


RULES = {"train_cnn": rule_train_cnn, "train_lm": rule_train_lm, "train_monolith": rule_train_monolith,
         "train_connector": rule_train_connector, "global_finetune": rule_global_finetune,
         "train_router": rule_train_router, "train_router_soft": lambda st: rule_train_router(st, soft=True),
         "evaluate": rule_evaluate}  # fmt: skip


def make_store(cfg: Config, dev: torch.device) -> dict:
    """Data and the model-axis bookkeeping every rule reads. Model m = (seed, n), seed-major (as the job DAG)."""
    data = T.load_mnist(dev)
    data["table"] = {k: v.to(dev) for k, v in T.sentence_table().items()}
    S, N = len(cfg.seeds), len(cfg.sizes)
    return {"cfg": cfg, "dev": dev, "data": data, "pool": T.joint_pool(cfg.seeds, len(data["y_train"]), dev),
            "seeds_mn": [s for s in cfg.seeds for _ in cfg.sizes], "n_mn": [n for _ in cfg.seeds for n in cfg.sizes],
            "seed_of": torch.arange(S, device=dev).repeat_interleave(N)}  # fmt: skip


def run_engine(cfg: Config, dev: torch.device, d: dict | None = None) -> tuple[dict, dict, dict]:
    """Run the training DAG: rules level by level (levels of A_rule), each rule one batched call over its models.
    Returns the store, the measured seconds per rule and the DAG."""
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


@torch.no_grad()
def greedy(w_cnn, w_con, w_lm, x: torch.Tensor, tokens: torch.Tensor) -> list[str]:
    """Let ONE stitched model (stack of 1) write the result word after '… is ' letter by letter.  x (N, 28, 28)."""
    tok = tokens.clone()[None]
    start = [T.decode(r).index("=") + 1 for r in tokens]
    pos = torch.tensor(start, device=x.device)
    rows = torch.arange(len(start), device=x.device)
    for k in range(6):  # 'seven.' is the longest answer
        z = eq.stitched_logits(x[None], tok[:, :, :-1], w_cnn, w_con, w_lm)[0]
        p = (pos + k).clamp(max=T.CTX - 1)
        nxt = z[rows, p - 1].argmax(-1)
        tok[0, rows, p] = nxt
    out = []
    for r, s in zip(tok[0], start):
        text = T.decode(r[s:])
        out.append(re.split(r"[._]", text)[0])
    return out
