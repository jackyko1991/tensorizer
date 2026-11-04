r"""Search, discretise, retrain, write the equation, read it back; train the CNN; run the data study.

Search: first-order DARTS (Liu, Simonyan & Yang 2019) with alternating updates. Each step takes a class-balanced
mini-batch from the train half and updates the operator parameters θ (filters, read-outs, prototypes) with α fixed,
then a mini-batch from the validation half and updates α with θ fixed; the second-order term of DARTS (the gradient
through one unrolled θ step) is dropped. The mixture is softmax(α/T) with the temperature T annealed geometrically
from 1 to 0.001, so the supernet ends as one discrete DAG (this closes the gap between the mixture and the DAG kept,
as the annealed relaxations of SNAS and GDAS do; plain DARTS keeps T = 1). Data constants (class means μ, principal
axes U, feature mean m and scale s) come from each training mini-batch (detached, like batch norm) and from all the
training images for evaluation.

Discretise: every edge keeps its argmax operator; 'zero' may win and removes the edge (DARTS instead excludes zero
and keeps the 2 strongest inputs of every node; with an annealed supernet that rule drops edges the supernet uses).
Retrain: the discrete DAG inherits the parameters the search trained and is fine-tuned briefly on the train half; the
step with the best validation accuracy is kept (early stopping). The CNN is trained with the same early stopping, on
the same images.
"""

import re
import time

import torch
import torch.nn.functional as F

from . import equation as eq
from . import tensors as T

LR_THETA, LR_ALPHA, LR_CNN = 5e-3, 1e-2, 3e-3
EVAL_EVERY = 10
TEMP = (1.0, 0.001)  # softmax temperature of the mixture, annealed geometrically over the search


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def statistics(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, th: dict, reds: list[str]) -> dict:
    with torch.no_grad():
        return eq.data_statistics(eq.pooled(x, w, th), y, reds)


def batch_forward(x, y, w, th, reds):
    """Logits of a training batch with the data constants taken from that same batch (detached, like batch norm):
    no stale constants while θ moves. Returns the logits and the constants."""
    Z = eq.pooled(x, w, th)
    st = eq.data_statistics(Z, y, reds)
    return eq.logits(Z, w, th, st), st


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).double().mean())


def predict(x: torch.Tensor, w: torch.Tensor, th: dict, st: dict, chunk: int = 500) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([eq.forward(x[i : i + chunk], w, th, st).argmax(1) for i in range(0, len(x), chunk)])


# ---------------------------------------------------------------- the search (supernet)
def search(xt, yt, xv, yv, steps: int, batch: int, seed: int, snapshots: int = 16) -> dict:
    """First-order DARTS on the small set. Returns α, θ, the data constants and the mixture weights over training."""
    dev = xt.device
    mask = eq.family_mask().to(dev)
    alpha = torch.zeros(len(eq.EDGES), len(eq.OPS), device=dev, requires_grad=True)
    th = T.init_supernet(len(eq.CELL_EDGES), eq.reduce_dims(), seed, dev)
    opt_t = torch.optim.Adam(th.values(), lr=LR_THETA)
    opt_a = torch.optim.Adam([alpha], lr=LR_ALPHA)
    gen = torch.Generator().manual_seed(seed)
    reds = list(eq.RED_OPS)
    at = sorted({round(k * steps / max(1, snapshots - 1)) for k in range(snapshots)})
    hist = {"step": [], "w": [], "loss": [], "temp": []}
    w = eq.mixture_weights(alpha / TEMP[0], mask)
    eq.init_heads(th, statistics(xt, yt, w, th, reds), reds)
    for step in range(steps + 1):
        temp = TEMP[0] * (TEMP[1] / TEMP[0]) ** (step / max(1, steps))
        w = eq.mixture_weights(alpha / temp, mask)
        if step in at:
            hist["step"].append(step)
            hist["w"].append(w.detach().cpu())
            hist["temp"].append(temp)
        if step == steps:
            break
        i = T.balanced_batch(yt, batch, gen)
        out, st = batch_forward(xt[i], yt[i], w.detach(), th, reds)  # θ step, α fixed
        loss = F.cross_entropy(out, yt[i])
        opt_t.zero_grad()
        loss.backward()
        opt_t.step()
        j = T.balanced_batch(yv, batch, gen)
        wa = eq.mixture_weights(alpha / temp, mask)  # α step, θ fixed; constants from this step's training batch
        loss_a = F.cross_entropy(eq.forward(xv[j], wa, th, st), yv[j])
        (alpha.grad,) = torch.autograd.grad(loss_a, [alpha])  # only ∂L/∂α: no weight gradients of θ computed
        opt_a.step()
        hist["loss"].append(float(loss_a.detach()))
    st = statistics(xt, yt, w.detach(), th, reds)
    return {"alpha": alpha.detach(), "w": w.detach(), "theta": th, "stats": st, "hist": hist}


def discretise(w: torch.Tensor) -> dict:
    """The discrete DAG from the mixture weights w (9, 23): every edge keeps its largest-weight operator. Unlike DARTS,
    'zero' may win (the edge is then removed) and there is no top-2-inputs rule: with the annealed temperature the
    supernet ends close to exactly this DAG. A node whose every input lost to 'zero' keeps its strongest other edge.
    Returns the kept cell edges, pool, reduce and head, and the one-hot w."""
    w = w.detach().cpu()
    nonzero = [eq.OPS.index(o) for o in eq.CELL_OPS[1:]]
    edges = []
    for i in range(1, eq.N_NODES + 1):
        inc = [e for e, (_, ii) in enumerate(eq.CELL_EDGES) if ii == i]
        keep = [(e, eq.OPS[int(w[e].argmax())]) for e in inc if int(w[e].argmax()) != 0]
        if not keep:  # every input of z_i lost to 'zero': keep its strongest non-zero edge (no empty node)
            e = max(inc, key=lambda e: float(w[e, nonzero].max()))
            keep = [(e, eq.OPS[nonzero[int(w[e, nonzero].argmax())]])]
        edges += keep
    edges.sort()
    pick = {name: eq.OPS[int(w[eq.EDGES.index(name)].argmax())] for name in ("pool", "reduce", "head")}
    arch = {"edges": edges, **pick}
    return {**arch, "onehot": onehot(arch)}


def onehot(arch: dict) -> torch.Tensor:
    W = torch.zeros(len(eq.EDGES), len(eq.OPS))
    for e, op in arch["edges"]:
        W[e, eq.OPS.index(op)] = 1.0
    for name in ("pool", "reduce", "head"):
        W[eq.EDGES.index(name), eq.OPS.index(arch[name])] = 1.0
    return W


def n_trainable(arch: dict) -> int:
    """Learned numbers of the discrete DAG: 9 per learned filter, plus the head (10 D + 10 linear, 10 D + 1
    prototypes)."""
    D = eq.reduce_dims()[arch["reduce"]]
    conv = 9 * sum(op == "conv3" for _, op in arch["edges"])
    return conv + (10 * D + 10 if arch["head"] == "linear" else 10 * D + 1)


def n_constants(arch: dict) -> int:
    """Numbers computed in closed form from the training images (no gradient): μ or U when used, m (D,) and one
    scale s per channel."""
    D = eq.reduce_dims()[arch["reduce"]]
    extra = {"template": 10 * eq.CHANNELS * 49, "pca": eq.CHANNELS * 49 * T.Q_PCA}.get(arch["reduce"], 0)
    return D + eq.CHANNELS + extra


def retrain(arch: dict, xt, yt, xv, yv, steps: int, batch: int, seed: int, inherit: dict | None = None) -> dict:
    """Train the discrete DAG's parameters on the train half (Adam), keeping the step with the best validation
    accuracy. With `inherit` (the supernet's θ) it starts from the parameters the search trained (weight inheritance,
    as in one-shot NAS) and is a brief fine-tune; without, from fresh filters and class-mean heads."""
    dev = xt.device
    W = arch["onehot"].to(dev)
    reds = [arch["reduce"]]
    if inherit is not None:
        th = {k: v.detach().clone().requires_grad_(True) for k, v in inherit.items()}
    else:
        th = T.init_supernet(len(eq.CELL_EDGES), eq.reduce_dims(), seed + 7, dev)
        eq.init_heads(th, statistics(xt, yt, W, th, reds), reds)
    used = [k for k in th if k == "K" or k.endswith("_" + arch["reduce"])]
    opt = torch.optim.Adam([th[k] for k in used], lr=LR_THETA)
    gen = torch.Generator().manual_seed(seed + 7)
    best = (-1.0, None, None)
    for step in range(steps + 1):
        if step % EVAL_EVERY == 0 or step == steps:
            st = statistics(xt, yt, W, th, reds)
            acc = accuracy(predict(xv, W, th, st), yv)
            if acc > best[0]:
                best = (acc, {k: v.detach().clone() for k, v in th.items()}, st)
        if step == steps:
            break
        i = T.balanced_batch(yt, batch, gen)
        loss = F.cross_entropy(batch_forward(xt[i], yt[i], W, th, reds)[0], yt[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"theta": best[1], "stats": best[2], "val_acc": best[0]}


def train_cnn(xt, yt, xv, yv, steps: int, batch: int, seed: int) -> dict:
    """The CNN with Adam on the train half, best validation step kept (early stopping)."""
    p = T.init_cnn(seed, xt.device)
    opt = torch.optim.Adam(p.values(), lr=LR_CNN)
    gen = torch.Generator().manual_seed(seed + 11)
    best = (-1.0, None)
    for step in range(steps + 1):
        if step % EVAL_EVERY == 0 or step == steps:
            with torch.no_grad():
                acc = accuracy(eq.cnn(xv, p).argmax(1), yv)
            if acc > best[0]:
                best = (acc, {k: v.detach().clone() for k, v in p.items()})
        if step == steps:
            break
        i = T.balanced_batch(yt, batch, gen)
        loss = F.cross_entropy(eq.cnn(xt[i], p), yt[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"params": best[1], "val_acc": best[0]}


# ---------------------------------------------------------------- the equation writer and its reader
def write_equation(arch: dict) -> list[str]:
    """The discrete DAG as LaTeX lines: one per cell node, then the features, their standardisation and the decision."""
    lines = []
    for i in range(1, eq.N_NODES + 1):
        terms = []
        for e, op in arch["edges"]:
            j, ii = eq.CELL_EDGES[e]
            if ii == i:
                terms.append(eq.LATEX[op].format(a="x" if j == 0 else f"z_{{{j}}}", e=e + 1))
        lines.append(f"z_{{{i}}} = " + (" + ".join(terms) if terms else "0"))
    inner = eq.LATEX[arch["pool"]].format(a="z")
    lines.append("f = " + eq.LATEX[arch["reduce"]].format(a=inner))
    lines.append(r"\hat{f} = (f - m) / s")
    if arch["head"] == "linear":
        lines.append(r"\hat{y} = \arg\max_{k}\, (W \hat{f} + b)_{k}")
    else:
        lines.append(r"\hat{y} = \arg\min_{k}\, \|\hat{f} - p_{k}\|^{2}")
    return lines


def _regex(template: str) -> re.Pattern:
    """A formula template → a regex: {a} becomes a group for the argument, {e} for the edge number."""
    text = re.escape(template.format(a="ARGXQ", e="EDGEXQ"))
    return re.compile(text.replace("ARGXQ", "(?P<a>.+)").replace("EDGEXQ", r"(?P<e>\d+)"))


def _split_terms(s: str) -> list[str]:
    """Split at top-level ' + ' (outside brackets)."""
    out, depth, start, i = [], 0, 0, 0
    while i < len(s):
        depth += s[i] in "([{" and 1 or (s[i] in ")]}" and -1 or 0)
        if depth == 0 and s.startswith(" + ", i):
            out.append(s[start:i])
            start = i = i + 3
            continue
        i += 1
    return out + [s[start:]]


def read_equation(lines: list[str]) -> dict:
    """Parse the written LaTeX back into a DAG (only the lines, nothing from the search): every term is matched
    against the operator formulas. Returns the same structure as discretise() (without the one-hot matrix)."""
    fam = {"cell": [o for o in eq.CELL_OPS if o not in ("identity", "zero")], "pool": eq.POOL_OPS, "reduce": eq.RED_OPS}
    pats = {o: _regex(eq.LATEX[o]) for o in eq.LATEX}
    edges = []
    for line in lines[: eq.N_NODES]:
        lhs, rhs = line.split(" = ", 1)
        i = int(re.fullmatch(r"z_\{(\d)\}", lhs).group(1))
        for term in _split_terms(rhs) if rhs != "0" else []:
            for op in fam["cell"] + ["identity"]:
                m = pats[op].fullmatch(term)
                if m and re.fullmatch(r"x|z_\{\d\}", m.group("a")):
                    j = 0 if m.group("a") == "x" else int(m.group("a")[3:-1])
                    e = eq.CELL_EDGES.index((j, i))
                    assert "e" not in m.groupdict() or int(m.group("e")) == e + 1, "filter index ≠ edge"
                    edges.append((e, op))
                    break
            else:
                raise ValueError(f"cannot read the term {term!r}")
    rhs = lines[eq.N_NODES].split(" = ", 1)[1]
    red = next(o for o in fam["reduce"] if pats[o].fullmatch(rhs))
    inner = pats[red].fullmatch(rhs).group("a")
    pool = next(o for o in fam["pool"] if pats[o].fullmatch(inner) and pats[o].fullmatch(inner).group("a") == "z")
    head = "linear" if r"W \hat{f}" in lines[-1] else "proto" if r"p_{k}" in lines[-1] else None
    assert lines[eq.N_NODES + 1] == r"\hat{f} = (f - m) / s" and head is not None
    return {"edges": sorted(edges), "pool": pool, "reduce": red, "head": head}


def evaluate_equation(lines: list[str], x: torch.Tensor, th: dict, st: dict) -> torch.Tensor:
    """ŷ from the written equation alone: read it, then evaluate it line by line with the operator functions and the
    named numbers (K_e, μ, U, m, s, W, b, p_k)."""
    arch = read_equation(lines)
    with torch.no_grad():
        z = {0: x}
        for i in range(1, eq.N_NODES + 1):
            terms = [eq.cell_op(op, z[eq.CELL_EDGES[e][0]], th["K"][e]) for e, op in arch["edges"]
                     if eq.CELL_EDGES[e][1] == i]  # fmt: skip
            z[i] = terms[0] if terms else torch.zeros_like(x)
            for t in terms[1:]:
                z[i] = z[i] + t
        Z = eq.pool_op(arch["pool"], torch.cat([z[0], z[1], z[2], z[3]], 1))
        r = arch["reduce"]
        fh = (eq.reduce_op(r, Z, st) - st[f"m_{r}"]) / st[f"s_{r}"]
        if arch["head"] == "linear":
            return (fh @ th[f"W_{r}"].T + th[f"b_{r}"]).argmax(1)
        return ((fh[:, None, :] - th[f"P_{r}"][None]) ** 2).sum(-1).argmin(1)


# ---------------------------------------------------------------- one run and the study
def run_tensor(data: dict, tr, va, cfg: dict, seed: int) -> dict:
    xt, yt, xv, yv = data["x_pool"][tr], data["y_pool"][tr], data["x_pool"][va], data["y_pool"][va]
    dev = xt.device
    sync(dev)
    t0 = time.perf_counter()
    s = search(xt, yt, xv, yv, cfg["search_steps"], cfg["batch"], seed)
    arch = discretise(s["w"])
    r = retrain(arch, xt, yt, xv, yv, cfg["retrain_steps"], cfg["batch"], seed, inherit=s["theta"])
    sync(dev)
    secs = time.perf_counter() - t0
    W = arch["onehot"].to(dev)
    pred = predict(data["x_test"], W, r["theta"], r["stats"])
    acc_super = accuracy(predict(data["x_test"], s["w"], s["theta"], s["stats"]), data["y_test"])
    return {"arch": arch, "theta": r["theta"], "stats": r["stats"], "hist": s["hist"], "seconds": secs,
            "acc": accuracy(pred, data["y_test"]), "acc_supernet": acc_super, "pred": pred.cpu(),
            "equation": write_equation(arch), "params": n_trainable(arch), "constants": n_constants(arch)}  # fmt: skip


def run_cnn(data: dict, tr, va, cfg: dict, seed: int) -> dict:
    xt, yt, xv, yv = data["x_pool"][tr], data["y_pool"][tr], data["x_pool"][va], data["y_pool"][va]
    sync(xt.device)
    t0 = time.perf_counter()
    c = train_cnn(xt, yt, xv, yv, cfg["cnn_steps"], cfg["batch"], seed)
    sync(xt.device)
    secs = time.perf_counter() - t0
    with torch.no_grad():
        pred = eq.cnn(data["x_test"], c["params"]).argmax(1)
    return {"seconds": secs, "acc": accuracy(pred, data["y_test"]), "params": T.cnn_param_count(), "pred": pred.cpu()}


def baselines(data: dict, tr, va) -> dict:
    """Parameter-free references from all n images per class (train and validation halves together)."""
    idx = torch.unique(torch.cat([tr, va]))
    x, y = data["x_pool"][idx], data["y_pool"][idx]
    return {"ncm": accuracy(eq.nearest_class_mean(x, y, data["x_test"]), data["y_test"]),
            "pca_lda": accuracy(eq.pca_lda(x, y, data["x_test"]), data["y_test"])}  # fmt: skip


def n_to_reach(sizes, acc_mean, target: float) -> float:
    """Images per class at which the mean accuracy first reaches `target`, linear in log n between sizes (nan if
    never reached)."""
    import math

    for k, a in enumerate(acc_mean):
        if a >= target:
            if k == 0:
                return float(sizes[0])
            a0, a1, n0, n1 = acc_mean[k - 1], a, math.log(sizes[k - 1]), math.log(sizes[k])
            return float(math.exp(n0 + (target - a0) / (a1 - a0) * (n1 - n0)))
    return float("nan")
