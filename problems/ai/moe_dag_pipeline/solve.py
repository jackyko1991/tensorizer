"""The workflow DAG engine (Snakefile → routed job DAG → one batched call per rule), the same computation as a
top-1 mixture-of-experts layer, and the four stages: split training, a learned router, end-to-end fine-tuning,
distillation and int8 quantisation.

The DAG algebra (Kronecker job adjacency, nilpotency, boolean closure and levels) is imported from
medical/cow_workflow_dag, whose engine runs a real Snakemake workflow; this file adds the one thing that workflow
did not have, a conditional input (an input function), whose job-level block is the router matrix G.
"""

import itertools
import math
import re
import time
from functools import partial

import torch

from problems._common.progress import steps as progress_steps
from problems.medical.cow_workflow_dag import equation as dag_eq
from problems.medical.cow_workflow_dag.solve import parse_snakefile

from . import equation as eq
from . import tensors as T

AXES = ("sample", "op")  # wildcard axes, in job-numbering order (row-major)


# ---------------------------------------------------------------- the workflow DAG
def build_dag(text: str, n_samples: int) -> dict:
    """Rules of the Snakefile, the rule graph A_rule (A_sr = 1 when rule s reads a pattern rule r writes), each
    rule's wildcard axes, the levels, and the conditional edges: an input function such as

        def routed(wildcards): return f"results/{wildcards.sample}/fused_{OP[wildcards.sample]}.npy"

    reads the pattern results/{sample}/fused_{op}.npy with the wildcard `op` chosen per sample."""
    rules = parse_snakefile(text)
    funcs = dict(re.findall(r'^def (\w+)\(wildcards\):\s*\n\s+return f"([^"]+)"', text, flags=re.M))
    routed = dict(re.findall(r"^rule (\w+):\s*\n\s+input: (\w+)\s*$", text, flags=re.M))
    for r in rules:
        if r["name"] in routed:
            pat = re.sub(r"\{wildcards\.(\w+)\}", r"{\1}", funcs[routed[r["name"]]])
            pat = re.sub(r"\{(\w+)\[wildcards\.\w+\]\}", lambda m: "{" + m.group(1).lower() + "}", pat)
            r["inputs"].append((pat, ()))
            r["routed_by"] = routed[r["name"]]
    wild = lambda p: set(re.findall(r"\{(\w+)\}", p))  # noqa: E731
    axes = []
    for r in rules:
        pats = r["outputs"] or [p for p, _ in r["inputs"]]
        wc = set().union(*map(wild, pats)) - ({w for _, agg in r["inputs"] for w in agg} if not r["outputs"] else set())
        axes.append(tuple(a for a in AXES if a in wc))
    R = len(rules)
    A_rule, conditional = torch.zeros(R, R), {}
    for s, rs in enumerate(rules):
        for pat, _ in rs["inputs"]:
            for r, rr in enumerate(rules):
                if pat in rr["outputs"]:
                    A_rule[s, r] = 1
                    chosen = wild(pat) - set(axes[s]) - {w for p2, agg in rs["inputs"] for w in agg}
                    if chosen and rs.get("routed_by"):
                        conditional[(s, r)] = chosen.pop()
    return {"rules": rules, "names": [r["name"] for r in rules], "A_rule": A_rule, "axes": axes,
            "levels": dag_eq.levels(A_rule), "conditional": conditional,
            "sizes": {"sample": n_samples, "op": T.E}}  # fmt: skip


def block(dag: dict, s: int, r: int, G: torch.Tensor) -> torch.Tensor:
    """Job-level block (n_s, n_r) of the edge r → s: the Kronecker product M^{sr} of cow_workflow_dag, and for a
    conditional edge that product masked by the router, (I_N ⊗ 𝟙ᵀ_E) ⊙ vec(G)ᵀ: job n reads job (n, e) iff G_ne = 1."""
    M = dag_eq.kron_block(dag["axes"][r], dag["axes"][s], dag["sizes"])
    if (s, r) in dag["conditional"]:
        M = M * G.reshape(1, -1).to(M.dtype)
    return M


def job_adjacency_routed(dag: dict, G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The whole (J, J) job adjacency with the router inside (small N only: J = N·(3 + E) + 1)."""
    A, off = dag_eq.job_adjacency(dag["A_rule"], dag["axes"], dag["sizes"])
    for s, r in dag["conditional"]:
        A[off[s] : off[s + 1], off[r] : off[r + 1]] = block(dag, s, r, G)
    return A, off


def needed_jobs(dag: dict, G: torch.Tensor) -> dict:
    """Which jobs the target needs (what Snakemake schedules): the target rule's jobs, then backwards level by level,
    need_r = ⋁_s M^{sr}ᵀ need_s (boolean). Block by block, so the (J, J) matrix is never built."""
    A, lev, names = dag["A_rule"], dag["levels"], dag["names"]
    need = {}
    for r in sorted(range(len(names)), key=lambda i: -int(lev[i])):
        n_r = math.prod(dag["sizes"][a] for a in dag["axes"][r])
        if A[:, r].sum() == 0:  # nobody reads it: a target
            need[names[r]] = torch.ones(n_r, dtype=torch.bool)
            continue
        v = torch.zeros(n_r)
        for s in A[:, r].nonzero().flatten().tolist():
            v = v + block(dag, s, r, G).T @ need[names[s]].float()
        need[names[r]] = v > 0
    return {n: need[n] for n in names}


def run_dag(w: dict, x: torch.Tensor, tok: torch.Tensor, dag: dict) -> tuple[torch.Tensor, dict]:
    """Run the workflow as tensors: rules level by level, each rule ONE call over all its needed jobs. The router is
    the workflow's own input function (the parsed operation), so `fuse` runs only the N chosen of N·E jobs: one
    batched expert call per value of its `op` wildcard. x (N, 784), tok (N, 3) → logits (M, N, 10), times (s)."""
    M = w["img"]["W1"].shape[0]
    xs, ts = x.expand(M, *x.shape), tok.expand(M, *tok.shape)
    G = T.router_matrix(tok)
    need = needed_jobs(dag, G)
    names = dag["names"]
    fz = w["fuse"]

    def fuse(st):
        jobs = need["fuse"].nonzero().flatten()  # (N,) needed (sample, op) jobs, sample-major
        n, e = jobs // T.E, jobs % T.E
        u = eq.features(st["encode_image"], st["encode_text"][1])
        out = torch.zeros(M, len(jobs), fz["W1"].shape[-1], dtype=u.dtype)
        for k in range(T.E):  # the values of the wildcard {op}: one batched call each
            sel = e == k
            out[:, sel] = eq.expert(u[:, n[sel]], fz["W1"][:, k], fz["b1"][:, k], fz["W2"][:, k], fz["b2"][:, k])
        return jobs, out

    def classify(st):
        jobs, out = st["fuse"]
        src = block(dag, names.index("classify"), names.index("fuse"), G).argmax(1)  # the file `routed` names
        return eq.classify(w["cls"], out[:, torch.searchsorted(jobs, src)])

    funcs = {"encode_image": lambda st: eq.image_logits(w["img"], xs),
             "encode_text": lambda st: eq.text_features(w["txt"], ts),
             "fuse": fuse, "classify": classify, "collect": lambda st: st["classify"]}  # fmt: skip
    store, times = {}, {}
    for k in range(int(dag["levels"].max()) + 1):
        for r in (dag["levels"] == k).nonzero().flatten().tolist():
            t0 = time.perf_counter()
            store[names[r]] = funcs[names[r]](store)
            times[names[r]] = time.perf_counter() - t0
    return store["collect"], times


def run_jobs_one_by_one(w: dict, x: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    """The same workflow the way a job scheduler runs it: one call per job (B = 1), N·4 calls (no process start-up
    or file I/O counted). Only for timing; same numbers as run_dag."""
    out = []
    for i in range(len(x)):
        xi, ti = x[i : i + 1].expand(1, 1, -1), tok[i : i + 1].expand(1, 1, -1)
        u = eq.features(eq.image_logits(w["img"], xi), eq.text_features(w["txt"], ti)[1])
        k = int(T.parse_op(tok[i : i + 1]))
        fz = w["fuse"]
        o = eq.expert(u, fz["W1"][:, k], fz["b1"][:, k], fz["W2"][:, k], fz["b2"][:, k])
        out.append(eq.classify(w["cls"], o))
    return torch.cat(out, 1)


# ---------------------------------------------------------------- the same computation as a MoE layer
def fuse_classify(w: dict, u: torch.Tensor, G: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Dispatch → all experts in one batched product → combine → classify. u (M, N, 20), G, g (M, N, E)."""
    D = eq.dispatch(G, int(G.sum(1).max()))  # capacity = the largest group: no sample is dropped
    U = torch.einsum("mnec,mni->meci", D, u)
    return eq.classify(w["cls"], eq.combine(D, g, eq.experts_stacked(U, w["fuse"])))


def moe_forward(w: dict, x: torch.Tensor, tok: torch.Tensor, G: torch.Tensor, combine: torch.Tensor) -> torch.Tensor:
    """The pipeline as one MoE layer with a given 0/1 router G and combine weights g (both (M, N, E))."""
    M = w["img"]["W1"].shape[0]
    xs = x if x.dim() == 3 else x.expand(M, *x.shape)
    ts = tok if tok.dim() == 3 else tok.expand(M, *tok.shape)
    u = eq.features(eq.image_logits(w["img"], xs), eq.text_features(w["txt"], ts)[1])
    return fuse_classify(w, u, G, combine)


def switch_forward(w: dict, x: torch.Tensor, tok: torch.Tensor):
    """Learned router (Switch Transformer): p = softmax gate on the text features, G = top-1(p), g = G ⊙ p.
    Returns logits, p, G."""
    h, z_txt = eq.text_features(w["txt"], tok)
    u = eq.features(eq.image_logits(w["img"], x), z_txt)
    p = eq.gate_probs(w["gate"], h)
    G = eq.top1(p)
    return fuse_classify(w, u, G, G * p), p, G


# ---------------------------------------------------------------- training helpers
def accuracy(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (z.argmax(-1) == y).float().mean(-1)


def batch(data: dict, M: int, B: int, gen: torch.Generator):
    """Each model its own minibatch: images (M, B, 784), tokens (M, B, 3), digit, operation, number, answer."""
    idx = torch.randint(len(data["x_train"]), (M, B), generator=gen)
    tok, op, k = T.make_texts(M * B, gen)
    d = data["y_train"][idx]
    op, k = op.reshape(M, B), k.reshape(M, B)
    return (
        data["x_train"][idx],
        tok.reshape(M, B, T.L),
        d,
        op,
        k,
        T.answer(d.flatten(), op.flatten(), k.flatten()).reshape(M, B),
    )


def checkpoints(n_steps: int, n: int) -> set:
    return {round(i * n_steps / n) for i in range(n + 1)}


def table_inputs(dtype=torch.float32):
    """All 300 (digit, operation, number) cases as one-hot features u (300, 20), the router G (300, E), answers."""
    d, op, k = (
        t.flatten() for t in torch.meshgrid(torch.arange(10), torch.arange(T.E), torch.arange(10), indexing="ij")
    )
    u = torch.cat([torch.nn.functional.one_hot(d, 10), torch.nn.functional.one_hot(k, 10)], 1).to(dtype)
    return u, torch.nn.functional.one_hot(op, T.E).to(dtype), T.answer(d, op, k)


def float_bytes(w: dict) -> int:
    return 4 * T.n_params(w)


def int8_bytes(st: dict) -> int:
    """Student stored as int8 weight matrices + one float32 scale per output channel + float32 biases."""
    total = 0
    for k, t in st.items():
        total += t[0].numel() + 4 * t.shape[-1] if k.startswith("W") else 4 * t[0].numel()
    return total


def quantise(st: dict) -> dict:
    return {k: eq.fake_quant(t)[0] if k.startswith("W") else t.clone() for k, t in st.items()}


def time_call(f, repeats: int) -> float:
    """Median wall-clock of f() over repeats (s), on one thread so the numbers compare."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        f()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
    finally:
        torch.set_num_threads(n)
    return sorted(ts)[len(ts) // 2]


# ---------------------------------------------------------------- the four stages
def run_stages(cfg: T.Config, n_checkpoints: int = 20, display: int = 10) -> dict:
    """Train and measure everything. All seeds run together on the model axis M."""
    torch.manual_seed(0)
    M = cfg.seeds
    data = T.load_mnist()
    gen = torch.Generator().manual_seed(1)
    xt, yt = data["x_test"][: cfg.test_size], data["y_test"][: cfg.test_size]
    Nt = len(xt)
    tok_t, op_t, k_t = T.make_texts(Nt, torch.Generator().manual_seed(2026))  # one fixed test set of texts
    y_t = T.answer(yt, op_t, k_t)
    Xt, Tt = xt.expand(M, Nt, T.P), tok_t.expand(M, Nt, T.L)
    G_t = T.router_matrix(tok_t).expand(M, Nt, T.E)
    u_tab, G_tab, y_tab = table_inputs()
    dag = T.SNAKEFILE.read_text(encoding="utf-8")
    dag = build_dag(dag, n_samples=Nt)
    res = {"cfg": cfg, "M": M, "curves": {}, "acc": {}, "subtask_acc": {}}

    def end_acc(w):
        with torch.no_grad():
            return accuracy(moe_forward(w, Xt, Tt, G_t, G_t), y_t)

    # stage 2: split training, each expert on its own sub-task (the losses touch disjoint weights)
    w = T.clone(T.init_weights(M, cfg, gen), grad=True)
    opt = torch.optim.Adam(T.leaves({k: w[k] for k in ("img", "txt", "fuse", "cls")}), lr=cfg.lr)
    cps, curve = checkpoints(cfg.split_steps, n_checkpoints), []
    for t in progress_steps(cfg.split_steps + 1, "stage 2: split training"):
        if t in cps:
            with torch.no_grad():
                Gtab = G_tab.expand(M, -1, -1)
                curve.append((t, accuracy(eq.image_logits(w["img"], Xt), yt.expand(M, Nt)),
                              accuracy(eq.text_features(w["txt"], Tt)[1], k_t.expand(M, Nt)),
                              accuracy(fuse_classify(w, u_tab.expand(M, -1, -1), Gtab, Gtab), y_tab.expand(M, -1)),
                              end_acc(w)))  # fmt: skip
        if t == cfg.split_steps:
            break
        x, tok, d, op, k, y = batch(data, M, cfg.batch, gen)
        dd, oo, kk = (torch.randint(n, (M, cfg.batch), generator=gen) for n in (10, T.E, 10))
        u1 = torch.cat([torch.nn.functional.one_hot(dd, 10), torch.nn.functional.one_hot(kk, 10)], -1).float()
        G1 = torch.nn.functional.one_hot(oo, T.E).float()
        y1 = T.answer(dd.flatten(), oo.flatten(), kk.flatten()).reshape(M, -1)
        loss = (eq.cross_entropy(eq.image_logits(w["img"], x), d)
                + eq.cross_entropy(eq.text_features(w["txt"], tok)[1], k)
                + eq.cross_entropy(fuse_classify(w, u1, G1, G1), y1))  # fmt: skip
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    res["curves"]["split"] = curve
    _, a_img, a_txt, a_fuse, a_chain = curve[-1]
    res["subtask_acc"] = {"image": a_img, "text": a_txt, "fuse": a_fuse}
    w_split = T.clone(w)

    # the DAG engine on the split-trained pipeline, and the equivalence with the fixed-router MoE layer
    with torch.no_grad():
        z_dag, rule_times = run_dag(w_split, xt, tok_t, dag)
        res["acc"]["split"] = accuracy(z_dag, y_t)
        eqv = {}
        for name, dt in (("float64", torch.float64), ("float32", torch.float32)):
            w64 = T.cast(w_split, dt)
            za, _ = run_dag(w64, xt.to(dt), tok_t, dag)
            Gd = G_t.to(dt)
            zb = moe_forward(w64, xt.to(dt), tok_t, Gd, Gd)
            eqv[name] = {"max_abs_diff": float((za - zb).abs().max()), "bitwise_equal": bool(torch.equal(za, zb)),
                         "same_predictions": int((za.argmax(-1) == zb.argmax(-1)).sum()),
                         "total": za.shape[0] * Nt}  # fmt: skip
        need = needed_jobs(dag, T.router_matrix(tok_t))
        eqv["jobs"] = {n: (int(v.sum()), len(v)) for n, v in need.items()}
        res["equivalence"], res["rule_times"] = eqv, rule_times

    # stage 1b: a learned router (softmax gate + load balancing), encoders frozen, two variants side by side on the
    # model axis: models 0…M−1 keep the split-trained fuse experts (only the gate learns), models M…2M−1 get fresh
    # experts that learn together with the gate
    rep2 = lambda t: torch.cat([t, t])  # noqa: E731
    fresh = T.init_experts(M, cfg, torch.Generator().manual_seed(7))
    wr = {"img": T.tmap(w_split["img"], rep2), "txt": T.tmap(w_split["txt"], rep2),
          "fuse": {k: torch.cat([w_split["fuse"][k], fresh["fuse"][k]]) for k in fresh["fuse"]},
          "cls": {k: torch.cat([w_split["cls"][k], fresh["cls"][k]]) for k in fresh["cls"]},
          "gate": T.init_experts(2 * M, cfg, torch.Generator().manual_seed(8))["gate"]}  # fmt: skip
    wr = T.clone(wr, grad=True)
    kept = T.leaves({k: wr[k] for k in ("fuse", "cls")})
    opt = torch.optim.Adam(kept + T.leaves(wr["gate"]), lr=cfg.lr)
    show = torch.cat([(op_t == e).nonzero().flatten()[:display] for e in range(T.E)])  # display rows, by operation
    X2, T2 = xt.expand(2 * M, Nt, T.P), tok_t.expand(2 * M, Nt, T.L)
    cps, curve, snaps = checkpoints(cfg.router_steps, n_checkpoints), [], []
    for t in progress_steps(cfg.router_steps + 1, "stage 1b: learned router"):
        if t in cps:
            with torch.no_grad():
                z, p, G = switch_forward(wr, X2, T2)
                curve.append((t, accuracy(z, y_t), eq.load_balance_loss(p, G)))
                snaps.append(G[:, show].clone())
        if t == cfg.router_steps:
            break
        x, tok, d, op, k, y = batch(data, 2 * M, cfg.batch, gen)
        z, p, G = switch_forward(wr, x, tok)
        loss = eq.cross_entropy(z, y) + cfg.balance_weight * eq.load_balance_loss(p, G)
        opt.zero_grad()
        loss.sum().backward()
        for q in kept:  # the given experts stay as they are (zero gradient ⇒ Adam leaves them unchanged)
            q.grad[:M] = 0
        opt.step()
    with torch.no_grad():
        z, p, G = switch_forward(wr, X2, T2)
    conf = torch.einsum("ne,mnf->mef", torch.nn.functional.one_hot(op_t, T.E).float(), G) / Nt  # true op × expert
    perms = list(itertools.permutations(range(T.E)))
    matched = torch.stack([conf[:, list(range(T.E)), list(q)].sum(1) for q in perms])  # (perms, 2M)
    best = matched.argmax(0)
    res.update({"router_confusion": conf, "router_recovery": matched.max(0).values,
                "router_identity": matched[0], "router_perm": [perms[int(b)] for b in best], "router_snaps": snaps,
                "show": show, "router_load": G.mean(1)})  # fmt: skip
    res["curves"]["router"] = curve
    acc_r = accuracy(z, y_t)
    res["acc"]["router_given"], res["acc"]["router"] = acc_r[:M], acc_r[M:]

    # stage 3: end-to-end fine-tuning through the whole DAG (fixed router), one backward pass through all stages
    wg = T.clone(w_split, grad=True)
    opt = torch.optim.Adam(T.leaves({k: wg[k] for k in ("img", "txt", "fuse", "cls")}), lr=cfg.lr_global)
    cps, curve = checkpoints(cfg.global_steps, n_checkpoints), []
    for t in progress_steps(cfg.global_steps + 1, "stage 3: global fine-tuning"):
        if t in cps:
            curve.append((t, end_acc(wg)))
        if t == cfg.global_steps:
            break
        x, tok, d, op, k, y = batch(data, M, cfg.batch, gen)
        G = T.router_matrix(tok.reshape(-1, T.L)).reshape(M, -1, T.E)
        loss = eq.cross_entropy(moe_forward(wg, x, tok, G, G), y)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    res["curves"]["global"] = curve
    w_glob = T.clone(wg)
    with torch.no_grad():
        z_glob, _ = run_dag(w_glob, xt, tok_t, dag)
        res["acc"]["global"] = accuracy(z_glob, y_t)
        res["digit_acc"] = {"split": a_img, "global": accuracy(eq.image_logits(w_glob["img"], Xt), yt.expand(M, Nt))}

    # stage 4: distil the whole DAG into one dense student (Hinton KD), then int8
    st = T.clone(T.init_student(M, cfg, torch.Generator().manual_seed(11)), grad=True)
    opt = torch.optim.Adam(T.leaves(st), lr=cfg.lr)
    cps, curve = checkpoints(cfg.student_steps, n_checkpoints), []
    student = lambda s, x, tok: eq.student_logits(s, x, tok, len(T.VOCAB))  # noqa: E731
    for t in progress_steps(cfg.student_steps + 1, "stage 4: distillation"):
        if t in cps:
            with torch.no_grad():
                curve.append((t, accuracy(student(st, Xt, Tt), y_t)))
        if t == cfg.student_steps:
            break
        x, tok, d, op, k, y = batch(data, M, 2 * cfg.batch, gen)
        with torch.no_grad():
            G = T.router_matrix(tok.reshape(-1, T.L)).reshape(M, -1, T.E)
            zt = moe_forward(w_glob, x, tok, G, G)
        zs = student(st, x, tok)
        loss = (1 - cfg.kd_hard) * eq.kd_loss(zs, zt, cfg.temperature) + cfg.kd_hard * eq.cross_entropy(zs, y)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    res["curves"]["student"] = curve
    st = T.clone(st)
    st_q = quantise(st)
    with torch.no_grad():
        z_st, z_q = student(st, Xt, Tt), student(st_q, Xt, Tt)
    res["acc"]["student"], res["acc"]["int8"] = accuracy(z_st, y_t), accuracy(z_q, y_t)
    res["agree_teacher"] = (z_st.argmax(-1) == z_glob.argmax(-1)).float().mean(-1)

    # size and time (per model, one thread)
    pipe_params = T.n_params({k: w_split[k] for k in ("img", "txt", "fuse", "cls")})
    res["params"] = {"split": pipe_params, "global": pipe_params, "student": T.n_params(st), "int8": T.n_params(st)}
    res["bytes"] = {"split": 4 * pipe_params, "global": 4 * pipe_params, "student": float_bytes(st),
                    "int8": int8_bytes(st)}  # fmt: skip
    timing = {k: [] for k in ("split", "global", "student", "int8", "moe", "one_by_one")}
    n_seq, rep = min(cfg.seq_samples, Nt), cfg.time_repeats
    with torch.no_grad():
        for m in range(M):
            ws, wgm, sm, qm = (T.select(v, m) for v in (w_split, w_glob, st, st_q))
            G1 = T.router_matrix(tok_t)[None]
            timing["split"].append(time_call(partial(run_dag, ws, xt, tok_t, dag), rep))
            timing["global"].append(time_call(partial(run_dag, wgm, xt, tok_t, dag), rep))
            timing["moe"].append(time_call(partial(moe_forward, ws, xt, tok_t, G1, G1), rep))
            timing["student"].append(time_call(partial(student, sm, xt[None], tok_t[None]), rep))
            timing["int8"].append(time_call(partial(student, qm, xt[None], tok_t[None]), rep))
            t1 = time_call(partial(run_jobs_one_by_one, ws, xt[:n_seq], tok_t[:n_seq]), max(1, rep // 2))
            timing["one_by_one"].append(t1 * Nt / n_seq)
    res["time"] = {k: torch.tensor(v) for k, v in timing.items()}

    # display: a few test examples, with the route each one takes
    res["examples"] = {"x": xt, "tok": tok_t, "op": op_t, "k": k_t, "d": yt, "y": y_t,
                       "pred_split": z_dag.argmax(-1), "pred_global": z_glob.argmax(-1),
                       "pred_student": z_st.argmax(-1), "pred_int8": z_q.argmax(-1)}  # fmt: skip
    res["G_fixed"] = T.router_matrix(tok_t)[show]
    return res
