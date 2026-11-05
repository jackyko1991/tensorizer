"""Solvers: the planning teacher, batched rollouts, imitation learning of all controllers at once (one model axis),
one DAgger round, evaluation on shared games, and the routing analysis.

Training layout on the model axis (S seeds each): [dense MoE × S, Switch MoE × S, monolithic × S]. Evaluation adds
(c): the dense-trained MoE run with top-1 routing, so the evaluated models are [dense, Switch, dense→top-1, mono].
"""

import math
import time

import torch

from problems._common.progress import steps as progress_steps

from . import equation as eq
from . import tensors as T


# ---------------------------------------------------------------- the teacher
def teacher_actions(s: T.State) -> torch.Tensor:
    """A planner: among the three moves, never a deadly one, prefer a move whose flood-fill area can hold the
    snake, then the shortest path to the food (tensor BFS), then the largest area. → actions (B,)."""
    B, n, _ = s.age.shape
    free = s.age <= 1  # the tail cell is free next step
    cand = eq.candidates(s)  # (B, 3, 2)
    inside = eq._inside(cand, n)
    safe = inside & (eq._cell(s.age, cand) <= 1)
    start = torch.zeros_like(free)
    start[torch.arange(B, device=s.age.device), s.food[:, 0], s.food[:, 1]] = True
    dist_map, label = eq.search(start, free)
    dist = eq._cell(dist_map, cand)  # (B, 3) steps from each candidate to the food
    lab = torch.where(safe, eq._cell(label, cand), -1)  # free region each candidate enters
    area = (label.flatten(1)[:, None, :] == lab[..., None]).sum(-1)  # (B, 3) cells in that region
    roomy = area >= s.length[:, None]
    straight = torch.tensor([0.0, 0.5, 0.0], device=s.age.device)
    key = 4e6 * safe + 2e6 * roomy - 1e3 * dist.clamp(max=999) + area + straight
    return key.argmax(1)


def random_actions(B: int, gen: torch.Generator, dev) -> torch.Tensor:
    return torch.randint(3, (B,), generator=gen).to(dev)


# ---------------------------------------------------------------- networks on the model axis
def sub(w: dict, a: int, b: int) -> dict:
    return T.tmap(w, lambda t: t[a:b])


def forward(x: torch.Tensor, P: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Views x (M, B, 4, W, W) → action logits z (M, B, 3), router p and features h of the MoE models.
    P['layout'] = (n_dense, n_top1, n_mono): models in that order on the model axis."""
    nd, nt, nm = P["layout"]
    h = eq.cnn_features(x, P["cnn"])
    hm = h[: nd + nt]
    p = eq.router_probs(hm, P["moe"]["router"])
    zs = []
    if nd:
        zs.append(eq.moe_dense(hm[:nd], p[:nd], sub(P["moe"], 0, nd)))
    if nt:
        G = eq.top1(p[nd:])
        zs.append(eq.moe_top1(hm[nd:], G, G * p[nd:], sub(P["moe"], nd, nd + nt)))
    if nm:
        zs.append(eq.mono_logits(h[nd + nt :], P["mono"]))
    return torch.cat(zs), p, hm


def init_models(cfg: T.Config, dev) -> dict:
    """Seed s gets the same CNN and MoE initialisation in every variant (a paired comparison)."""
    S, K, F, H = cfg.seeds, cfg.experts, cfg.features, cfg.hidden
    cnn = [T.init_cnn(1, F, cfg.grid, torch.Generator().manual_seed(100 + s)) for s in range(S)]
    moe = [T.init_moe(1, K, F, H, torch.Generator().manual_seed(200 + s)) for s in range(S)]
    mono = [T.init_mono(1, F, T.mono_hidden(cfg), torch.Generator().manual_seed(300 + s)) for s in range(S)]
    P = {"cnn": T.stack(cnn * 3), "moe": T.stack(moe * 2), "mono": T.stack(mono)}
    P = {k: T.clone(T.to(v, dev), grad=True) for k, v in P.items()}
    P["layout"] = (S, S, S)
    return P


def eval_models(P: dict, S: int) -> dict:
    """[dense, Switch, dense→top-1 (the dense weights, routed top-1), mono] for evaluation."""
    idx = list(range(S)) + list(range(S, 2 * S)) + list(range(S))
    cnn_idx = idx + list(range(2 * S, 3 * S))
    return {"cnn": T.tmap(P["cnn"], lambda t: t.detach()[cnn_idx]), "moe": T.tmap(P["moe"], lambda t: t.detach()[idx]),
            "mono": T.tmap(P["mono"], lambda t: t.detach()), "layout": (S, 2 * S, S)}  # fmt: skip


# ---------------------------------------------------------------- data: states the teacher labels
POOL_KEYS = ("age", "head", "dir", "food", "length")
EVAL_SEED = 12345


def pool_state(pool: dict, idx: torch.Tensor) -> T.State:
    """Rows idx of a pool of recorded states as a State (only what the view and the teacher need)."""
    z = torch.zeros(len(idx), dtype=torch.long, device=idx.device)
    st = {k: pool[k][idx].long() for k in POOL_KEYS}
    return T.State(**st, alive=torch.ones_like(z, dtype=torch.bool), hunger=z, eaten=z, steps=z)


def add_to_pool(pool: dict, s: T.State, keep: torch.Tensor, action: torch.Tensor | None = None) -> None:
    for k in POOL_KEYS:
        v = getattr(s, k)[keep]
        pool.setdefault(k, []).append(v.to(torch.int16) if k == "age" else v)
    if action is not None:
        pool.setdefault("action", []).append(action[keep])


def cat_pool(pool: dict) -> dict:
    return {k: torch.cat(v) for k, v in pool.items()}


def label(pool: dict, chunk: int = 8192) -> dict:
    """The teacher's action for every recorded state (in chunks: one batched search each)."""
    n = len(pool["dir"])
    idx = torch.arange(n, device=pool["dir"].device)
    pool["action"] = torch.cat([teacher_actions(pool_state(pool, idx[i : i + chunk])) for i in range(0, n, chunk)])
    return pool


def merge(a: dict, b: dict) -> dict:
    return {k: torch.cat([a[k], b[k]]) for k in a}


def cat_states(states: list[T.State]) -> T.State:
    return T.State(**{k: torch.cat([getattr(s, k) for s in states]) for k in vars(states[0])})


def repeat_state(s: T.State, M: int) -> T.State:
    return T.State(**{k: v.repeat(M, *[1] * (v.dim() - 1)) for k, v in vars(s).items()})


def noise(seed: int, B: int, n: int, repeat: int = 1):
    """The food-respawn numbers u (B·repeat, N²) for step t: the same stream for every copy of a game."""
    g = torch.Generator().manual_seed(seed)
    return lambda t: torch.rand(B, n * n, generator=g).repeat(repeat, 1)


def play(cfg: T.Config, s: T.State, act, u_of, n_steps: int, on_step=None) -> T.State:
    """Run games with act(state) → actions until all have ended or n_steps; on_step(t, state, actions)."""
    for t in range(n_steps):
        if not bool(s.alive.any()):
            break
        a = act(s)
        if on_step is not None:
            on_step(t, s, a)
        s = eq.step(s, a, u_of(t).to(s.age.device), cfg.hunger_limit())
    return s


def eval_start(cfg: T.Config) -> T.State:
    return eq.reset(cfg.eval_games, cfg.grid, torch.Generator().manual_seed(EVAL_SEED))


def teacher_run(cfg: T.Config, dev) -> tuple[dict, T.State]:
    """The teacher plays its training games (recorded with its actions, for at most teacher_steps moves) and the
    evaluation games in ONE batch: a search costs about the same for 100 games as for 1000 on a CPU."""
    Bd, n = cfg.teacher_games, cfg.grid
    s = cat_states([eq.reset(Bd, n, torch.Generator().manual_seed(0)), eval_start(cfg)]).to(dev)
    ud, ue = noise(1, Bd, n), noise(EVAL_SEED + 1, cfg.eval_games, n)
    pool: dict = {}

    def record(t, st, a):
        if t < cfg.teacher_steps:
            keep = st.alive.clone()
            keep[Bd:] = False
            add_to_pool(pool, st, keep, a)

    s = play(cfg, s, teacher_actions, lambda t: torch.cat([ud(t), ue(t)]), cfg.eval_steps, record)
    return cat_pool(pool), s.index(slice(Bd, None))


def student_pool(cfg: T.Config, P: dict, dev, seed: int) -> dict:
    """DAgger: the students play greedily; the teacher labels the states they reach."""
    M, B, n = sum(P["layout"]), cfg.dagger_games, cfg.grid
    s = repeat_state(eq.reset(B, n, torch.Generator().manual_seed(seed)), M).to(dev)
    pool: dict = {}

    def act(st):
        with torch.no_grad():
            return forward(eq.planes(st).view(M, B, T.C, 2 * n - 1, 2 * n - 1), P)[0].argmax(-1).flatten()

    play(cfg, s, act, noise(seed + 1, B, n, M), cfg.dagger_steps, lambda t, st, a: add_to_pool(pool, st, st.alive))
    return label(cat_pool(pool))


# ---------------------------------------------------------------- imitation learning
def train(cfg: T.Config, P: dict, pool: dict, n_steps: int, gen: torch.Generator, t0: int = 0, opt=None):
    """Cross-entropy on the teacher's actions (+ α ℒ_aux for the MoE models), every model its own minibatches,
    AdamW (elementwise, so the models stay independent) with warm-up and cosine decay over cfg.steps.
    Returns the loss and accuracy histories (steps, M) and the optimiser."""
    S, W = cfg.seeds, 2 * cfg.grid - 1
    M = 3 * S
    if opt is None:
        opt = torch.optim.AdamW([t for k in ("cnn", "moe", "mono") for t in T.leaves(P[k])], lr=cfg.lr, weight_decay=0)
    n_pool = len(pool["dir"])
    losses, accs = [], []
    for t in progress_steps(n_steps, "imitation"):
        tt = t0 + t
        for g in opt.param_groups:
            g["lr"] = cfg.lr * min(1.0, (tt + 1) / 30) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * tt / cfg.steps)))
        rows = torch.randint(n_pool, (M, cfg.batch), generator=gen).to(pool["dir"].device)
        x = eq.planes(pool_state(pool, rows.flatten())).view(M, cfg.batch, T.C, W, W)
        y = pool["action"][rows]
        z, p, _ = forward(x, P)
        ce = eq.cross_entropy(z, y)
        aux = eq.load_balance_loss(p, eq.top1(p))  # both MoE variants get the Switch balancing term
        loss = ce + cfg.balance_weight * torch.cat([aux, torch.zeros(S, device=aux.device)])
        opt.zero_grad()
        loss.sum().backward()  # model m's loss reaches only model m's weights
        opt.step()
        losses.append(ce.detach().cpu())
        accs.append((z.argmax(-1) == y).float().mean(1).detach().cpu())
    return torch.stack(losses), torch.stack(accs), opt


# ---------------------------------------------------------------- evaluation and routing
def free_ahead(x: torch.Tensor) -> torch.Tensor:
    """Free cells straight ahead of the head before a wall or body cell, from the views (B, 4, W, W) → (B,)."""
    R = x.shape[-1] // 2
    col = x[:, :, :R, R].flip(-1)  # (B, 4, R) cells ahead, nearest first
    free = (col[:, T.WALL] == 0) & (col[:, T.BODY] == 0)
    return free.long().cumprod(-1).sum(-1)


def evaluate(cfg: T.Config, P: dict, dev) -> dict:
    """The 4S networks and a random policy on the evaluation games (same start, same food stream as the teacher's
    run). Records food eaten, moves survived, the routing of the 3S routed models and the first games for the GIF."""
    S, B, n = cfg.seeds, cfg.eval_games, cfg.grid
    W = 2 * n - 1
    s0 = eval_start(cfg)
    out = {"food": {}, "steps": {}}
    ga = torch.Generator().manual_seed(EVAL_SEED + 2)
    s = play(cfg, s0.to(dev), lambda st: random_actions(len(st.dir), ga, dev), noise(EVAL_SEED + 1, B, n),
             cfg.eval_steps)  # fmt: skip
    out["food"]["random"], out["steps"]["random"] = s.eaten.float().cpu(), s.steps.float().cpu()

    E = eval_models(P, S)
    M, nr = sum(E["layout"]), 3 * S  # 4S models, the first 3S routed
    show = min(cfg.show_games, B)
    rec = {k: [] for k in ("model", "expert", "dist", "ahead", "length", "turn")}
    frames = {k: [] for k in ("age", "food", "alive", "length", "p")}

    def act(st):
        with torch.no_grad():
            x = eq.planes(st)
            z, p, _ = forward(x.view(M, B, T.C, W, W), E)
        a = z.argmax(-1)  # (M, B)
        alive = st.alive.view(M, B)[:nr]
        head, food = st.head.view(M, B, 2)[:nr], st.food.view(M, B, 2)[:nr]
        feats = {"model": torch.arange(nr, device=dev)[:, None].expand(nr, B), "expert": p.argmax(-1),
                 "dist": (head - food).abs().sum(-1), "ahead": free_ahead(x).view(M, B)[:nr],
                 "length": st.length.view(M, B)[:nr], "turn": a[:nr]}  # fmt: skip
        for k, v in feats.items():
            rec[k].append(v[alive].cpu())
        for k in ("age", "food", "alive", "length"):
            v = getattr(st, k)
            frames[k].append(v.view(M, B, *v.shape[1:])[:nr, :show].cpu())
        frames["p"].append(p[:, :show].cpu())
        return a.flatten()

    s = play(cfg, repeat_state(s0, M).to(dev), act, noise(EVAL_SEED + 1, B, n, M), cfg.eval_steps)
    eaten, moves = s.eaten.view(M, B).float().cpu(), s.steps.view(M, B).float().cpu()
    for j, v in enumerate(T.VARIANTS):
        out["food"][v], out["steps"][v] = eaten[j * S : (j + 1) * S], moves[j * S : (j + 1) * S]
    out["routing"] = {k: torch.cat(v) for k, v in rec.items()}
    out["frames"] = {k: torch.stack(v) for k, v in frames.items()}  # (T, nr, show, ...)
    return out


BINS = {"dist": [2, 4, 7], "ahead": [1, 2, 4], "length": [5, 9, 15]}  # bin edges: ≤1, 2–3, 4–6, ≥7 …


def specialisation(routing: dict, nr: int, K: int) -> dict:
    """Per routed model: expert loads, P(expert | feature bin) and the mutual information (bits) between the routed
    expert and binned state features (distance to food, free cells ahead, body length) and the move the controller
    makes. 0 bits = the routing tells nothing about that feature; log2 K bits is the most it can."""
    feats = (*BINS, "turn")
    out = {"load": torch.zeros(nr, K), "mi": {k: torch.zeros(nr) for k in feats},
           "table": {k: torch.zeros(nr, 3 if k == "turn" else 4, K) for k in feats}}  # fmt: skip
    for m in range(nr):
        sel = routing["model"] == m
        e = routing["expert"][sel]
        if len(e) == 0:
            continue
        out["load"][m] = torch.bincount(e, minlength=K).float() / len(e)
        for k in feats:
            v = routing[k][sel]
            b = v if k == "turn" else torch.bucketize(v, torch.tensor(BINS[k]), right=True)
            joint = torch.zeros(out["table"][k].shape[1], K)
            joint.index_put_((b, e), torch.ones(len(e)), accumulate=True)
            out["table"][k][m] = joint / joint.sum(1, keepdim=True).clamp(min=1)  # P(expert | bin)
            pj = joint / joint.sum()
            indep = pj.sum(1, keepdim=True) * pj.sum(0, keepdim=True)
            nz = pj > 0
            out["mi"][k][m] = (pj[nz] * torch.log2(pj[nz] / indep[nz])).sum()
    return out


# ---------------------------------------------------------------- everything
def run_all(cfg: T.Config, dev) -> dict:
    tm = {}
    t0 = time.time()
    pool, s_teacher = teacher_run(cfg, dev)
    tm["teacher (data + eval)"] = time.time() - t0
    P = init_models(cfg, dev)
    gen = torch.Generator().manual_seed(7)
    first = int(cfg.steps * 0.6) if cfg.dagger else cfg.steps
    t1 = time.time()
    loss, acc, opt = train(cfg, P, pool, first, gen)
    n_teacher = len(pool["dir"])
    if cfg.dagger:
        t2 = time.time()
        pool = merge(pool, student_pool(cfg, P, dev, seed=1))
        tm["DAgger data"] = time.time() - t2
        l2, a2, _ = train(cfg, P, pool, cfg.steps - first, gen, first, opt)
        loss, acc = torch.cat([loss, l2]), torch.cat([acc, a2])
    tm["training"] = time.time() - t1 - tm.get("DAgger data", 0.0)
    t3 = time.time()
    ev = evaluate(cfg, P, dev)
    tm["evaluation"] = time.time() - t3
    ev["food"]["teacher"], ev["steps"]["teacher"] = s_teacher.eaten.float().cpu(), s_teacher.steps.float().cpu()
    return {"cfg": cfg, "loss": loss, "acc": acc, "dagger_at": first if cfg.dagger else None,
            "pool": (n_teacher, len(pool["dir"])), **ev, "spec": specialisation(ev["routing"], 3 * cfg.seeds,
            cfg.experts), "time": tm, "params": {k: T.n_params(P[k]) for k in ("cnn", "moe", "mono")}}  # fmt: skip
