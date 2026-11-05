r"""The laws as tensor operations: one step of B Snake games, the network's view, the teacher's breadth-first search,
the CNN, and the mixture-of-experts controller (dispatch → experts → combine, the layer of moe_dag_pipeline).

Indices: m = model (variant × seed), b = game, r, c = board cell, i, j = view cell, k = expert, s = slot in an
expert's buffer, f = CNN feature, h = hidden unit, a = action. Repeated indices are summed unless in the output.

    move           d'_b = (d_b + a_b − 1) mod 4,   x'_b = x_b + Δ_{d'_b}            (relative action a ∈ {0, 1, 2})
    collision      dead_b = [x'_b outside the board] ∨ [age_{b, x'_b} > 1]    (age 1 = the tail, which moves away)
    body           age'_{brc} = max(age_{brc} − 1, 0) (no food) or age_{brc} (food: the tail stays),
                   age'_{b, x'_b} = length'_b,   length'_b = length_b + [x'_b = food_b]
    food respawn   food'_b = argmax_{rc} u_{brc} [age'_{brc} = 0]          (u uniform from the seeded generator)
    view           v_{bij} = board_b(x_b + (R − i) Δ_{d_b} + (j − R) Δ_{d_b + 1}),  R = N − 1  (head-centred)
    distance       dist ← min(dist, 1 + min_{neighbours} dist) on free cells until fixed       (BFS by relaxation)
    flood fill     label ← min(label, min_{neighbours} label): one label per free region, area = its cell count
    CNN            h_{mbf} = relu(W^{fc}_{m·f} · relu(conv_2(relu(conv_1(v_{mb})))) + b_{mf})
    router         p_{mbk} = softmax_k(h_{mbf} W^G_{mfk} + b^G_{mk})
    expert         y_{mkba} = W2_{mkha} relu(W1_{mkfh} h_{mbf} + b1_{mkh}) + b2_{mka}
    dense          z_{mba} = p_{mbk} y_{mkba}
    top-1 (Switch) G_{mbk} = onehot(argmax_k p_{mbk}),  D_{mbks} = G_{mbk} [s = rank of b in group k],
                   U_{mksf} = D_{mbks} h_{mbf},  Y_{mksa} = f_k(U_{mks}),  z_{mba} = D_{mbks} G_{mbk} p_{mbk} Y_{mksa}
    load balance   ℒ_aux = K Σ_k f_k P_k,  f_k = ⟨G_{bk}⟩_b,  P_k = ⟨p_{bk}⟩_b                 (Fedus et al. 2021)
    imitation      ℒ = −⟨log softmax_a z_{mba}[a*_b]⟩_b   (a* = the teacher's action; behaviour cloning)

dispatch, combine, top1 and load_balance_loss are imported from problems/ai/moe_dag_pipeline.
"""

import torch
import torch.nn.functional as F

from problems.ai.moe_dag_pipeline.equation import combine, cross_entropy, dispatch, load_balance_loss, top1

from . import tensors as T

__all__ = ["combine", "cross_entropy", "dispatch", "load_balance_loss", "top1"]  # re-exported for the tests
FAR = 10**6  # distance of an unreachable cell


# ---------------------------------------------------------------- the game
def _cell(age: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
    """age (B, N, N) at cells rc (B, ..., 2) (clamped inside the board) → (B, ...)."""
    n = age.shape[1]
    rc = rc.clamp(0, n - 1)
    idx = (rc[..., 0] * n + rc[..., 1]).reshape(len(rc), -1)
    return age.flatten(1).gather(1, idx).view(rc.shape[:-1])


def _inside(rc: torch.Tensor, n: int) -> torch.Tensor:
    return ((rc >= 0) & (rc < n)).all(-1)


def place_food(age: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A uniformly random empty cell per game: argmax of u over the empty cells → food (B, 2), any_empty (B,)."""
    n = age.shape[1]
    empty = age.flatten(1) == 0
    k = torch.where(empty, u, torch.full_like(u, -1.0)).argmax(1)
    return torch.stack([k // n, k % n], 1), empty.any(1)


def reset(B: int, n: int, gen: torch.Generator) -> T.State:
    """B new games: a 3-long snake with its head in the board's interior, a random heading, the body behind it."""
    head = torch.randint(2, n - 2, (B, 2), generator=gen)
    d = torch.randint(4, (B,), generator=gen)
    k = torch.arange(T.START_LENGTH)
    body = head[:, None] - k[None, :, None] * T.DIRS[d][:, None]  # (B, 3, 2) head, neck, tail
    age = torch.zeros(B, n * n, dtype=torch.long)
    age.scatter_(1, body[..., 0] * n + body[..., 1], (T.START_LENGTH - k).expand(B, -1).contiguous())
    age = age.view(B, n, n)
    food, _ = place_food(age, torch.rand(B, n * n, generator=gen))
    z = torch.zeros(B, dtype=torch.long)
    return T.State(age=age, head=head, dir=d, food=food, length=torch.full((B,), T.START_LENGTH),
                   alive=torch.ones(B, dtype=torch.bool), hunger=z, eaten=z.clone(), steps=z.clone())  # fmt: skip


def step(s: T.State, action: torch.Tensor, u: torch.Tensor, hunger_limit: int) -> T.State:
    """All B games one move: action (B,) relative, u (B, N²) uniform numbers for the food respawn. Games that have
    ended stay frozen. A game ends on a wall, on its own body, after hunger_limit moves without food, or when the
    snake fills the board."""
    B, n, _ = s.age.shape
    d = (s.dir + action - 1) % 4
    new = s.head + T.DIRS.to(s.head.device)[d]  # (B, 2)
    inside = _inside(new, n)
    hit = ~inside | (_cell(s.age, new) > 1)  # the tail (age 1) leaves this step, so its cell is free
    eat = inside & (new == s.food).all(-1)
    hunger = torch.where(eat, 0, s.hunger + 1)
    dies = s.alive & (hit | (hunger >= hunger_limit))
    moves = s.alive & ~dies
    grow = moves & eat
    length = s.length + grow
    age = torch.where((moves & ~eat)[:, None, None], (s.age - 1).clamp(min=0), s.age)
    nc = new.clamp(0, n - 1)
    flat = age.flatten(1).clone()
    idx = (nc[:, 0] * n + nc[:, 1])[:, None]
    flat.scatter_(1, idx, torch.where(moves, length, flat.gather(1, idx)[:, 0])[:, None])
    age = flat.view(B, n, n)
    food_new, room = place_food(age, u)
    m2 = moves[:, None]
    return T.State(
        age=age,
        head=torch.where(m2, nc, s.head),
        dir=torch.where(moves, d, s.dir),
        food=torch.where((grow & room)[:, None], food_new, s.food),
        length=length,
        alive=moves & ~(grow & ~room),  # a full board ends the game (a win)
        hunger=torch.where(moves, hunger, s.hunger),
        eaten=s.eaten + grow,
        steps=s.steps + moves,
    )


# ---------------------------------------------------------------- what the network sees
def view_cells(s: T.State) -> torch.Tensor:
    """Board cell under each view cell: (B, W, W, 2) with W = 2N − 1, centred on the head, heading pointing up."""
    n = s.age.shape[1]
    R = n - 1
    i = torch.arange(2 * R + 1, device=s.age.device)
    ahead = (R - i)[:, None, None]  # (W, 1, 1) rows above the centre are ahead
    right = (i - R)[None, :, None]  # (1, W, 1) columns right of the centre are to the right
    dirs = T.DIRS.to(s.age.device)
    fwd, rgt = dirs[s.dir][:, None, None], dirs[(s.dir + 1) % 4][:, None, None]  # (B, 1, 1, 2)
    return s.head[:, None, None] + ahead * fwd + right * rgt


def planes(s: T.State) -> torch.Tensor:
    """The CNN input (B, 4, W, W): wall (outside the board), body, age/length (1 at the head, 1/length at the
    tail: which cells free up first) and food."""
    n = s.age.shape[1]
    cells = view_cells(s)
    inside = _inside(cells, n)
    age = torch.where(inside, _cell(s.age, cells), 0).float()
    food = (cells == s.food[:, None, None]).all(-1) & inside
    return torch.stack([(~inside).float(), (age > 0).float(), age / s.length[:, None, None], food.float()], 1)


# ---------------------------------------------------------------- the teacher's search (tensor BFS)
def min_propagate(v: torch.Tensor, inc: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
    """Breadth-first search as repeated relaxation, all games and channels at once: on free cells
    v ← min(v, min over the 4 neighbours of v + inc) until nothing changes. v (B, Q, N, N) long, inc (Q,),
    free (B, N, N) bool. inc = 1: distances from the cells where v = 0; inc = 0: the smallest label of each
    connected free region (a flood fill)."""
    inc = inc.view(1, -1, 1, 1)
    blocked = ~free[:, None]
    while True:
        p = F.pad(v, (1, 1, 1, 1), value=FAR)
        nb = torch.minimum(torch.minimum(p[..., :-2, 1:-1], p[..., 2:, 1:-1]),
                           torch.minimum(p[..., 1:-1, :-2], p[..., 1:-1, 2:]))  # fmt: skip
        new = torch.where(blocked, v, torch.minimum(v, nb + inc))
        if torch.equal(new, v):
            return v
        v = new


def distance_field(start: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
    """Steps from the start cells to every cell through free cells: (B, N, N) bool, bool → (B, N, N) long, FAR
    where unreachable."""
    v = torch.where(start, 0, FAR)[:, None]
    return min_propagate(v, torch.ones(1, dtype=torch.long, device=v.device), free)[:, 0]


def search(start: torch.Tensor, free: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Both searches of the teacher in one relaxation: distance from the food and the label of each free region.
    → dist (B, N, N), label (B, N, N) (FAR outside free cells)."""
    B, n, _ = free.shape
    cell = torch.arange(n * n, device=free.device).view(1, n, n)
    v = torch.stack([torch.where(start, 0, FAR), torch.where(free, cell, FAR)], 1)
    v = min_propagate(v, torch.tensor([1, 0], device=free.device), free)
    return v[:, 0], v[:, 1]


def candidates(s: T.State) -> torch.Tensor:
    """Cell the head would enter for each relative action: (B, 3, 2)."""
    d = (s.dir[:, None] + torch.arange(3, device=s.dir.device) - 1) % 4
    return s.head[:, None] + T.DIRS.to(s.head.device)[d]


# ---------------------------------------------------------------- the networks, model axis m first
def cnn_features(x: torch.Tensor, w: dict) -> torch.Tensor:
    """Views x (M, B, 4, W, W), model m sees its own games → h (M, B, F). The model axis becomes the groups of one
    grouped convolution per layer, so the M networks never mix (as in stitch_llm_cnn_dag)."""
    M, B = x.shape[:2]
    z = x.transpose(0, 1).flatten(1, 2)  # (B, M·4, W, W)
    z = F.relu(F.conv2d(z, w["c1"].flatten(0, 1), w["b1"].flatten(), padding=1, groups=M))  # (B, 8M, W, W)
    z = F.relu(F.conv2d(z, w["c2"].flatten(0, 1), w["b2"].flatten(), stride=2, groups=M))  # (B, 16M, S, S)
    z = z.view(B, M, -1).transpose(0, 1)  # (M, B, 16·S·S)
    return F.relu(torch.einsum("mbi,mif->mbf", z, w["fc"]) + w["bf"][:, None])


def router_probs(h: torch.Tensor, router: dict) -> torch.Tensor:
    """p (M, B, K) = softmax(h W^G + b^G): how much each game wants each expert."""
    return torch.softmax(torch.einsum("mbf,mfk->mbk", h, router["W"]) + router["b"][:, None], -1)


def experts_on(U: torch.Tensor, ex: dict) -> torch.Tensor:
    """All experts at once on their own inputs: U (M, K, S, F) → action logits Y (M, K, S, 3)."""
    hid = torch.relu(torch.einsum("mksf,mkfh->mksh", U, ex["W1"]) + ex["b1"][:, :, None])
    return torch.einsum("mksh,mkha->mksa", hid, ex["W2"]) + ex["b2"][:, :, None]


def moe_dense(h: torch.Tensor, g: torch.Tensor, w: dict) -> torch.Tensor:
    """Dense mixture: every expert on every game, z_{mba} = g_{mbk} y_{mkba}.  h (M, B, F), g (M, B, K) → (M, B, 3)."""
    K = g.shape[-1]
    Y = experts_on(h[:, None].expand(-1, K, -1, -1), w["experts"])  # (M, K, B, 3)
    return torch.einsum("mbk,mkba->mba", g, Y)


def moe_top1(h: torch.Tensor, G: torch.Tensor, g: torch.Tensor, w: dict) -> torch.Tensor:
    """Sparse layer: each game runs only its routed expert. G (M, B, K) one-hot, g (M, B, K) gate values (G for a
    fixed router, G ⊙ p for Switch). Dispatch into per-expert buffers, one batched expert call, combine."""
    cap = max(1, int(G.sum(1).max()))  # the largest group: no game is dropped
    D = dispatch(G, cap)  # (M, B, K, S)
    U = torch.einsum("mbks,mbf->mksf", D, h)  # each expert's buffer of games
    return combine(D, g, experts_on(U, w["experts"]))  # (M, B, 3)


def mono_logits(h: torch.Tensor, w: dict) -> torch.Tensor:
    """The monolithic controller, one MLP: h (M, B, F) → (M, B, 3)."""
    hid = torch.relu(torch.einsum("mbf,mfh->mbh", h, w["W1"]) + w["b1"][:, None])
    return torch.einsum("mbh,mha->mba", hid, w["W2"]) + w["b2"][:, None]
