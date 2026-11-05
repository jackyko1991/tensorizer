"""Checks written first (TDD): a batched tensor Snake game, a planning teacher and a mixture-of-experts controller.

Run with `pytest problems/ai/snake_moe_control/test.py`. Exact checks: wall and self collision end a game, moving
into the cell the tail is leaving is allowed, eating grows the snake by one, food never spawns on the body, the body
always holds the ages 1..length, and one batched step for B games equals a plain per-game Python loop given the same
random numbers. The head-centred, heading-up view puts the three relative moves next to the centre, and a move is
deadly exactly when that neighbour is a wall or a body cell that will not have moved away. The teacher's distance
field equals a breadth-first search, and it never picks a deadly move when a safe one exists. The MoE layer with a
fixed one-hot router equals running each game's chosen expert on its own; a uniform dense router gives the mean of
the experts; router rows sum to 1; the Switch load-balancing loss on known cases; the monolithic controller has the
same number of weights. Slow: a short training run plays far better than a random policy.
"""

from collections import deque

import pytest
import torch

from . import equation as eq
from . import solve
from . import tensors as T

N = 8


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _state(bodies, food, dirs, n=N):
    """Games from explicit bodies (head first, list of (row, col)), food cells and headings."""
    B = len(bodies)
    age = torch.zeros(B, n, n, dtype=torch.long)
    for b, body in enumerate(bodies):
        for k, (r, c) in enumerate(body):
            age[b, r, c] = len(body) - k
    return T.State(
        age=age,
        head=torch.tensor([body[0] for body in bodies]),
        dir=torch.tensor(dirs),
        food=torch.tensor(food),
        length=torch.tensor([len(body) for body in bodies]),
        alive=torch.ones(B, dtype=torch.bool),
        hunger=torch.zeros(B, dtype=torch.long),
        eaten=torch.zeros(B, dtype=torch.long),
        steps=torch.zeros(B, dtype=torch.long),
    )


def _step(s, actions, seed=0, hunger_limit=1000):
    u = torch.rand(len(actions), s.age.shape[1] ** 2, generator=_gen(seed))
    return eq.step(s, torch.tensor(actions), u, hunger_limit)


# ---------------------------------------------------------------- the game
def test_wall_collision_ends_the_game():
    # heading up (dir 0) on the top row, and heading right (dir 1) on the right edge
    s = _state([[(0, 3), (1, 3), (2, 3)], [(4, N - 1), (4, N - 2), (4, N - 3)]], [(5, 5), (0, 0)], [0, 1])
    s2 = _step(s, [T.STRAIGHT, T.STRAIGHT])
    assert not s2.alive.any()
    assert torch.equal(s2.length, s.length) and torch.equal(s2.age, s.age)  # a dead game is frozen
    s3 = _step(s, [T.RIGHT, T.LEFT])  # turning away from the wall instead is fine
    assert s3.alive.all()


def test_self_collision_ends_the_game_but_the_leaving_tail_cell_is_free():
    # a 5-long snake curled so that a right turn hits its own body (age 3), heading left (dir 3)
    body = [(3, 3), (3, 4), (2, 4), (2, 3), (2, 2)]  # head (3,3); turning right from "left" = up → (2,3), age 2
    s = _state([body, body], [(6, 6), (6, 6)], [3, 3])
    s2 = _step(s, [T.RIGHT, T.STRAIGHT])
    assert not bool(s2.alive[0]) and bool(s2.alive[1])
    # a 4-long loop: the head moves into the cell the tail leaves in the same step → allowed
    loop = [(3, 3), (3, 4), (2, 4), (2, 3)]  # tail (2,3) has age 1; heading left, turn right = up → (2,3)
    s = _state([loop], [(6, 6)], [3])
    s2 = _step(s, [T.RIGHT])
    assert bool(s2.alive[0]) and int(s2.age[0, 2, 3]) == 4


def test_eating_grows_by_one_and_food_respawns_off_the_body():
    s = _state([[(4, 4), (4, 3), (4, 2)]], [(4, 5)], [1])  # food straight ahead
    s2 = _step(s, [T.STRAIGHT])
    assert int(s2.length[0]) == 4 and int(s2.eaten[0]) == 1 and int(s2.hunger[0]) == 0
    assert int((s2.age[0] > 0).sum()) == 4  # the tail stayed
    fr, fc = s2.food[0].tolist()
    assert int(s2.age[0, fr, fc]) == 0 and (fr, fc) != (4, 5)
    s3 = _step(s2, [T.STRAIGHT])  # no food: the length stays and the tail moves
    assert int(s3.length[0]) == 4 and int((s3.age[0] > 0).sum()) == 4


def _random_games(B=64, n=N, seed=0):
    return eq.reset(B, n, _gen(seed))


def test_food_never_on_the_body_and_the_body_holds_ages_one_to_length():
    g = _gen(3)
    s = _random_games(256, seed=1)
    for _ in range(120):
        a = torch.randint(3, (256,), generator=g)
        s = eq.step(s, a, torch.rand(256, N * N, generator=g), hunger_limit=60)
        on = s.age.flatten(1).gather(1, (s.food[:, 0] * N + s.food[:, 1])[:, None])[:, 0]
        assert (on[s.alive] == 0).all()
        srt = s.age.flatten(1).sort(1, descending=True).values  # ages, largest first
        k = torch.arange(N * N)
        expect = (s.length[:, None] - k).clamp(min=0)
        assert torch.equal(srt, expect)  # exactly the ages length, length − 1, …, 1 and zeros
        hr, hc = s.head[:, 0], s.head[:, 1]
        assert torch.equal(s.age[torch.arange(256), hr, hc], s.length)  # the head carries age = length


def _step_reference(s: T.State, actions, u, hunger_limit):
    """One game at a time, with Python lists and ifs (the slow reference)."""
    n = s.age.shape[1]
    out = {k: v.clone() for k, v in vars(s).items()}
    for b in range(len(actions)):
        if not bool(s.alive[b]):
            continue
        age = s.age[b].tolist()
        d = (int(s.dir[b]) + int(actions[b]) - 1) % 4
        dr, dc = T.DIRS[d].tolist()
        r, c = int(s.head[b, 0]) + dr, int(s.head[b, 1]) + dc
        hunger = int(s.hunger[b]) + 1
        length = int(s.length[b])
        if not (0 <= r < n and 0 <= c < n) or age[r][c] > 1:
            out["alive"][b] = False
            continue
        eat = [r, c] == s.food[b].tolist()
        if eat:
            hunger, length = 0, length + 1
        if hunger >= hunger_limit:
            out["alive"][b] = False
            continue
        if not eat:
            age = [[max(v - 1, 0) for v in row] for row in age]
        age[r][c] = length
        out["age"][b] = torch.tensor(age)
        out["head"][b] = torch.tensor([r, c])
        out["dir"][b], out["length"][b], out["hunger"][b] = d, length, hunger
        out["steps"][b] += 1
        if eat:
            out["eaten"][b] += 1
            empty = [i for i in range(n * n) if age[i // n][i % n] == 0]
            if not empty:
                out["alive"][b] = False  # the board is full
                continue
            best = max(empty, key=lambda i: float(u[b, i]))
            out["food"][b] = torch.tensor([best // n, best % n])
    return T.State(**out)


def test_batched_step_equals_a_per_game_reference_loop():
    g = _gen(7)
    s = _random_games(48, seed=2)
    for _ in range(80):
        a = torch.randint(3, (48,), generator=g)
        a = torch.where(torch.rand(48, generator=g) < 0.6, solve.teacher_actions(s), a)  # live longer: eat, grow
        u = torch.rand(48, N * N, generator=g)
        ref = _step_reference(s, a, u, 40)
        s = eq.step(s, a, u, 40)
        for k, v in vars(ref).items():
            assert torch.equal(getattr(s, k), v), k
    assert int(s.eaten.sum()) > 0


# ---------------------------------------------------------------- what the network sees
def test_view_is_head_centred_and_deadly_neighbours_match_the_step():
    g = _gen(5)
    s = _random_games(128, seed=4)
    for _ in range(25):
        s = eq.step(s, solve.teacher_actions(s), torch.rand(128, N * N, generator=g), 1000)
    x = eq.planes(s)
    R = N - 1
    assert x.shape == (128, T.C, 2 * R + 1, 2 * R + 1)
    assert torch.equal(x[:, T.BODY, R, R], torch.ones(128))  # the head is always the centre
    # the walls: N × N inside cells, the rest of the window is wall
    assert torch.equal(x[:, T.WALL].sum((1, 2)), torch.full((128,), float((2 * R + 1) ** 2 - N * N)))
    assert torch.equal(x[:, T.FOOD].sum((1, 2)), torch.ones(128))
    for a, (i, j) in zip((T.LEFT, T.STRAIGHT, T.RIGHT), ((R, R - 1), (R - 1, R), (R, R + 1))):
        deadly = (x[:, T.WALL, i, j] > 0) | (x[:, T.AGE, i, j] * s.length > 1.5)
        died = s.alive & ~eq.step(s, torch.full((128,), a), torch.rand(128, N * N, generator=g), 1000).alive
        assert torch.equal(deadly[s.alive], died[s.alive])


# ---------------------------------------------------------------- the teacher
def _bfs(free, start):
    n = len(free)
    dist = [[-1] * n for _ in range(n)]
    dist[start[0]][start[1]] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and free[rr][cc] and dist[rr][cc] < 0:
                dist[rr][cc] = dist[r][c] + 1
                q.append((rr, cc))
    return dist


def test_distance_field_equals_breadth_first_search_and_teacher_is_safe():
    g = _gen(9)
    s = _random_games(64, seed=6)
    for _ in range(40):
        s = eq.step(s, solve.teacher_actions(s), torch.rand(64, N * N, generator=g), 1000)
    free = s.age <= 1
    start = torch.zeros_like(free)
    start[torch.arange(64), s.food[:, 0], s.food[:, 1]] = True
    dist = eq.distance_field(start, free)
    for b in range(64):
        ref = torch.tensor(_bfs(free[b].tolist(), s.food[b].tolist()))
        assert torch.equal(torch.where(dist[b] >= eq.FAR, -1, dist[b]), ref)
    a = solve.teacher_actions(s)
    safe = torch.stack([eq.step(s, torch.full((64,), k), torch.rand(64, N * N, generator=g), 1000).alive for k in
                        range(3)], 1)  # fmt: skip
    chosen_safe = safe.gather(1, a[:, None])[:, 0]
    assert (chosen_safe | ~safe.any(1) | ~s.alive).all()


# ---------------------------------------------------------------- the mixture-of-experts layer
def _moe(M=2, K=4, F=16, H=8, seed=0):
    return T.init_moe(M, K, F, H, _gen(seed)), torch.randn(M, 37, F, generator=_gen(seed + 1))


def test_fixed_one_hot_router_equals_each_games_own_expert():
    w, h = _moe()
    M, B, K = 2, 37, 4
    choice = torch.randint(K, (M, B), generator=_gen(2))
    G = torch.nn.functional.one_hot(choice, K).float()
    z = eq.moe_top1(h, G, G, w)  # dispatch → experts on their buffers → combine
    for m in range(M):
        for b in range(B):
            k = int(choice[m, b])
            one = {n: v[m : m + 1, k : k + 1] for n, v in w["experts"].items()}
            ref = eq.experts_on(h[m : m + 1, None, b : b + 1], one)[0, 0, 0]  # that expert alone on that game
            assert torch.allclose(z[m, b], ref, atol=1e-6, rtol=0)


def test_uniform_dense_router_gives_the_mean_of_the_experts_and_rows_sum_to_one():
    w, h = _moe()
    K = 4
    Y = eq.experts_on(h[:, None].expand(-1, K, -1, -1), w["experts"])  # every expert on every game (M, K, B, 3)
    z = eq.moe_dense(h, torch.full((2, 37, K), 1 / K), w)
    assert torch.allclose(z, Y.mean(1), atol=1e-6, rtol=0)
    p = eq.router_probs(h, w["router"])
    assert torch.allclose(p.sum(-1), torch.ones(2, 37), atol=1e-6) and (p >= 0).all()


def test_load_balance_loss_known_cases():
    K = 4
    p = torch.full((1, 8, K), 1 / K)
    G = torch.nn.functional.one_hot(torch.arange(8) % K, K).float()[None]
    assert torch.allclose(eq.load_balance_loss(p, G), torch.tensor([1.0]))  # balanced and uniform → 1
    Gc = torch.zeros(1, 8, K)
    Gc[..., 0] = 1
    pc = Gc.clone()
    assert torch.allclose(eq.load_balance_loss(pc, Gc), torch.tensor([float(K)]))  # collapsed → K
    # K = 2, 4 games: f = (3/4, 1/4), P = (0.6, 0.4) → 2 (0.75·0.6 + 0.25·0.4) = 1.1
    p2 = torch.tensor([[[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]])
    assert torch.allclose(eq.load_balance_loss(p2, eq.top1(p2)), torch.tensor([1.1]))


def test_monolith_has_the_same_number_of_weights_as_the_moe_controller():
    cfg = T.Config()
    w_moe = T.init_moe(1, cfg.experts, cfg.features, cfg.hidden, _gen())
    w_mono = T.init_mono(1, cfg.features, T.mono_hidden(cfg), _gen())
    a, b = T.n_params(w_moe), T.n_params(w_mono)
    assert abs(a - b) / a < 0.01


# ---------------------------------------------------------------- learning (slow)
@pytest.mark.slow
def test_trained_policy_beats_random_by_a_clear_margin():
    cfg = T.Config(grid=8, seeds=1, teacher_games=96, steps=500, eval_games=64, eval_steps=200, dagger=False)
    res = solve.run_all(cfg, torch.device("cpu"))
    rnd = float(res["food"]["random"].mean())
    best = max(float(res["food"][v].mean()) for v in ("dense", "switch", "dense_top1", "mono"))
    assert float(res["food"]["teacher"].mean()) > 3 * rnd + 2
    assert best > 3 * rnd + 1.0
