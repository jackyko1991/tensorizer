r"""Rules and search → tensor translation.

The board is a graph: points p, q are neighbours when A_{pq} = 1. Growing a set of points r by one step is

    (A r)_p = A_{pq} r_q > 0       (q summed),

and every Go rule here is a "flood fill" built from it: starting from a seed set, repeatedly add points of an
allowed set that touch what has been reached,

    r \leftarrow r \lor (s \land A r)      until nothing changes.

* A group is alive if it has a liberty: seed = empty points, allowed = stones of that colour. Unreached stones
  are captured (opponent first, then the mover's own stones: Tromp–Taylor rules).
* Area score: seed = black stones, allowed = empty points gives the territory black reaches; the same for
  white; an empty point counts for a colour only if the other colour cannot reach it.

AlphaGo's search (Monte Carlo tree search with PUCT) picks at each node the move

    a^* = \arg\max_a \Big[ Q_a + c\,P_a\,\frac{\sqrt{\sum_b N_b}}{1 + N_a} \Big],

where N_a counts visits, Q_a is the mean value of the move, and P_a is a prior. After all simulations it
plays in proportion to the visit counts, π_a = N_a / Σ_b N_b.
"""

import torch


def grow(r: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    r"""(A r)_p > 0: points touching the set r.   r: (B, P) bool → (B, P) bool"""
    return torch.einsum("pq,bq->bp", adj, r.float()) > 0


def reach(seed: torch.Tensor, allowed: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    r"""Flood fill r ← r ∨ (s ∧ A r) from seed through allowed points.   (B, P) bool → (B, P) bool"""
    r = seed
    while True:
        nxt = r | (allowed & grow(r, adj))
        if torch.equal(nxt, r):
            return r
        r = nxt


def play(board: torch.Tensor, player: torch.Tensor, move: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    r"""Place a stone (move = P means pass), then capture groups without liberties.
    board: (B, P) in {−1, 0, 1}, player: (B,), move: (B,) → (B, P)"""
    b = board.clone()
    n_points = b.shape[1]
    stone = move < n_points
    b[stone, move[stone]] = player[stone]
    for colour in (-player, player):  # opponent's groups first, then suicide of one's own
        mine = b == colour[:, None]
        alive = reach(b == 0, mine, adj)
        b[mine & ~alive] = 0
    return b


def score(board: torch.Tensor, adj: torch.Tensor, komi: float) -> torch.Tensor:
    r"""Area score for black: stones + empty points only black reaches, minus the same for white, minus komi.
    (B, P) → (B,)"""
    empty = board == 0
    black_reach = reach(board > 0, empty, adj)
    white_reach = reach(board < 0, empty, adj)
    black = (board > 0) | (empty & black_reach & ~white_reach)
    white = (board < 0) | (empty & white_reach & ~black_reach)
    return black.sum(-1).float() - white.sum(-1).float() - komi


def puct(q: torch.Tensor, prior: torch.Tensor, visits: torch.Tensor, c: float) -> torch.Tensor:
    r"""U_a = Q_a + c P_a √(Σ_b N_b) / (1 + N_a): the score PUCT maximises.   (M,) → (M,)"""
    return q + c * prior * visits.sum().sqrt() / (1 + visits)
