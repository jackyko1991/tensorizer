"""AlphaGo-style search: a tree of positions explored with PUCT, leaves judged by a batch of random playouts."""

import math
from dataclasses import dataclass, field

import torch

from .equation import play, puct, score


def legal_moves(board: torch.Tensor, player: float, previous: torch.Tensor | None, adj: torch.Tensor):
    """Try all P points + pass at once (one batched play): legal if the point was empty, the new stone survives
    (no suicide) and the position does not repeat the one before the opponent's move (simple ko).
    Returns a mask (P+1,) and the resulting boards (P+1, P)."""
    n_points = board.shape[0]
    moves = torch.arange(n_points + 1, device=board.device)
    after = play(board.expand(n_points + 1, -1), torch.full((n_points + 1,), player, device=board.device), moves, adj)
    legal = torch.ones(n_points + 1, dtype=torch.bool, device=board.device)
    legal[:n_points] = (board == 0) & (after[torch.arange(n_points), torch.arange(n_points)] == player)
    if previous is not None:
        legal[:n_points] &= ~(after[:n_points] == previous).all(-1)
    return legal, after


def playouts(board, player: float, adj, n: int, depth: int, komi: float, gen: torch.Generator) -> float:
    """Play n random games at once from this position for `depth` moves; random moves never fill one's own eye.
    Returns the mean result for black: +1 black wins, −1 white wins."""
    b = board.expand(n, -1).clone()
    to_play = torch.full((n,), player, device=board.device)
    degree = adj.sum(1)  # neighbours of every point (2 at corners, 3 on edges, 4 inside)
    for _ in range(depth):
        own = torch.einsum("pq,bq->bp", adj, (b == to_play[:, None]).float())
        eye = (b == 0) & (own == degree)  # an empty point surrounded by one's own stones
        weights = torch.cat([((b == 0) & ~eye).float(), torch.full((n, 1), 1e-3, device=b.device)], 1)
        move = torch.multinomial(weights, 1, generator=gen)[:, 0]
        b = play(b, to_play, move, adj)
        to_play = -to_play
    return torch.sign(score(b, adj, komi)).mean().item()


@dataclass
class Node:
    board: torch.Tensor
    to_play: float
    previous: torch.Tensor | None  # board before the last move, for the ko rule
    passes: int = 0  # consecutive passes leading here; 2 ends the game
    moves: torch.Tensor | None = None  # legal move indices, filled on expansion
    children: list = field(default_factory=list)
    prior: torch.Tensor | None = None
    visits: torch.Tensor | None = None
    value_sum: torch.Tensor | None = None  # summed values for black


def expand(node: Node, adj) -> None:
    legal, after = legal_moves(node.board, node.to_play, node.previous, adj)
    node.moves = legal.nonzero()[:, 0]
    n_points = node.board.shape[0]
    p = torch.ones(len(node.moves), device=node.board.device)
    p[node.moves == n_points] = 0.1  # uniform prior; passing is discouraged (AlphaGo uses a policy network here)
    node.prior = p / p.sum()
    node.visits = torch.zeros(len(node.moves), device=node.board.device)
    node.value_sum = torch.zeros(len(node.moves), device=node.board.device)
    node.children = [
        Node(after[m], -node.to_play, node.board, node.passes + 1 if m == n_points else 0) for m in node.moves.tolist()
    ]


def search(root: Node, adj, sims: int, c: float, rollouts: int, depth: int, komi: float, gen) -> torch.Tensor:
    """Run `sims` simulations from root; return its visit counts N_a for all P+1 moves."""
    if root.moves is None:
        expand(root, adj)
    for _ in range(sims):
        node, path = root, []
        while node.moves is not None and node.passes < 2:  # selection: follow the PUCT maximum
            # mean value of each move from the mover's side; untried moves start at the node's average
            parent_mean = node.value_sum.sum() / node.visits.sum().clamp(min=1)
            q = torch.where(node.visits > 0, node.value_sum / node.visits.clamp(min=1), parent_mean) * node.to_play
            k = int(torch.argmax(puct(q, node.prior, node.visits, c)))
            path.append((node, k))
            node = node.children[k]
        if node.passes >= 2:  # both passed: the game is over, count the score exactly
            value = math.copysign(1.0, score(node.board[None], adj, komi).item())
        else:  # expansion + evaluation by random playouts
            expand(node, adj)
            value = playouts(node.board, node.to_play, adj, rollouts, depth, komi, gen)
        for parent, k in path:  # backup
            parent.visits[k] += 1
            parent.value_sum[k] += value
    counts = torch.zeros(root.board.shape[0] + 1, device=root.board.device)
    counts[root.moves] = root.visits
    return counts


def self_play(board, adj, sims: int, c: float, rollouts: int, depth: int, komi: float, max_moves: int, seed: int):
    """One game of the search against itself. Returns boards (M+1, P), moves (M,), visit counts (M, P+1) and
    the root value estimates for black (M,)."""
    gen = torch.Generator().manual_seed(seed)
    node = Node(board, 1.0, None)
    boards, moves, counts, values = [board], [], [], []
    while node.passes < 2 and len(moves) < max_moves:
        n = search(node, adj, sims, c, rollouts, depth, komi, gen)
        k = int(torch.argmax(node.visits))  # play the most visited move
        values.append((node.value_sum.sum() / node.visits.sum()).item())
        moves.append(int(node.moves[k]))
        counts.append(n)
        node = node.children[k]  # reuse the subtree for the next move
        boards.append(node.board)
    return torch.stack(boards), torch.tensor(moves), torch.stack(counts), torch.tensor(values)


def show(board: torch.Tensor, n: int) -> str:
    """Text picture of the board: X black, O white, . empty."""
    chars = {1.0: "X", -1.0: "O", 0.0: "."}
    return "\n".join(" ".join(chars[v] for v in board.view(n, n)[r].tolist()) for r in range(n))
