"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import score


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n = params["board_size"]
    adj = tensors.adjacency(n, dev)  # (P, P)
    board = tensors.empty_board(n, dev)  # (P,)
    max_moves = 6 if fast_mode() else n * n + n
    print(
        f"{n}×{n} Go, komi {params['komi']}, {params['simulations']} simulations per move, "
        f"{params['rollouts']} random playouts per leaf"
    )

    boards, moves, counts, values = solve.self_play(
        board, adj, params["simulations"], params["c_puct"], params["rollouts"], n * n, params["komi"], max_moves, 0
    )
    for i, (m, v) in enumerate(zip(moves.tolist(), values.tolist())):
        colour = "black" if i % 2 == 0 else "white"
        where = "pass" if m == n * n else f"({m // n + 1}, {m % n + 1})"
        if i < 6 or i == len(moves) - 1:
            print(f"move {i + 1:2d}: {colour:5s} plays {where:8s}  search thinks black wins with p = {(v + 1) / 2:.2f}")
    final = score(boards[-1][None], adj, params["komi"]).item()
    print(f"final position after {len(moves)} moves:\n{solve.show(boards[-1], n)}")
    print(f"area score black − white − komi = {final:+.1f}  →  {'black' if final > 0 else 'white'} wins")
    return plot.animate(boards, moves, counts, values, n, out_dir / "alphago_mini.gif")


if __name__ == "__main__":
    cli(run, __file__)
