"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import neighbours


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    k = tensors.kernel(dev)  # (3, 3)
    x0 = tensors.board(params["pattern"], params["size"], params["n_boards"], params["density"], 0, dev)  # (B, H, W)
    print(f"pattern {params['pattern']}: {x0.shape[0]} board(s) of {x0.shape[1]}×{x0.shape[2]} cells")
    print(f"neighbour counts at generation 0 range over {sorted(neighbours(x0, k).unique().int().tolist())}")

    history = solve.evolve(x0, k, params["generations"])  # (G+1, B, H, W)
    pop = solve.population(history)  # (G+1, B)
    for g in (0, len(pop) // 4, len(pop) // 2, len(pop) - 1):
        print(f"generation {g:4d}: population {pop[g].tolist()}")
    repeat = solve.first_repeat(history)
    if repeat is None:
        print("board 0 has not repeated yet: still evolving")
    else:
        print(f"board 0 repeats an earlier state at generation {repeat}: only still lifes and oscillators remain")
    return plot.animate(history, pop, params["pattern"], out_dir / "game_of_life.gif")


if __name__ == "__main__":
    cli(run, __file__)
