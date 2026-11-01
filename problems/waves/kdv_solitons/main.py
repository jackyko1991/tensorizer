"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    c1, c2, t_max = params["c_tall"], params["c_short"], params["t_max"]
    x, k, dx = tensors.grid(1024, dev)
    dealias = tensors.dealias_mask(k)
    u0 = tensors.two_solitons(x, c1, c2)

    times, u = solve.solve(u0, k, dealias, t_max)

    # "ghosts": the two solitons translated at their own speeds as if they did not interact (periodic box)
    ghosts = torch.stack(
        [
            equation.soliton(x, c1, (tensors.X_TALL + c1 * t) % tensors.BOX)
            + equation.soliton(x, c2, (tensors.X_SHORT + c2 * t) % tensors.BOX)
            for t in times.tolist()
        ]
    )

    M, P = equation.mass(u, dx), equation.momentum(u, dx)
    print(f"mass     M = {M[0]:.6f}, largest relative change {((M - M[0]).abs().max() / M[0]):.1e}")
    print(f"momentum P = {P[0]:.6f}, largest relative change {((P - P[0]).abs().max() / P[0]):.1e}")
    x1 = solve.peak_position(x, u[-1])
    x2 = solve.second_peak_position(x, u[-1], x1)
    box = tensors.BOX
    shift1 = (x1 - tensors.X_TALL - c1 * t_max + box / 2) % box - box / 2
    shift2 = (x2 - tensors.X_SHORT - c2 * t_max + box / 2) % box - box / 2
    d1, d2 = equation.phase_shifts(c1, c2)
    print(f"tall soliton shifted {shift1:+.3f}  (theory {d1:+.3f} once the collision is over)")
    print(f"short soliton shifted {shift2:+.3f}  (theory {-d2:+.3f})")
    return plot.animate(times, x, u, ghosts, out_dir / "kdv_solitons.gif")


if __name__ == "__main__":
    cli(run, __file__)
