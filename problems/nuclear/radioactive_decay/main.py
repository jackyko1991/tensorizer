"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import count, decay_constant, exact_count


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    lam = decay_constant(params["half_life"])
    alive0 = tensors.nuclei(params["n_side"], dev)
    times = tensors.time_axis(params["t_max"], params["dt"], dev)

    history = solve.simulate(alive0, lam, params["dt"], len(times) - 1)  # (T, S, S)
    counts = count(history)  # (T,)
    exact = exact_count(alive0.numel(), lam, times)  # (T,)

    print(f"λ = {lam:.4f} 1/s   half-life = {params['half_life']:.2f} s   N0 = {alive0.numel()}")
    for k in range(0, len(times), max(1, len(times) // 8)):
        print(
            f"t = {times[k].item():6.2f} s   simulated N = {counts[k].item():5d}   N0 e^(-λt) = {exact[k].item():8.1f}"
        )
    return plot.animate(times, history, counts, exact, params["half_life"], out_dir / "radioactive_decay.gif")


if __name__ == "__main__":
    cli(run, __file__)
