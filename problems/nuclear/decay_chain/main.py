"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import activity, bateman_matrix


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    lam = tensors.decay_constants([params["half_life_parent"], params["half_life_daughter"]], dev)  # (3,)
    a = bateman_matrix(lam)  # (3, 3)
    n0 = tensors.initial_populations(1000.0, len(lam), dev)
    t = tensors.time_axis(params["t_max"], 300, dev)

    n = solve.exact(a, n0, t)  # (T, 3)
    t_rk4, n_rk4 = solve.rk4(a, n0, params["t_max"], params["rk4_steps"])
    act = activity(n, lam)  # (T, 3)

    l1, l2 = lam[0].item(), lam[1].item()
    print(f"λ1 = {l1:.4f} 1/h   λ2 = {l2:.4f} 1/h")
    print(f"max |N_exact − N_RK4| at t_max: {(n_rk4[-1] - solve.exact(a, n0, t_rk4[-1:])[0]).abs().max().item():.2e}")
    print(f"total N at t_max: {n[-1].sum().item():.6f} (started with {n0.sum().item():.0f})")
    print(f"activity ratio a2/a1 at t_max: {act[-1, 1].item() / act[-1, 0].item():.4f}")
    if l2 > l1:
        print(f"equilibrium prediction λ2/(λ2 − λ1) = {l2 / (l2 - l1):.4f}")
    else:
        print("daughter lives longer than parent: no equilibrium, the ratio keeps growing")
    names = ["parent", "daughter", "stable end"]
    return plot.animate(t, n, t_rk4, n_rk4, act, names, out_dir / "decay_chain.gif")


if __name__ == "__main__":
    cli(run, __file__)
