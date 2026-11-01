"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    G, delta = tensors.G, 10.0 ** params["log10_delta"]
    m, x0, v0 = tensors.initial_state(delta, dev)  # (3,), (K=2, 3, 2), (K=2, 3, 2)

    times, xs, vs, n_steps = solve.solve(m, x0, v0, G, params["t_max"], 10.0 ** params["log10_tol"])
    dist = equation.phase_space_distance(xs, vs)  # (T,)
    fit = solve.lyapunov_fit(times, dist)
    eject = solve.ejection(xs[:, 0], vs[:, 0], m, G)  # reference run
    eject_twin = solve.ejection(xs[:, 1], vs[:, 1], m, G)

    e_err = solve.energy_error(xs, vs, m, G)[:, 0].max().item()
    com = equation.center_of_mass(xs[:, 0], m).norm(dim=-1).max().item()
    print(f"{n_steps} adaptive steps, t_max = {params['t_max']:g}, δ = {delta:.0e}")
    print(f"reference run: max relative energy error {e_err:.1e}, centre of mass moves at most {com:.1e}")
    print(f"phase-space distance: {dist[0]:.1e} -> {dist.max():.2f}; fitted Lyapunov exponent λ = {fit[0]:.3f}")
    for name, (step, body) in (("reference", eject), ("twin", eject_twin)):
        if step is None:
            print(f"{name}: no escape before t = {params['t_max']:g}")
        else:
            print(f"{name}: mass {m[body].item():g} escapes at t ≈ {times[step].item():.1f}")
    return plot.animate(times, xs, dist, m, fit, eject, out_dir / "three_body_chaos.gif")


if __name__ == "__main__":
    cli(run, __file__)
