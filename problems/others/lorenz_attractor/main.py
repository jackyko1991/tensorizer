"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    sigma, rho, beta = params["sigma"], params["rho"], params["beta"]
    A, Q = tensors.lorenz_tensors(sigma, rho, beta, dev)  # (3, 3), (3, 3, 3)

    center = solve.settle(A, Q)  # a point on the attractor
    radius = 10.0 ** params["log10_radius"]
    x0 = tensors.initial_ball(center, radius, params["n_traj"])  # (B, 3)
    times, traj = solve.solve(x0, A, Q, params["t_max"])  # (T,), (T, B, 3)
    s = equation.spread(traj)  # (T,)
    fit = solve.lyapunov_fit(times, s)

    print(f"A =\n{A.cpu().numpy()}")
    print(f"non-zero entries of Q: {[tuple(i) for i in Q.nonzero().tolist()]} -> {Q[Q != 0].tolist()}")
    print(
        f"divergence A_ii + (Q_iik + Q_iki) x_k = {equation.divergence(x0[:1], A, Q).item():.4f}"
        f"  (-(σ + 1 + β) = {-(sigma + 1 + beta):.4f})"
    )
    print(f"{traj.shape[1]} trajectories, spread {s[0]:.1e} -> {s[-1]:.2f}")
    print(f"fitted largest Lyapunov exponent λ = {fit[0]:.3f}  (literature: 0.906 for σ=10, ρ=28, β=8/3)")
    return plot.animate(times, traj, s, fit, radius, out_dir / "lorenz_attractor.gif")


if __name__ == "__main__":
    cli(run, __file__)
