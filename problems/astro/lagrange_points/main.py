"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    mu = params["mu"]
    points = solve.lagrange_points(tensors.lagrange_guesses(mu, dev), mu)
    residual = equation.grad_potential(points, mu).norm(dim=-1)  # (5,) |∇Φ_eff| after Newton
    phi_points = equation.effective_potential(points, mu)

    nudges = [params["nudge"] * f for f in (0.25, 0.6, 1.0)]  # degrees ahead of L4
    y0 = tensors.particles_near_l4(mu, nudges, dev)
    times, traj = solve.solve(y0, mu, params["t_max"], params["dt"])
    C = equation.jacobi_constant(traj, mu)  # (T, B)

    for k in range(5):
        x, y = points[k, 0].item(), points[k, 1].item()
        print(f"L{k + 1}: ({x:+.5f}, {y:+.5f})   |∇Φ_eff| = {residual[k]:.1e}   Φ_eff = {phi_points[k]:.5f}")
    star, planet = points.new_tensor([-mu, 0, 0]), points.new_tensor([1 - mu, 0, 0])
    d_star, d_planet = (points[3] - star).norm(), (points[3] - planet).norm()
    print(f"L4 distances to star and planet: {d_star:.6f}, {d_planet:.6f} (equilateral triangle: both 1)")
    print(f"Jacobi constant drift over the run: {(C - C[0]).abs().max():.1e}")

    grid = tensors.potential_grid(240, 1.6, dev)
    phi = equation.effective_potential(grid, mu)
    return plot.animate(times, traj, points, grid, phi, phi_points, mu, nudges, out_dir / "lagrange_points.gif")


if __name__ == "__main__":
    cli(run, __file__)
