"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    T, mu1, mu2 = tensors.TENSION, tensors.MU1, params["mu_ratio"] * tensors.MU1
    c1, c2 = tensors.speed(mu1), tensors.speed(mu2)
    x, dx = tensors.grid(params["n_points"], dev)
    mu = tensors.density(x, mu1, mu2)
    dt = 0.9 * dx / max(c1, c2)  # Courant condition on the faster string
    u_prev, u0 = tensors.right_moving_pulse(x, params["width"], c1, dt)

    t_meas = solve.measuring_time(c1, c2, tensors.X0)
    times, traj = solve.solve(u_prev, u0, mu, T, dx, dt, max(params["t_max"], t_meas))

    Z1, Z2 = equation.impedance(T, mu1), equation.impedance(T, mu2)
    R, Tr = equation.reflection_transmission(Z1, Z2)
    R_num, T_num = solve.measure(x, traj[(times - t_meas).abs().argmin()])
    print(f"Z1 = {Z1:.3f}, Z2 = {Z2:.3f},  c1 = {c1:.3f} m/s, c2 = {c2:.3f} m/s")
    print(f"reflected   amplitude {R_num:+.4f}   theory R = (Z1-Z2)/(Z1+Z2) = {R:+.4f}")
    print(f"transmitted amplitude {T_num:+.4f}   theory T = 2Z1/(Z1+Z2)     = {Tr:+.4f}")
    print(f"energy flux balance R² + (Z2/Z1) T² = {R_num**2 + Z2 / Z1 * T_num**2:.4f}  (should be 1)")
    E = equation.energy(traj, mu, T, dx, dt)
    print(f"largest relative energy change: {((E - E[0]).abs().max() / E[0]).item():.2e}")

    shown = times <= params["t_max"] + 1e-9
    shown[0] = False  # skip the t = −Δt row
    info = {"R": R, "T": Tr, "Z1": Z1, "Z2": Z2, "mu2": mu2}
    return plot.animate(times[shown], x, traj[shown], info, out_dir / "string_junction.gif")


if __name__ == "__main__":
    cli(run, __file__)
