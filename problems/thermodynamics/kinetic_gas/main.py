"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n, radius = params["n_particles"], params["radius"]
    x0 = tensors.initial_positions(n, radius, dev)  # (N, 2)
    v0 = tensors.initial_velocities(n, params["speed"], dev)  # (N, 2)

    times, xs, vs, pressure = solve.solve(x0, v0, tensors.BOX, radius, params["t_max"], tensors.MASS)

    energy = equation.kinetic_energy(vs, tensors.MASS)  # (F,)
    kT = equation.temperature(v0, tensors.MASS).item()  # same at every step: E is conserved
    area = (tensors.BOX - 2 * radius) ** 2  # region the disk centres can reach
    ideal = n * kT / area
    print(f"kinetic energy {energy[0]:.4f} -> {energy[-1]:.4f}   (kT = {kT:.3f})")
    print(f"pressure from the walls {pressure:.2f}   ideal gas N kT / A = {ideal:.2f}   ratio {pressure / ideal:.3f}")

    title = f"kT = {kT:.2f}    P = {pressure:.0f}  vs  NkT/A = {ideal:.0f}"
    return plot.animate(times, xs, vs, kT, tensors.BOX, radius, title, out_dir / "kinetic_gas.gif")


if __name__ == "__main__":
    cli(run, __file__)
