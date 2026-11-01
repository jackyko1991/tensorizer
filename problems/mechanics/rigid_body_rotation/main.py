"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    a, b, c = sorted([params["a"], params["b"], params["c"]], reverse=True)  # a ≥ b ≥ c
    inertia = tensors.box_inertia(1.0, a, b, c, dev)
    y0 = tensors.initial_state(params["spin_axis"], params["spin"], params["perturbation"], dev)

    sol = solve.solve(y0, inertia, params["t_max"])
    T = equation.kinetic_energy(sol["omega"], inertia)  # (T,)
    L_sq = equation.angular_momentum(sol["omega"], inertia).pow(2).sum(-1)  # (T,)

    print("principal moments I_11, I_22, I_33 =", [round(x, 4) for x in inertia.diagonal().tolist()])
    drift_T = ((T - T[0]) / T[0]).abs().max()
    drift_L = ((L_sq - L_sq[0]) / L_sq[0]).abs().max()
    print(f"relative drift: energy {drift_T:.1e}, |L|² {drift_L:.1e}")
    w = sol["omega"][:, tensors.AXES[params["spin_axis"]]]
    print(f"spin component: min {w.min():.2f}, max {w.max():.2f} rad/s (a sign change means the body flipped)")
    return plot.animate(sol, inertia, (a, b, c), out_dir / "rigid_body_rotation.gif")


if __name__ == "__main__":
    cli(run, __file__)
