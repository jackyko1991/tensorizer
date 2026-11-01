"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import numpy as np

from problems._common.runtime import cli

from . import plot, solve, tensors
from .equation import field_tensor, lower_second, minkowski_norm


def run(params: dict, out_dir: Path) -> Path:
    config, strength, t_max = params["config"], params["strength"], params["t_max"]
    eta = tensors.metric()  # (4, 4)   tiny tensors: CPU float64
    E, B = tensors.fields(config, strength)  # (3,), (3,)
    F_mixed = lower_second(field_tensor(E, B, tensors.levi_civita()), eta)  # F^μ_ν (4, 4)
    qm, y0 = tensors.particles(config, params["n_particles"], params["v_max"])  # (P,), (P, 2, 4)

    tau, traj = solve.solve(y0, qm, F_mixed, t_max)  # (T,), (T, P, 2, 4)
    t = np.linspace(0.0, t_max, 300)
    lab = solve.to_lab_time(traj, t)
    newton = solve.newtonian(config, qm, y0, strength, t)

    U = traj[:, :, 1]  # (T, P, 4)
    drift = ((minkowski_norm(U, eta) - 1) / U[..., 0] ** 2).abs().max()  # relative to γ², the size of each term
    print(f"{len(tau)} proper-time steps, largest relative drift of U^mu U_mu from 1: {drift:.1e}")
    if config == "cyclotron":
        v = (y0[:, 1, 2] / y0[:, 1, 0]).tolist()
        labels = [f"v = {s:.2f}, γ = {g:.2f}" for s, g in zip(v, y0[:, 1, 0].tolist())]
    else:
        labels = [f"qE/m = {a * strength:g}" for a in qm.tolist()]
    for label, g in zip(labels, lab["gamma"][-1]):
        print(f"{label:28s} gamma at t_max = {g:.3f}")
    return plot.animate(lab, newton, config, labels, strength, out_dir / "relativistic_particle.gif")


if __name__ == "__main__":
    cli(run, __file__)
