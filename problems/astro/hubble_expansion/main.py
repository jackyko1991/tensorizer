"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    model = params["model"]
    H0 = tensors.hubble_constant(params["H0"])
    spacing = params["spacing"]
    q = tensors.galaxy_grid(params["n_grid"], spacing, dev)
    observer = tensors.nearest_galaxy(q, params["observer_x"] * spacing, params["observer_y"] * spacing)
    centre = tensors.nearest_galaxy(q, 0.0, 0.0)
    t_start = equation.time_of(params["a_start"], model, H0)
    t_end = equation.time_of(params["a_end"], model, H0)
    t = tensors.time_axis(t_start, t_end, 200, dev)

    sol = solve.solve(q, t, model, H0, observer)
    H_fit = solve.fitted_hubble_rate(sol["x_rel"], sol["v_rel"])

    print(f"{model}: today t0 = {equation.today(model, H0):.2f} Gyr, run from t = {t_start:.2f} to {t_end:.2f} Gyr")
    for k in (0, len(t) // 2, len(t) - 1):
        print(
            f"t = {t[k]:6.2f} Gyr  a = {sol['a'][k]:.3f}  H = ȧ/a = {sol['H'][k] * equation.KMS_PER_MPC_GYR:6.1f}"
            f"  fitted slope v/d from observer = {H_fit[k] * equation.KMS_PER_MPC_GYR:6.1f} km/s/Mpc"
        )
    print(f"photon emitted at a = {sol['a'][0]:.2f}, received at a = {sol['a'][-1]:.2f}: z = {sol['z'][-1]:.3f}")
    return plot.animate(sol, observer, centre, out_dir / "hubble_expansion.gif")


if __name__ == "__main__":
    cli(run, __file__)
