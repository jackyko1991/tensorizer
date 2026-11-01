"""Main loop: parameters → tensors → train the PINN (and a plain network for comparison) → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors

M = 1.0  # mass (kg): only the ratios μ/m and k/m matter
T_MAX = 1.0  # time interval [0, T_MAX] in s
WIDTHS = [1, 32, 32, 32, 1]  # input t → 3 hidden tanh layers of 32 neurons → output x̂
SEED = 0


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    k, mu_true = params["k"], params["mu"]
    inverse = params["mode"] == "inverse"
    lam_phys = 10.0 ** params["log10_lambda"]

    t_d, x_d = tensors.observations(
        params["n_data"], params["data_fraction"] * T_MAX, params["noise"], M, mu_true, k, SEED, dev
    )  # (D,), (D,)
    t_c = tensors.collocation_points(params["n_colloc"], T_MAX, dev)  # (C,)
    t_plot = tensors.plot_grid(T_MAX, dev)  # (300,)
    x_exact = equation.exact_solution(t_plot, M, mu_true, k)

    # PINN: data + physics + initial condition. In inverse mode μ starts from a wrong guess and is learned.
    mu_start = params["mu_guess"] if inverse else mu_true
    pinn = solve.train(
        tensors.init_weights(WIDTHS, SEED, dev), mu_start, inverse, t_d, x_d, t_c, M, k,
        lam_phys, params["n_iter"], params["lr"], t_plot, label="PINN",
    )  # fmt: skip
    # The same network (same starting weights) without the physics term, for comparison.
    plain = None
    if params["compare"] == "on":
        plain = solve.train(
            tensors.init_weights(WIDTHS, SEED, dev), mu_true, False, t_d, x_d, t_c, M, k,
            0.0, params["n_iter"], params["lr"], t_plot, label="no physics",
        )  # fmt: skip

    extrap = (t_plot > t_d.max()).cpu()  # beyond the last measurement
    exact = x_exact.cpu()

    def rel_err(x: torch.Tensor, mask=slice(None)) -> float:
        return (torch.norm(x[mask] - exact[mask]) / torch.norm(exact[mask])).item()

    print(f"m = {M} kg, k = {k} N/m, true μ = {mu_true} kg/s   ·   {params['n_data']} data points on "
          f"[0, {t_d.max().item():.2f}] s, {params['n_colloc']} collocation points on [0, {T_MAX}] s")  # fmt: skip
    data, phys, init = pinn["losses"][-1].tolist()
    print(f"PINN final loss: data {data:.2e}   λ·physics {lam_phys * phys:.2e}   initial {init:.2e}")
    x_pinn = pinn["snapshots"][-1]
    print(f"PINN        rel. L2 error: whole interval {rel_err(x_pinn):6.2%}   "
          f"extrapolation {rel_err(x_pinn, extrap):6.2%}")  # fmt: skip
    if plain is not None:
        x_plain = plain["snapshots"][-1]
        print(f"no physics  rel. L2 error: whole interval {rel_err(x_plain):6.2%}   "
              f"extrapolation {rel_err(x_plain, extrap):6.2%}")  # fmt: skip
    if inverse:
        err = abs(pinn["mu_final"] - mu_true) / mu_true
        print(f"inferred μ = {pinn['mu_final']:.3f} kg/s (started at {mu_start}, true {mu_true}, error {err:.1%})")

    return plot.animate(
        t_plot, x_exact, t_d, x_d, t_c, pinn, plain, lam_phys, mu_true, inverse, out_dir / "pinn_oscillator.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
