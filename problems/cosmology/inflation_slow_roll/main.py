"""Main loop: five potentials → one batched background integration in e-folds → n_s and r → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors

N_START = 68.0  # e-folds before the end at which every field starts (slow-roll estimate)


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    model = tensors.model(tensors.KINDS, f=params["f"], mu=params["mu"], device=dev)
    phi0 = tensors.start_field(model, N_START)
    sol = solve.integrate(phi0, equation.slow_roll_velocity(phi0, model), model, dN=0.02, N_max=N_START + 25)

    N_star = params["N_star"]
    N_grid = torch.linspace(50.0, 60.0, 41, dtype=torch.float64, device=dev)  # the usual pivot window
    track = solve.predictions(sol, model, N_grid)  # each (41, B)
    pred = solve.predictions(sol, model, torch.tensor([N_star], dtype=torch.float64, device=dev))
    pred = {k: v[0] for k, v in pred.items()}  # (B,)
    ok = tensors.allowed(pred["ns"], pred["r"], sigmas=2)
    scale = solve.energy_scale(pred["eps_V"], tensors.A_S)

    print(f"Planck 2018: n_s = {tensors.NS_MEAN} ± {tensors.NS_SIGMA};  BICEP/Keck 2021: r < {tensors.R_MAX} (95 %)")
    print(
        f"pivot N_* = {N_star:.0f} e-folds before the end;"
        f" natural f = {params['f']}, hilltop μ = {params['mu']} (M_pl = 1)"
    )
    print(f"{'potential':16s} {'N_total':>7s} {'φ_*':>6s} {'ε_V':>8s} {'η_V':>8s} {'n_s(SR)':>8s} {'n_s(num)':>8s}"
          f" {'r(SR)':>7s} {'r(num)':>7s} {'V_*^¼ (GeV)':>11s}  verdict")  # fmt: skip
    for b, kind in enumerate(tensors.KINDS):
        print(
            f"{tensors.LABELS[kind]:16s} {sol['N_end'][b]:7.1f} {pred['phi'][b]:6.2f} {pred['eps_V'][b]:8.5f}"
            f" {pred['eta_V'][b]:+8.5f} {pred['ns_V'][b]:8.4f} {pred['ns'][b]:8.4f} {pred['r_V'][b]:7.4f}"
            f" {pred['r'][b]:7.4f} {scale[b]:11.2e}  {'inside' if ok[b] else 'disfavoured'}"
        )
    print(
        "SR = potential slow-roll (first order);"
        " num = Hubble-flow parameters of the numerical background (second order)"
    )

    view = {
        "N": sol["N"], "phi": sol["phi"], "N_end": sol["N_end"], "phi0": phi0,
        "N_grid": N_grid, "ns_track": track["ns"], "r_track": track["r"],
        "ns": pred["ns"], "r": pred["r"], "ns_V": pred["ns_V"], "r_V": pred["r_V"], "ok": ok,
    }  # fmt: skip
    view = {k: v.cpu().numpy() for k, v in view.items()}
    view["V"] = lambda phi: equation.shapes(
        torch.as_tensor(phi, dtype=torch.float64), params["f"], params["mu"]
    ).numpy()
    return plot.animate(view, N_star, N_START, out_dir / "inflation_slow_roll.gif")


if __name__ == "__main__":
    cli(run, __file__)
