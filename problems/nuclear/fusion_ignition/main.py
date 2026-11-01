"""Main loop: grid of hot spots → power balance → batched RK4 burn → ignition map → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, fast_mode

from . import plot, solve, tensors
from .equation import confinement_time, ideal_ignition_temperature, number_density, sigma_v


def run(params: dict, out_dir: Path) -> Path:
    # CPU, float64: ⟨σv⟩ spans many decades and the burn is stiff.
    fast = fast_mode()
    rho, loss = params["density"], params["loss"]
    n = 10 if fast else params["n_grid"]
    t0, r0 = tensors.initial_grid(n, n, (params["t_min"], params["t_max_init"]), (0.05, 1.5))
    y0 = tensors.initial_state(t0, r0)  # (B, 3)

    for T in (5.0, 10.0, 20.0, 64.0):
        sv = sigma_v(torch.tensor(T, dtype=torch.float64)).item()
        print(f"⟨σv⟩_DT({T:4.0f} keV) = {sv:.3e} m³/s")
    print(f"ideal ignition temperature (alpha heating = bremsstrahlung): {ideal_ignition_temperature():.2f} keV")

    # Lawson-type boundary: where self-heating balances bremsstrahlung + PdV losses at t = 0
    t_curve = torch.logspace(0, 2, 300, dtype=torch.float64)
    r_curve = solve.balance_boundary(t_curve, rho, loss)
    ok = torch.isfinite(r_curve)
    k = int(torch.argmin(torch.where(ok, r_curve * t_curve, torch.full_like(t_curve, 1e9))))
    tau = confinement_time(t_curve[k], r_curve[k], rho).item()
    ntt = number_density(rho) * t_curve[k].item() * tau
    print(
        f"heating = losses curve (ρ = {rho:.0f} g/cm³, loss ×{loss}): min ρR·T = {r_curve[k] * t_curve[k]:.2f} "
        f"g cm⁻² keV at T = {t_curve[k]:.1f} keV, ρR = {r_curve[k]:.2f} g/cm²  →  n T τ = {ntt:.2e} keV s/m³ "
        f"(τ = R/3c_s = {tau * 1e12:.0f} ps)"
    )

    times, states = solve.integrate(
        y0, rho, loss, params["rho_r_max"], 40.0 if fast else params["t_end"], 0.25, 30 if fast else 120
    )
    ign = solve.ignited(states)
    print(f"\n{ign.sum().item()} of {len(ign)} hot spots ignite within {times[-1]:.0f} ps")
    first = solve.min_ignition_rho_r(t0, r0, ign)
    for i in range(0, len(t0), max(1, len(t0) // 6)):
        val = f"ρR₀ ≥ {first[i]:.2f} g/cm²" if torch.isfinite(first[i]) else "never (in this ρR range)"
        print(f"T₀ = {t0[i]:5.1f} keV: ignites for {val}")
    if ign.any():
        print(
            f"ignited hot spots: peak T up to {states[:, ign, 0].exp().max():.0f} keV, "
            f"mean final burn-up {100 * states[-1, ign, 2].mean():.0f} %"
        )

    rows = [int((t0 - target).abs().argmin()) for target in (6.0, 10.0)]
    cols = torch.linspace(0, len(r0) - 1, 5).round().long().tolist()  # five ρR₀ from smallest to largest
    shown = [i * len(r0) + j for i in rows for j in cols]
    return plot.animate(times, states, t0, r0, ign, t_curve, r_curve, shown, out_dir / "fusion_ignition.gif")


if __name__ == "__main__":
    cli(run, __file__)
