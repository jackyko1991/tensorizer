"""Main loop: fcc lattices → thermostatted equilibration → NVE production → g(r), pressure and energy drift."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import R_CUT, second_virial


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    rho, T, T_start = tensors.state_points(params["rho_liquid"], params["T_liquid"], dev)  # (S,) each
    n_cells = 4 if fast_mode() else params["n_cells"]  # n_cells ≥ 4 keeps L/2 > r_c in the solid
    x, L = tensors.fcc_lattice(n_cells, rho)  # (S, N, 3), (S,)
    S, N, _ = x.shape
    v = tensors.maxwell_velocities(x.shape, T_start, dev)
    dt = params["dt"]
    n_eq = 60 if fast_mode() else params["equilibration_steps"]
    n_prod = 60 if fast_mode() else params["production_steps"]
    print(f"{S} state points × N = {N} atoms; pair tensor d_saij {(S, 3, N, N)}; r_c = {R_CUT} σ")
    for s, name in enumerate(tensors.NAMES):
        print(f"  {name:6s} ρ* = {rho[s]:.3f}  T* = {T[s]:.2f}  box L = {L[s]:.2f} σ")

    r_max = min(0.5 * L.min().item(), 4.0)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # one thread per small step is steadier than many fighting over a busy CPU
    try:
        x, v = solve.equilibrate(x, v, L, T, T_start, dt, n_eq)
        res = solve.production(x, v, L, rho, dt, n_prod, r_max)
    finally:
        torch.set_num_threads(threads)
    print(f"equilibration {n_eq} + NVE production {n_prod} steps, dt = {dt} τ ({(n_eq + n_prod) * dt:.0f} τ in all)")

    E, Tk, P = res["E"], res["T"], res["P"]
    peak = solve.first_peak(res["r"], res["g"])
    tail = res["r"] > 0.8 * r_max
    print("  state    ⟨T*⟩   ⟨P*⟩     energy drift / atom   g(r) peak at   ⟨g⟩ for r > {:.1f} σ".format(0.8 * r_max))
    for s, name in enumerate(tensors.NAMES):
        drift = (E[-1, s] - E[0, s]).item()
        print(
            f"  {name:6s}  {Tk[:, s].mean():.3f}  {P[:, s].mean():+.3f}   {drift:+.2e} ε"
            f"           {peak[s]:.3f} σ        {res['g'][s, tail].mean():.3f}"
        )
    print(f"LJ minimum 2^(1/6) σ = {2 ** (1 / 6):.3f} σ")

    # low density: compare with the ideal gas and the second-virial correction Z = 1 + B₂ρ
    T_gas, rho_gas = Tk[:, 0].mean().item(), rho[0].item()
    Z = P[:, 0].mean().item() / (rho_gas * T_gas)
    B2 = second_virial(T_gas)
    print(
        f"gas: P/(ρT) = {Z:.3f}   ideal gas 1   with B₂(T*={T_gas:.2f}) = {B2:.3f} σ³: 1 + B₂ρ = {1 + B2 * rho_gas:.3f}"
    )
    return plot.animate(res, L, rho, T, dt, out_dir / "lj_liquid_rdf.gif")


if __name__ == "__main__":
    cli(run, __file__)
