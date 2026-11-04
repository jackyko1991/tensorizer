"""Main loop: (ε, K) grid → batched Boltzmann equations → η_B, compared with the observed 6.1 × 10⁻¹⁰."""

from pathlib import Path

import torch

from problems._common.runtime import cli, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    # CPU, float64: Boltzmann suppression e^{−z} spans tens of decades.
    fast = fast_mode()
    n_steps = 600 if fast else params["steps"]
    z = tensors.z_grid(params["z_max"], n_steps)
    n_eps, n_K = (9, 13) if fast else (params["n_eps"], params["n_K"])
    eps, K = tensors.parameter_grid((-8.0, -4.0), (-2.0, 3.0), n_eps, n_K)

    grid = solve.solve(z, eps, K, initial=params["initial"])
    eta = solve.baryon_to_photon(grid["n_bl_final"])  # (E, K)
    kappa = solve.efficiency(grid["n_bl_final"][0], eps[0, 0])  # (K,) the same for every ε (linear source)
    kappa_zero = solve.efficiency(solve.solve(z, eps[:1], K, initial="zero")["n_bl_final"][0], eps[0, 0])
    print(f"{n_eps}×{n_K} models (ε = 1e-8 … 1e-4, K = 0.01 … 1000), {n_steps} exponential-integrator steps in z = M/T")
    print(
        f"η_B ranges from {eta.min():.2e} to {eta.max():.2e}; observed η_B = {tensors.ETA_OBSERVED:.2e} (Planck, BBN)"
    )
    for k_val in (0.01, 1.0, 10.0, 100.0):
        j = int((K[0] - k_val).abs().argmin())
        need = solve.required_epsilon(kappa[j], tensors.ETA_OBSERVED)
        print(f"K = {K[0, j]:8.3g} (m̃1 = {equation.effective_mass(K[0, j].item()):.1e} eV): κ_f = {kappa[j]:.3e}"
              f" → observed η_B needs ε ≈ {need:.1e}")  # fmt: skip

    eps_pick = 10 ** params["log_eps"]
    K_pick = 10 ** params["log_K"]
    one = solve.solve(z, torch.tensor([[eps_pick]], dtype=torch.float64),
                      torch.tensor([[K_pick]], dtype=torch.float64), initial=params["initial"])  # fmt: skip
    eta_pick = solve.baryon_to_photon(one["n_bl_final"]).item()
    print(f"chosen model ε = {eps_pick:.2e}, K = {K_pick:.3g}: η_B = {eta_pick:.2e} "
          f"({eta_pick / tensors.ETA_OBSERVED:.2f} × observed)")  # fmt: skip
    print("checks: ε = 0 or N_N held at equilibrium give exactly zero asymmetry (see test.py)")
    data = {"eps": eps, "K": K, "eta": eta, "kappa": kappa}
    return plot.animate(data, one, kappa_zero, eps_pick, K_pick, tensors.ETA_OBSERVED, out_dir / "baryon_asymmetry.gif")


if __name__ == "__main__":
    cli(run, __file__)
