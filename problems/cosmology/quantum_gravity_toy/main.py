"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    scales = params["gap_scale"] * torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64, device=dev)
    lam = tensors.LAMBDA * scales  # (B,) three loop sizes: ρ_c × 1, 4, 16
    p = torch.full_like(lam, params["p_phi"])
    rho_c = equation.critical_density(lam)
    y0 = tensors.initial_state(p, lam, 10 ** params["log10_rho_start"] * float(rho_c[0]))
    t_bounce = equation.time_to_bounce(y0, p, lam)
    sol = solve.evolve(y0, p, lam, params["dt"], 2 * float(t_bounce.max()))  # y: (T, B, 3)

    y = sol["y"]
    rho, H = equation.density(y, p), equation.hubble(y, lam)
    C = equation.constraint(y, p, lam) / equation.matter_energy(y, p)
    print(
        f"Planck units. γ = {tensors.GAMMA}, area gap λ² = {tensors.LAMBDA**2:.3f} ℓ_P², p_φ = {params['p_phi']:.0f} ħ"
    )
    for j in range(len(lam)):
        V_min = float(y[:, j, 0].min())
        V_B = float(p[j] / torch.sqrt(2 * rho_c[j]))
        ratio = float(rho[:, j].max() / rho_c[j])
        print(f"λ × {float(scales[j]):5.3f}: ρ_c = {float(rho_c[j]):8.3f} ρ_P   max sampled ρ/ρ_c = {ratio:.6f}"
              f"   V_min = {V_min:9.2f} (V_B = {V_B:9.2f})")  # fmt: skip
    print(f"max |C| / (p_φ²/2V) along all trajectories: {float(C.abs().max()):.1e} (constraint preserved)")
    t_sing = equation.classical_singularity_time(y0[0, 0], p[0])
    print(f"classical GR from the same start hits V = 0 after t = {t_sing:.3f} t_P; "
          f"LQC bounces at t = {float(t_bounce[0]):.3f}")  # fmt: skip

    # Wheeler–DeWitt packet in the (φ, ln V) plane, aligned with the bounce of the first universe
    phi = y[:, 0, 2]
    phi_b = float(phi[y[:, 0, 0].argmin()])
    x_b = math.log(float(p[0] / torch.sqrt(2 * rho_c[0])))
    x, phi_grid = tensors.wdw_grid(x_b - 2, float(torch.log(y[:, 0, 0]).max()) + 0.5, float(phi[0]), float(phi[-1]),
                                   260, 180, dev)  # fmt: skip
    psi = equation.wdw_packet(x, phi_grid - phi_b, float(p[0]), x_b, spread=params["spread"])  # (P, X)
    print("Wheeler–DeWitt packet: follows the classical line ln V = ln V_B − ln 2 − √(12πG)(φ − φ_B) into V → 0")
    return plot.animate(sol, p, lam, scales, rho_c, H, rho, (x, phi_grid, psi, x_b, phi_b),
                        out_dir / "quantum_gravity_toy.gif")  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
