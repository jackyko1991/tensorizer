"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    eps = tensors.epsilon(params["eps_abs"] * 1e-3, params["eps_phase"])
    delta = complex(params["re_delta"] * 1e-4, 0.0)
    dm = params["dm"] * 1e10 * tensors.TAU_S  # ħ/s → ħ/τ_S
    t = tensors.times(params["t_max"], 600, dev)  # (T,) in units of τ_S
    e, d = tensors.scenario_parameters(eps, delta, dev)  # (3,): chosen, δ = 0, ε = δ = 0

    out = solve.simulate(dm, tensors.GAMMA_S, tensors.GAMMA_L, e, d, t)
    h = out["h"]
    print(f"Δm τ_S = {dm:.4f},  Γ_L/Γ_S = {tensors.GAMMA_L:.3e},  ε = {abs(eps):.3e} ∠ {params['eps_phase']:.1f}°")
    print(f"H (K⁰, K̄⁰ basis, units ħ/τ_S, common mass removed):\n{h[0].cpu().numpy().round(5)}")
    print(f"CPT check: H₁₁ − H₂₂ = {complex((h[0, 0, 0] - h[0, 1, 1]).item()):.2e}  (0 when δ = 0)")
    gamma = equation.decay_matrix(h[0])
    print(f"unitarity: eigenvalues of Γ = i(H − H†) = {torch.linalg.eigvalsh(gamma).cpu().numpy().round(6)} (≥ 0)")
    v = equation.eigenvectors(e[0], d[0])
    overlap = torch.vdot(v[:, 0], v[:, 1])
    lhs = (1j * dm + (tensors.GAMMA_S + tensors.GAMMA_L) / 2) * overlap
    rhs = torch.vdot(v[:, 0], gamma @ v[:, 1])
    print(f"Bell–Steinberger: ⟨K_S|K_L⟩ = {complex(overlap.item()):.3e},  |LHS − RHS| = {abs(lhs - rhs).item():.1e}")
    a_t = out["a_t"][0, 1:].mean().item()
    print(f"Kabir T asymmetry A_T = {a_t:.4e}   (4 Re ε = {4 * eps.real:.4e};  CPLEAR 1998: (6.6 ± 1.6) × 10⁻³)")
    a_cpt = out["a_cpt"][0, -1].item()
    print(f"CPT asymmetry at t = {params['t_max']:g} τ_S: {a_cpt:.3e}   (4 Re δ = {4 * delta.real:.1e})")
    print(f"no CP violation (ε = δ = 0): max |A_T| = {out['a_t'][2, 1:].abs().max().item():.1e}")
    return plot.animate(t, out, (eps, delta), out_dir / "kaon_cp_violation.gif")


if __name__ == "__main__":
    cli(run, __file__)
