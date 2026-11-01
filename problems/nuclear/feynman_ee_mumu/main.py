"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import sigma_point_like, trace_tensor


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    title, m_f, sym = tensors.FINAL[params["final"]]
    gamma, eta = tensors.gamma_matrices(dev), tensors.metric(dev)  # (4, 4, 4), (4, 4)

    # Clifford algebra check: {γ^μ, γ^ν} = 2η^{μν}, so Tr(γ^μ γ^ν) = 4η^{μν}.
    one = torch.eye(4, dtype=gamma.dtype, device=dev)
    tr2 = trace_tensor(one, one, gamma).real
    print(f"Tr(γ^μ γ^ν) = 4η^μν: max error {(tr2 - 4 * eta.real).abs().max().item():.1e}")

    threshold = 2 * m_f
    sqrt_s_max = max(params["sqrt_s_max"], threshold * 1.5)
    theta = tensors.angles(91, dev)  # (B,)
    sqrt_s = tensors.energies(threshold, sqrt_s_max, 120, dev)  # (E,)
    dsdo, sigma = solve.cross_sections(sqrt_s, theta, m_f, tensors.ALPHA, gamma, eta)  # (E, B), (E,)
    exact = sigma_point_like(sqrt_s, m_f, tensors.ALPHA)
    afb = solve.forward_backward(dsdo, theta)

    print(f"{title}: threshold √s = 2m_{sym} = {threshold:.4f} GeV, sweep up to {sqrt_s_max:.1f} GeV")
    rel = ((sigma - exact) / exact).abs().max()
    print(f"σ from the traces vs 4πα²β(1+2m²/s)/3s: max relative difference {rel:.1e}")
    for j in (len(sqrt_s) // 6, len(sqrt_s) // 2, len(sqrt_s) - 1):
        shape = dsdo[j, 0] / dsdo[j, len(theta) // 2]
        print(
            f"√s = {sqrt_s[j]:7.3f} GeV   σ = {sigma[j]:9.3f} nb   "
            f"dσ/dΩ(0°)/dσ/dΩ(90°) = {shape:.3f}   A_FB = {afb[j]:+.1e}"
        )
    print("high-energy limit: dσ/dΩ ∝ 1 + cos²θ, so the 0°/90° ratio → 2; σ·s → 4πα²/3 = 86.8 nb·GeV²")
    return plot.animate(sqrt_s, theta, dsdo, sigma, exact, (title, sym, threshold), out_dir / "feynman_ee_mumu.gif")


if __name__ == "__main__":
    cli(run, __file__)
