"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import lame, lame_stress, stiffness


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    E, nu = params["E"] * 1e9, params["poisson"]
    lam, mu = lame(E, nu)
    delta = tensors.kronecker(dev)  # (3, 3)
    C = stiffness(lam, mu, delta)  # (3, 3, 3, 3)
    x = tensors.block_points(9, dev)  # (9, 9, 3, 3)
    sigma_applied = tensors.loading(60, params["sigma_max"] * 1e6, params["tau_max"] * 1e6, dev)  # (T, 3, 3)

    result = solve.load_path(sigma_applied, C, x)

    eps, sigma = result["eps"][-1], result["sigma"][-1]
    print(f"λ = {lam / 1e9:.1f} GPa, μ = {mu / 1e9:.1f} GPa")
    print(f"C_ijkl symmetric: minor {bool((C - C.transpose(0, 1)).abs().max() < 1e-3)}, "
          f"major {bool((C - C.permute(2, 3, 0, 1)).abs().max() < 1e-3)}")  # fmt: skip
    mid = len(sigma_applied) // 2
    e_mid = result["eps"][mid]
    if e_mid[0, 0] != 0:
        print(f"uniaxial step: E = σ_xx/ε_xx = {sigma_applied[mid, 0, 0] / e_mid[0, 0] / 1e9:.1f} GPa, "
              f"ν = -ε_yy/ε_xx = {-e_mid[1, 1] / e_mid[0, 0]:.3f}")  # fmt: skip
    print(f"final strain (×1e-6): ε_xx {eps[0, 0] * 1e6:.0f}, ε_yy {eps[1, 1] * 1e6:.0f}, "
          f"ε_zz {eps[2, 2] * 1e6:.0f}, ε_xy {eps[0, 1] * 1e6:.0f}")  # fmt: skip
    print(
        "einsum C:ε vs λ tr(ε) δ + 2μ ε: max difference "
        f"{(sigma - lame_stress(eps, lam, mu, delta)).abs().max():.2e} Pa"
    )
    s1, s2 = result["in_plane"][-1].flip(0) / 1e6
    print(f"principal stresses: σ1 = {s1:.1f} MPa, σ2 = {s2:.1f} MPa, max shear {(s1 - s2) / 2:.1f} MPa")
    return plot.animate(x, result, out_dir / "stress_strain.gif")


if __name__ == "__main__":
    cli(run, __file__)
