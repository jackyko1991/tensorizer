"""Solvers: the potential crossover for every n, Yukawa fits, the ADD table against the limits, the RS warp scan."""

import math

import torch

from . import equation, tensors


def crossover(ns: list[int], m_max: int, r: torch.Tensor) -> dict:
    """phi(r) r / (G_4 M) for each n on a log grid r (R,): images inside the compact space, KK modes outside."""
    counts = {n: tensors.shell_counts(n, m_max, r.device) for n in ns}
    inner, outer = r < 1.0, r >= 1.0
    ratio, slope, fits = [], [], []
    for n in ns:  # 6 values of n; each line is fully batched over r
        phi = torch.cat(
            [equation.potential_images(r[inner], n, counts[n]), equation.potential_kk(r[outer], n, counts[n])]
        )
        ratio.append(-phi * r / equation.g4(n))
        f_in = lambda x, n=n: equation.potential_images(x, n, counts[n])  # noqa: E731
        f_out = lambda x, n=n: equation.potential_kk(x, n, counts[n])  # noqa: E731
        slope.append(torch.cat([equation.log_slope(f_in, r[inner]), equation.log_slope(f_out, r[outer])]))
        r_fit = torch.linspace(16.0, 24.0, 25, dtype=r.dtype, device=r.device)
        fits.append(equation.fit_yukawa(r_fit, equation.kk_deviation(r_fit, n, counts[n])))
    agree = max(
        (equation.potential_images(r[(r > 0.6) & (r < 3)], n, counts[n])
         / equation.potential_kk(r[(r > 0.6) & (r < 3)], n, counts[n]) - 1).abs().max().item()
        for n in ns
    )  # fmt: skip
    return {"ratio": torch.stack(ratio), "slope": torch.stack(slope), "fits": fits, "poisson_gap": agree,
            "counts": counts}  # fmt: skip


def add_table(ns: torch.Tensor, m_star: torch.Tensor) -> dict:
    """ADD radius R(n, M*) (N, M) in metres; Yukawa strength alpha = 2n (torus); exclusions from the bundled limits."""
    R = equation.add_radius_m(ns[:, None], m_star[None, :])
    alpha = (2 * ns)[:, None].expand_as(R)
    lim = {row["source"]: float(row["lambda_max_um"]) * 1e-6 for row in tensors.torsion_limits()}
    lam_lee = lim["Lee et al."]
    torsion = equation.excluded_by_torsion(alpha, R, lam_lee)
    n_lhc, m_lhc = tensors.lhc_limits()
    lhc_min = torch.zeros_like(ns)
    for i, n in enumerate(ns.tolist()):
        hit = (n_lhc == n).nonzero()
        lhc_min[i] = m_lhc[hit[0, 0]] if len(hit) else float("nan")
    lhc = m_star[None, :] < lhc_min[:, None]  # NaN (n = 1) compares False
    return {"R": R, "alpha": alpha, "torsion": torsion, "lhc": lhc, "lam_limits": lim, "lhc_min": lhc_min}


def rs_scan(krc: torch.Tensor) -> torch.Tensor:
    """TeV-brane scale M_Pl e^(-k pi r_c) in GeV for every k r_c (K,)."""
    return tensors.M_PL_REDUCED_GEV * equation.rs_warp(krc)


def slice_field(half_width: float, n_grid: int, device) -> dict:
    """n = 1 slice (x, y): x in our 3D space, y around the compact circle (unrolled), window [-w, w]^2."""
    s = torch.linspace(-half_width, half_width, n_grid, dtype=torch.float64, device=device)
    y, x = torch.meshgrid(s, s, indexing="ij")
    xg = x.clone().requires_grad_(True)
    yg = y.clone().requires_grad_(True)
    phi = equation.slice_potential_n1(xg, yg)
    gx, gy = torch.autograd.grad(phi.sum(), (xg, yg))
    return {"x": s, "phi": phi.detach(), "Fx": -gx, "Fy": -gy, "w": half_width}


def window_scales(n_frames: int, w_max: float = 6 * math.pi, w_min: float = 0.25) -> torch.Tensor:
    """Zoom from far outside (w >> 2 pi R) to deep inside the compact space (w << R), log spaced."""
    return torch.logspace(math.log10(w_max), math.log10(w_min), n_frames, dtype=torch.float64)
