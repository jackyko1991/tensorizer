"""One batched autograd fit: four models × all galaxies, minimised together (Adam, then L-BFGS).

The four models share no parameters, so the sum of their χ² is minimised exactly when each is; MOND's a₀ is one
number for all galaxies. Parameters live in log space (they are all positive). Mass-to-light ratios carry a
log-normal prior of 0.1 dex around the stellar-population values Υ_d = 0.5, Υ_b = 0.7 (as in SPARC halo fits).
"""

import math

import torch

from problems._common.progress import steps

from . import equation, tensors

MODELS = ["baryons", "nfw", "burkert", "mond"]
LOG_RS = (math.log(0.3), math.log(300.0))  # halo scale radius kept between 0.3 and 300 kpc
LOG_VS = (math.log(5.0), math.log(2000.0))


def initial_parameters(data: dict) -> dict:
    """Raw (log) parameters for every model, each tensor (G,) except log a₀ (scalar)."""
    G = data["R"].shape[0]
    V_max = (data["V"] * data["mask"]).amax(1)
    R_max = (data["R"] * data["mask"]).amax(1)
    ones = torch.ones(G, dtype=tensors.F64, device=data["R"].device)

    def base():
        return {"ups_d": ones * math.log(tensors.UPS_DISK), "ups_b": ones * math.log(tensors.UPS_BULGE)}

    raw = {m: base() for m in MODELS}
    raw["nfw"].update(V_s=torch.log(2.0 * V_max), r_s=torch.log(0.5 * R_max))
    raw["burkert"].update(V_s=torch.log(1.5 * V_max), r_s=torch.log(0.3 * R_max))
    raw["mond"]["a0"] = torch.tensor(math.log(tensors.A0_DEFAULT), dtype=tensors.F64, device=data["R"].device)
    return {m: {k: v.clone().requires_grad_(True) for k, v in p.items()} for m, p in raw.items()}


def physical(raw: dict) -> dict:
    """exp of the raw parameters, with the halo parameters clamped to their ranges."""
    p = {}
    for k, v in raw.items():
        if k == "r_s":
            v = v.clamp(*LOG_RS)
        elif k == "V_s":
            v = v.clamp(*LOG_VS)
        p[k] = torch.exp(v)
    return p


def prior(raw: dict, data: dict, dex: float | None) -> torch.Tensor:
    """Σ_g ((log₁₀Υ − log₁₀Υ₀)/dex)² for disks and (where present) bulges."""
    if dex is None:
        return torch.zeros((), dtype=tensors.F64, device=data["R"].device)
    ld = (raw["ups_d"] - math.log(tensors.UPS_DISK)) / math.log(10) / dex
    lb = (raw["ups_b"] - math.log(tensors.UPS_BULGE)) / math.log(10) / dex
    return (ld**2).sum() + (lb**2 * data["has_bulge"]).sum()


def fit_all(data: dict, iterations: int = 800, lr: float = 0.05) -> dict:
    """Fit all models to all galaxies. Returns, per model, the parameters, χ² (G,) and reduced χ² (G,)."""
    raw = initial_parameters(data)
    dex = data.get("prior_dex", 0.1)
    params = [v for p in raw.values() for v in p.values()]

    def objective():
        return sum(equation.chi2(equation.model_v2(m, physical(raw[m]), data), data).sum() + prior(raw[m], data, dex)
                   for m in MODELS)  # fmt: skip

    opt = torch.optim.Adam(params, lr=lr)
    for _ in steps(iterations, "Adam"):  # robust first descent
        opt.zero_grad()
        objective().backward()
        opt.step()
    polish = torch.optim.LBFGS(params, max_iter=300, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        polish.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    polish.step(closure)  # quasi-Newton to the bottom of the valley

    n = data["mask"].sum(1).to(tensors.F64)
    k_bar = 1 + data["has_bulge"]  # Υ_d (and Υ_b)
    n_par = {"baryons": k_bar, "nfw": k_bar + 2, "burkert": k_bar + 2, "mond": k_bar}
    out = {}
    with torch.no_grad():
        for m in MODELS:
            p = physical(raw[m])
            c2 = equation.chi2(equation.model_v2(m, p, data), data)
            out[m] = {**p, "chi2": c2, "chi2_nu": c2 / (n - n_par[m]).clamp(min=1)}
    return out


def model_curves(fit: dict, data: dict, n: int = 120) -> tuple[torch.Tensor, dict]:
    """Smooth model curves on a fine radius grid: SPARC baryon speeds interpolated linearly, (G, n) each."""
    R_max = (data["R"] * data["mask"]).amax(1, keepdim=True)
    R_min = data["R"][:, :1]  # first measured radius (the baryon models are only known where SPARC measured)
    s = torch.linspace(0, 1, n, dtype=tensors.F64, device=R_max.device)[None, :]
    R = R_min + s * (R_max - R_min)  # (G, n)
    fine = {"R": R, "mask": torch.ones_like(R, dtype=torch.bool)}
    for k in ("gas", "disk", "bul"):
        fine[k] = torch.stack([_interp(R[g], data["R"][g][data["mask"][g]], data[k][g][data["mask"][g]])
                               for g in range(R.shape[0])])  # fmt: skip
    curves = {m: equation.total_velocity(equation.model_v2(m, fit[m], fine), torch.zeros_like(R)) for m in MODELS}
    return R, curves


def _interp(x, xp, fp):
    """Linear interpolation (flat beyond the ends), like numpy.interp."""
    k = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    w = ((x - xp[k - 1]) / (xp[k] - xp[k - 1])).clamp(0, 1)
    return fp[k - 1] * (1 - w) + fp[k] * w
