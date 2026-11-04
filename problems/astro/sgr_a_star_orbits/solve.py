"""Fit the orbit: weighted residuals, their Jacobian by autograd, and batched Levenberg–Marquardt from many starts."""

import torch
from torch.func import jvp, vmap

from problems._common.progress import steps

from . import tensors
from .equation import observables, period


def residuals(theta: torch.Tensor, data: dict) -> torch.Tensor:
    """(model − data)/σ for every RA, Dec and RV measurement: θ (B, 8) → (B, R), R = 2 Na + Nv."""
    p = tensors.constrained(theta)
    ra, dec, _ = observables(p, data["t_a"])
    _, _, rv = observables(p, data["t_v"])
    return torch.cat(
        [(ra - data["ra"]) / data["ra_err"], (dec - data["dec"]) / data["dec_err"], (rv - data["rv"]) / data["rv_err"]],
        dim=-1,
    )


def jacobian(theta: torch.Tensor, data: dict) -> torch.Tensor:
    """J_{brk} = ∂r_{br}/∂θ_{bk} by forward-mode autograd: (B, R, 8).

    One Jacobian-vector product per parameter direction k (θ_k nudged in every orbit of the batch at once; the
    orbits are independent), and the 8 directions themselves are vectorised with vmap.
    """
    directions = torch.eye(theta.shape[-1], dtype=theta.dtype)[:, None, :].expand(-1, *theta.shape)  # (8, B, 8)

    def column(v):  # ∂r/∂θ · v for the whole batch: (B, R)
        return jvp(lambda th: residuals(th, data), (theta,), (v,))[1]

    return vmap(column)(directions).permute(1, 2, 0)


def fit(data: dict, n_starts: int = 256, n_iter: int = 60, gen: torch.Generator | None = None) -> dict:
    """Levenberg–Marquardt on every starting orbit at once; returns all fits and the best one.

    Each step solves (JᵀJ + λ diag JᵀJ) δ = −Jᵀr for the whole batch (a (B, 8, 8) solve); λ shrinks after an
    accepted step and grows after a rejected one, separately for every start.
    """
    gen = gen or torch.Generator().manual_seed(0)
    theta = tensors.unconstrained(tensors.random_starts(n_starts, gen))  # (B, 8)
    lam = torch.full((n_starts,), 1e-2, dtype=theta.dtype)
    chi2 = residuals(theta, data).pow(2).sum(-1)
    chi2 = torch.nan_to_num(chi2, nan=torch.inf)
    history = [chi2.clone()]
    for _ in steps(n_iter, "Levenberg–Marquardt"):
        r = residuals(theta, data)  # (B, R)
        jac = torch.nan_to_num(jacobian(theta, data))  # (B, R, 8)
        jtj = torch.einsum("brk,brl->bkl", jac, jac)  # r summed
        grad = torch.einsum("brk,br->bk", jac, torch.nan_to_num(r))
        damp = lam[:, None, None] * torch.diag_embed(torch.diagonal(jtj, dim1=-2, dim2=-1) + 1e-9)
        delta = torch.linalg.solve(jtj + damp, -grad[..., None])[..., 0]
        trial = theta + delta
        chi2_trial = torch.nan_to_num(residuals(trial, data).pow(2).sum(-1), nan=torch.inf)
        better = chi2_trial < chi2
        theta = torch.where(better[:, None], trial, theta)
        chi2 = torch.where(better, chi2_trial, chi2)
        lam = torch.where(better, lam / 3, lam * 4).clamp(1e-9, 1e9)
        history.append(chi2.clone())
    p = tensors.constrained(theta)
    k = int(chi2.argmin())
    best = {name: p[k, i].item() for i, name in enumerate(tensors.NAMES)}
    if best["inc"] % 360.0 > 180.0:  # (i, Ω, ω) and (360° − i, Ω + 180°, ω + 180°) are the same orbit
        best["inc"], best["Omega"], best["omega"] = (
            360.0 - best["inc"] % 360.0,
            best["Omega"] + 180,
            best["omega"] + 180,
        )
    for key in ("inc", "Omega", "omega"):
        best[key] = best[key] % 360.0
    best["P"] = period(p[k : k + 1]).item()
    return {"p": p, "chi2": chi2, "history": torch.stack(history), "best": best, "k": k,
            "errors": errors(theta[k : k + 1], data, chi2[k].item())}  # fmt: skip


def errors(theta: torch.Tensor, data: dict, chi2: float) -> dict:
    """1σ errors of the physical parameters from the covariance (JᵀJ)⁻¹, scaled by χ²/dof, via the chain rule."""
    jac = jacobian(theta, data)[0]  # (R, 8)
    cov_theta = torch.linalg.inv(jac.T @ jac) * max(1.0, chi2 / (data["n_res"] - 8))
    dp = torch.func.jacfwd(lambda th: tensors.constrained(th[None])[0])(theta[0])  # (8, 8) ∂p/∂θ
    cov = dp @ cov_theta @ dp.T
    sig = cov.diagonal().sqrt()
    out = {name: sig[i].item() for i, name in enumerate(tensors.NAMES)}
    dper = torch.func.jacfwd(lambda th: period(tensors.constrained(th[None]))[0])(theta[0])
    out["P"] = (dper @ cov_theta @ dper).sqrt().item()
    return out
