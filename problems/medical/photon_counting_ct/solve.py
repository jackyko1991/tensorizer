"""Scan the phantom, count photons per bin, decompose every ray into basis line integrals and reconstruct."""

import torch

from problems._common.progress import steps

from . import equation, tensors


def basis_sinograms(maps: torch.Tensor, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Basis line integrals A_{θsm} in physical units (cm of water, cm of bone, mg/cm² iodine): (A, S, M)."""
    return (equation.radon(maps, theta, x) * tensors.FOV_HALF_WIDTH).permute(1, 2, 0)


def count_photons(lam: torch.Tensor, seed: int) -> torch.Tensor:
    """Poisson counts y_{rb} ~ Poisson(λ_{rb}) with a fixed seed."""
    gen = torch.Generator(device=lam.device).manual_seed(seed)
    return torch.poisson(lam, generator=gen)


def decompose(y: torch.Tensor, n0: float, s, R, mu, n_iter: int = 15) -> torch.Tensor:
    """Maximum-likelihood basis line integrals for every ray at once: y (R, B) counts → A (R, M).

    1. Start from weighted least squares on the log counts, −ln(y_b / y_{0b}) ≈ μ̄_{bm} A_m (μ̄ = bin-averaged μ,
       weights ≈ y_b because Var ln y ≈ 1/y). This ignores beam hardening inside a bin, so it is biased.
    2. Fisher scoring: the gradient of the Poisson NLL comes from autograd (rays are independent, so the gradient of
       the summed NLL is every ray's own gradient), the curvature from the Fisher matrix F_{rmn}, and
       A ← A − F⁻¹ ∇L solves one small M×M system per ray, batched.
    """
    w0 = R * s  # (B, E) open-beam photons per bin and energy
    mu_bar = (w0 @ mu) / w0.sum(1, keepdim=True)  # (B, M) bin-averaged attenuation at zero thickness
    log_ratio = -torch.log(y.clamp_min(0.5) / (n0 * w0.sum(1)))  # (R, B)
    W = y.clamp_min(1.0)  # (R, B) inverse variance of the log counts
    normal = torch.einsum("rb,bm,bn->rmn", W, mu_bar, mu_bar)
    rhs = torch.einsum("rb,bm,rb->rm", W, mu_bar, log_ratio)
    A = torch.linalg.solve(normal, rhs)  # (R, M) initial guess

    eye = 1e-12 * torch.eye(mu.shape[1], dtype=A.dtype, device=A.device)
    for _ in range(n_iter):
        A = A.detach().requires_grad_(True)
        lam = equation.expected_counts(n0, s, R, mu, A)
        (grad,) = torch.autograd.grad(equation.poisson_nll(y, lam).sum(), A)  # (R, M)
        with torch.no_grad():
            F = equation.fisher(equation.jacobian(n0, s, R, mu, A), lam)  # (R, M, M)
            step = torch.linalg.solve(F + eye, grad)
            A = A - step.clamp(-5.0, 5.0)  # keep early steps sane far from the optimum
    return A.detach()


def reconstruct(sino: torch.Tensor, theta: torch.Tensor, x: torch.Tensor):
    """Per-angle filtered back-projections and the final images of C sinograms (C, A, S) given in physical units.

    Returns (Q (C, A, N, N), images (C, N, N)); a running sum of Q over θ shows the image forming.
    """
    Q = equation.smear_filtered(sino / tensors.FOV_HALF_WIDTH, theta, x)
    return Q, equation.backproject(Q, theta)


def iodine_error_vs_dose(lam_unit, A_shape, s, R, mu, theta, x, maps, rois, doses, seed=1):
    """For each dose N_0: Poisson counts → decomposition → FBP; RMS error of the iodine map inside the inserts (mg/mL).

    lam_unit is the noise-free count per photon of open beam, λ_{rb}/N_0, of shape (R, B).
    """
    inside = rois.any(0)
    errors = []
    for k in steps(len(doses), "doses"):
        n0 = doses[k]
        y = count_photons(lam_unit * n0, seed + k)
        A = decompose(y, n0, s, R, mu).reshape(*A_shape)
        _, img = reconstruct(A[..., 2:].permute(2, 0, 1), theta, x)
        errors.append(float(((img[0] - maps[2])[inside] ** 2).mean().sqrt()))
    return errors


def roi_means(img: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
    """Mean of an image inside each ROI: (K,)."""
    return torch.einsum("kyx,yx->k", rois.to(img.dtype), img) / rois.sum((1, 2))
