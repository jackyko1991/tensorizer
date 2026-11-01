r"""Physics → tensor translation.

Bundle adjustment finds the camera poses and 3D points that best explain what the cameras saw. Camera c has a
rotation R_c = exp([ω_c]×) (axis-angle ω_c) and a translation t_c; point p has position X_p. Indices: c = camera,
p = point, i, j = 3D axes, a = image axis (u, v), m = residual, n = unknown.

    rotation      R_{cij} = δ_{ij} + sin θ_c K_{cij} + (1 - cos θ_c) K_{cik} K_{ckj}   (Rodrigues, K = [ω_c/θ_c]×)
    to camera     x_{cpi} = R_{cij} X_{pj} + t_{ci}                     j summed
    projection    u_{cpa} = f x_{cpa} / x_{cp3} + c_a
    residual      r_{cpa} = u_{cpa} - ũ_{cpa}                           ũ = the measured (noisy) image point
    cost          E = ½ r_m r_m                                          m = (c, p, a) flattened, summed

Gauss–Newton linearises r(θ + Δ) ≈ r + J Δ with the Jacobian J_{mn} = ∂r_m / ∂θ_n (from autograd) and solves the
normal equations (JᵀJ)_{nn'} Δ_{n'} = -(Jᵀr)_n; Levenberg–Marquardt adds λ diag(JᵀJ) to keep each step safe.
"""

import torch


def rodrigues(omega: torch.Tensor) -> torch.Tensor:
    """Axis-angle ω (C, 3) → rotation matrices R (C, 3, 3) = I + sin θ K + (1 - cos θ) K², θ = |ω|, K = [ω/θ]×."""
    theta = torch.sqrt((omega**2).sum(-1) + 1e-20)  # (C,); the tiny shift keeps the gradient finite at ω = 0
    k = omega / theta[:, None]
    zero = torch.zeros_like(theta)
    K = torch.stack(
        [
            torch.stack([zero, -k[:, 2], k[:, 1]], -1),
            torch.stack([k[:, 2], zero, -k[:, 0]], -1),
            torch.stack([-k[:, 1], k[:, 0], zero], -1),
        ],
        -2,
    )  # (C, 3, 3) cross-product matrix: K v = k × v
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    s, c = torch.sin(theta)[:, None, None], torch.cos(theta)[:, None, None]
    return eye + s * K + (1 - c) * torch.einsum("cik,ckj->cij", K, K)


def to_camera(omega: torch.Tensor, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """x_{cpi} = R_{cij} X_{pj} + t_{ci}   (j summed): every point in every camera frame.   → (C, P, 3)."""
    return torch.einsum("cij,pj->cpi", rodrigues(omega), X) + t[:, None, :]


def project(x_cam: torch.Tensor, focal: float, centre: float) -> torch.Tensor:
    """Pinhole camera u_a = f x_a / x_3 + c_a.   (..., 3) → (..., 2) pixels."""
    return focal * x_cam[..., :2] / x_cam[..., 2:3] + centre


def residuals(omega, t, X, observed, focal, centre) -> torch.Tensor:
    """Reprojection errors r_{cpa} = π(R_c X_p + t_c) - ũ_{cp}: (C, P, 2) pixels."""
    return project(to_camera(omega, t, X), focal, centre) - observed


def cost(r: torch.Tensor) -> torch.Tensor:
    """E = ½ Σ_m r_m²: the total squared reprojection error."""
    return 0.5 * (r**2).sum()


def rms(r: torch.Tensor) -> torch.Tensor:
    """Root-mean-square reprojection error per image coordinate, in pixels."""
    return torch.sqrt((r**2).mean())


def normal_equations(J: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """H_{nn'} = J_{mn} J_{mn'} and g_n = J_{mn} r_m   (m summed): the Gauss–Newton matrix JᵀJ and gradient Jᵀr."""
    return torch.einsum("mn,mk->nk", J, J), torch.einsum("mn,m->n", J, r)
