r"""Mathematics → tensor translation: Ricci flow ∂g/∂t = −2 Ric in two symmetric settings.

1. Surfaces: on a 2-sphere g = e^{2u} g_round, Ric = K g, and the normalised flow ∂g/∂t = 2(k̄ − K) g (k̄ = 4π/A
   keeps the area A) is the scalar PDE

       ∂u/∂t = k̄ − K,      K = e^{−2u} (1 − Δ_{S²} u),      (Δu)_i = L_ij u_j   (j summed)

   with L the flux-form Laplacian, so ∫K dA = 2π Σ_i (1 − L_ij u_j) w_i = 4π exactly (discrete Gauss–Bonnet).

2. Rotationally symmetric 3-spheres g = ds² + ψ(s)² g_{S²}. In arclength form −2 Ric gives
   ∂ψ/∂t = ψ_ss − (1 − ψ_s²)/ψ (Angenent–Knopf), with sectional curvatures K_rad = −ψ_ss/ψ (planes through ∂_s) and
   K_tan = (1 − ψ_s²)/ψ² (planes tangent to the S² slices). Every such metric is conformally round,
   g = e^{2u(x)}(dx² + sin²x g_{S²}) with ψ = e^u sin x, which is regular at the poles; we evolve u (see s3_rhs).
   The round sphere has K_rad = K_tan = 1/r²; a neck behaves like a cylinder S² × R with K_rad ≈ 0.
"""

import math

import torch


def gauss_curvature(u: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
    """K_bi = e^{−2u_bi}(1 − L_ij u_bj).   u (B, N), lap (N, N) → (B, N)"""
    return torch.exp(-2 * u) * (1 - torch.einsum("ij,bj->bi", lap, u))


def area(u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A_b = 2π Σ_i e^{2u_bi} w_i.   (B, N) → (B,)"""
    return 2 * math.pi * torch.einsum("bi,i->b", torch.exp(2 * u), w)


def total_curvature(k: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """∫K dA = 2π Σ_i K_bi e^{2u_bi} w_i (= 4π for every metric on the sphere).   → (B,)"""
    return 2 * math.pi * torch.einsum("bi,bi,i->b", k, torch.exp(2 * u), w)


def surface_rhs(u: torch.Tensor, w: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
    """Normalised Ricci flow on S²: ∂u/∂t = k̄ − K with k̄ = 4π / A.   (B, N) → (B, N)"""
    return (4 * math.pi / area(u, w))[:, None] - gauss_curvature(u, lap)


def curvature_spread(k: torch.Tensor) -> torch.Tensor:
    """(max K − min K) / mean K: 0 exactly when the curvature is uniform (a round sphere).   (B, N) → (B,)"""
    return (k.max(dim=-1).values - k.min(dim=-1).values) / k.mean(dim=-1)


def embed_profile(u: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Surface of revolution with the metric e^{2u}(dx² + sin²x dθ²): radius ρ = e^u sin x and height
    z = ∫ √(e^{2u} − ρ_x²) dx. Returns ρ, z (B, N) and the slack e^{2u} − ρ_x² (≥ 0 ⇔ embeddable in R³)."""
    rho = torch.exp(u) * torch.sin(x)
    rho_x = torch.gradient(rho, spacing=(x,), dim=-1)[0]
    slack = torch.exp(2 * u) - rho_x**2
    dz = torch.sqrt(slack.clamp_min(0))
    z = torch.cat([torch.zeros_like(dz[:, :1]), torch.cumulative_trapezoid(dz, x, dim=-1)], dim=-1)
    return rho, z - z[:, -1:] / 2, slack


def s3_curvatures(u: torch.Tensor, grid: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """Sectional curvatures of g = e^{2u}(dx² + sin²x g_{S²}) at the cell centres (u even about both poles):
    K_rad = e^{−2u}(1 − u_xx − u_x cot x),  K_tan = e^{−2u}(1 − u_x² − 2u_x cot x).
    grid = (x, x_η, x_ηη, h): cells uniform in η, x = X(η) denser near the poles (chain rule for ∂_x)."""
    x, xe, xee, h = grid
    ext = torch.cat([u[:, :1], u, u[:, -1:]], dim=1)  # mirror ghost cells: u is even about each pole
    u_e = (ext[:, 2:] - ext[:, :-2]) / (2 * h)
    u_ee = (ext[:, 2:] - 2 * u + ext[:, :-2]) / h**2
    u_x = u_e / xe
    u_xx = (u_ee - xee * u_x) / xe**2
    cot = torch.cos(x) / torch.sin(x)
    e = torch.exp(-2 * u)
    return e * (1 - u_xx - u_x * cot), e * (1 - u_x**2 - 2 * u_x * cot)


def s3_rhs(u: torch.Tensor, grid: tuple) -> torch.Tensor:
    """Ricci flow of g = e^{2u} g_round(S³) kept in this conformal form by a diffeomorphism X = ξ ∂_x (allowed: Ricci
    flow is only defined up to diffeomorphisms):

        ∂u/∂t = −(K_rad + K_tan) + (u_x + cot x) ξ,      ξ = sin x (∫₀^x (K_rad − K_tan)/sin x′ dx′ − c)

    c (half the full integral) keeps the gauge symmetric between the poles.   u (B, N) → (B, N)"""
    x, xe, xee, h = grid
    k_rad, k_tan = s3_curvatures(u, grid)
    integrand = (k_rad - k_tan) / torch.sin(x) * xe * h  # (B, N), vanishes like x² · (1/x) at the poles
    integral = torch.cumsum(integrand, dim=1) - 0.5 * integrand  # midpoint rule up to each centre
    xi = torch.sin(x) * (integral - 0.5 * integrand.sum(dim=1, keepdim=True))
    ext = torch.cat([u[:, :1], u, u[:, -1:]], dim=1)
    u_x = (ext[:, 2:] - ext[:, :-2]) / (2 * h * xe)
    return -(k_rad + k_tan) + (u_x + torch.cos(x) / torch.sin(x)) * xi


def profile(u: torch.Tensor, grid: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """Arclength s = ∫ e^u dx from the north pole and radius ψ = e^u sin x of the S² slices.   → (B, N) each"""
    x, xe, xee, h = grid
    ds = torch.exp(u) * xe * h
    return torch.cumsum(ds, dim=1) - 0.5 * ds, torch.exp(u) * torch.sin(x)


def neck_radius(psi: torch.Tensor) -> torch.Tensor:
    """Smallest interior local minimum of ψ (the neck); +∞ if ψ has none (a convex, sphere-like profile).   → (B,)"""
    mid = psi[:, 1:-1]
    is_min = (mid < psi[:, :-2]) & (mid < psi[:, 2:])
    return torch.where(is_min, mid, torch.inf).min(dim=-1).values
