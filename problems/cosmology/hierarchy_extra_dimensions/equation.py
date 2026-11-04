"""Physics -> tensors: gravity on R^3 x T^n as an image sum and as a Kaluza-Klein sum, ADD and Randall-Sundrum.

Units: compact radius R = 1 (each circle has circumference L = 2 pi), G_D M = 1. A shell index m = |k|^2 groups the
mode vectors k of Z^n; r_n(m) counts them, so a sum over k becomes a sum over m (summed index, shapes in comments).
"""

import math

import torch

from . import tensors

L = 2 * math.pi  # circumference of each compact circle (R = 1)


def sphere_area(k: int) -> float:
    """Area of the unit sphere S^k in R^(k+1): 2 pi^((k+1)/2) / Gamma((k+1)/2)."""
    return 2 * math.pi ** ((k + 1) / 2) / math.gamma((k + 1) / 2)


def g4(n: int) -> float:
    """Effective 4D Newton constant from Gauss's law: G_4 = (n+1) |S^(n+2)| G_D / (4 pi (2 pi R)^n)."""
    return (n + 1) * sphere_area(n + 2) / (4 * math.pi * L**n)


def potential_images(r: torch.Tensor, n: int, counts: torch.Tensor, n_theta: int = 4001) -> torch.Tensor:
    """phi(r) = -sum_k G_D M / (r^2 + |k L|^2)^((n+1)/2): the mass and its images around the circles (r: (R,)).

    Shells m <= m_max are summed exactly; farther images are replaced by their continuum integral
    |S^(n-1)| / L^n * (1/r) * int_{theta_c}^{pi/2} sin^(n-1)(theta) d theta  (y = r tan theta).
    """
    m = torch.arange(len(counts), dtype=r.dtype, device=r.device)  # (M,)
    near = torch.einsum("m,rm->r", counts, (r[:, None] ** 2 + m * L**2) ** (-(n + 1) / 2))  # repeated m => summed
    ball = math.pi ** (n / 2) / math.gamma(n / 2 + 1)  # volume of the unit n-ball
    rho_c = L * (counts.sum().item() / ball) ** (1 / n)  # radius of the ball that holds as many lattice points
    t = torch.linspace(0, 1, n_theta, dtype=r.dtype, device=r.device)
    theta_c = torch.atan(rho_c / r)  # (R,)
    theta = theta_c[:, None] + (math.pi / 2 - theta_c[:, None]) * t  # (R, T)
    tail = torch.trapezoid(torch.sin(theta) ** (n - 1), theta, dim=1)
    return -(near + sphere_area(n - 1) / L**n * tail / r)


def kk_deviation(r: torch.Tensor, n: int, counts: torch.Tensor) -> torch.Tensor:
    """sum_{m >= 1} r_n(m) e^(-sqrt(m) r / R): the massive Kaluza-Klein modes, Yukawa tails of mass sqrt(m)/R (R,)."""
    m = torch.arange(1, len(counts), dtype=r.dtype, device=r.device)
    return torch.einsum("m,rm->r", counts[1:], torch.exp(-torch.sqrt(m) * r[:, None]))


def potential_kk(r: torch.Tensor, n: int, counts: torch.Tensor) -> torch.Tensor:
    """phi(r) = -(G_4 M / r) sum_k e^(-|k| r / R): the zero mode (Newton) plus the KK tower (R,)."""
    return -(g4(n) / r) * (1 + kk_deviation(r, n, counts))


def radial_field(phi, r: torch.Tensor) -> torch.Tensor:
    """d phi / dr by autograd (the pull towards the mass), for any potential function phi(r)."""
    x = r.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(phi(x).sum(), x)
    return grad


def log_slope(phi, r: torch.Tensor) -> torch.Tensor:
    """Local power law d ln|phi| / d ln r (-(1+n) inside the compact space, -1 outside)."""
    r = r.to(torch.float64)
    return r * radial_field(phi, r) / phi(r)


def yukawa(r, alpha: float, lam: float, g: float = 1.0):
    """V = -(G M / r) (1 + alpha e^(-r / lambda)): the standard parametrisation of short-range deviations."""
    return -(g / r) * (1 + alpha * torch.exp(-r / lam))


def fit_yukawa(r: torch.Tensor, deviation: torch.Tensor) -> tuple[float, float]:
    """Least squares ln(V/V_Newton - 1) = ln alpha - r / lambda -> (alpha, lambda)."""
    design = torch.stack([torch.ones_like(r), -r], 1)  # (R, 2)
    coef = torch.linalg.lstsq(design, torch.log(deviation)[:, None]).solution[:, 0]
    return math.exp(coef[0].item()), 1 / coef[1].item()


def add_radius_m(n: torch.Tensor, m_star_tev: torch.Tensor) -> torch.Tensor:
    """ADD: M_Pl^2 = M*^(n+2) R^n  =>  R = (M_Pl^2 / M*^(n+2))^(1/n), in metres (broadcast over n and M*)."""
    log_r = (2 * math.log(tensors.M_PL_GEV) - (n + 2) * torch.log(m_star_tev * 1e3)) / n  # ln(R * GeV)
    return torch.exp(log_r) * tensors.HBARC_M


def rs_warp(krc: torch.Tensor) -> torch.Tensor:
    """Randall-Sundrum warp factor e^(-k pi r_c): mass M_Pl on one brane looks like
    M_Pl e^(-k pi r_c) on the other."""
    return torch.exp(-math.pi * krc)


def weak_over_gravity() -> float:
    """G_F / G_N = G_F M_Pl^2: how much stronger the weak interaction is than gravity at the same energy."""
    return tensors.G_FERMI * tensors.M_PL_GEV**2


def slice_potential_n1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """n = 1, closed form of the image sum -sum_k 1/(x^2 + (y - kL)^2) for L = 2 pi:
    -(sinh|x| / 2|x|) / (cosh x - cos y)."""
    ax = x.abs().clamp_min(1e-9)
    return -torch.sinh(ax) / (2 * ax) / (torch.cosh(ax) - torch.cos(y))


def excluded_by_torsion(alpha: torch.Tensor, lam_m: torch.Tensor, lam_limit_m: float) -> torch.Tensor:
    """Conservative: the measured exclusion curve falls with lambda, so alpha >= 1 and
    lambda > lambda(alpha=1) is out."""
    return (alpha >= 1) & (lam_m > lam_limit_m)
