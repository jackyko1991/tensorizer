r"""Physics → tensor translation.

Light passing a black hole of mass M follows a null geodesic. In the equatorial plane of the Schwarzschild metric
the orbit equation for u = 1/r as a function of the angle φ is (units G = c = 1)

    \frac{d^2 u_b}{dφ^2} = -u_b + 3M u_b^2

for a batch of rays b, each coming in from infinity with impact parameter b: u = 0, du/dφ = 1/b at φ = 0.
Without the 3Mu² term the ray is a straight line; with it, the ray is bent by α (it leaves at φ = π + α).
For b ≫ M the deflection is Einstein's α = 4GM/(c² b). Rays with b < \sqrt{27}\,M spiral in: that is the
black hole's shadow, the edge of which is the photon sphere at r = 3M.

Far from the hole the thin-lens approximation applies. Angles θ on the sky (in units of the Einstein radius
θ_E = \sqrt{4GM D_{ls} / (c^2 D_l D_s)}) map back to the source plane by the lens equation, for every pixel p:

    β_{pi} = θ_{pi} - \frac{θ_{pi}}{θ_{pj} θ_{pj}}      (j summed)

Surface brightness is conserved, so the lensed image is I(θ) = S(β(θ)); a source right behind the lens becomes an
Einstein ring, otherwise two images. A point source at offset u (in θ_E) is magnified by μ = (u² + 2)/(u\sqrt{u² + 4}).
"""

import math

import torch


def binet_rhs(y: torch.Tensor, M: float) -> torch.Tensor:
    """d/dφ of the state y (B, 2) = (u, du/dφ) for every ray."""
    u, du = y[:, 0], y[:, 1]
    return torch.stack([du, -u + 3 * M * u**2], dim=1)


def thin_lens_deflection(b: torch.Tensor, M: float) -> torch.Tensor:
    """Einstein's weak-field deflection α = 4GM/(c²b)."""
    return 4 * M / b


def shadow_radius(M: float) -> float:
    r"""Critical impact parameter \sqrt{27} GM/c²: rays closer than this are captured."""
    return math.sqrt(27) * M


def einstein_radius(GM_over_c2: float, D_l: float, D_s: float) -> float:
    r"""θ_E = \sqrt{4GM D_{ls} / (c^2 D_l D_s)} in radians (D_ls = D_s − D_l for flat space)."""
    return math.sqrt(4 * GM_over_c2 * (D_s - D_l) / (D_l * D_s))


def lens_map(theta: torch.Tensor) -> torch.Tensor:
    r"""β_{pi} = θ_{pi} − θ_{pi} / (θ_{pj} θ_{pj}) for any batch of sky positions (θ_E units).   (..., 2) → (..., 2)"""
    r2 = torch.einsum("...j,...j->...", theta, theta)[..., None]
    return theta - theta / r2


def source_brightness(beta: torch.Tensor, centre: torch.Tensor, radius: float) -> torch.Tensor:
    """A round source with a soft edge, centred at `centre` (..., 2) and evaluated at β (..., 2) → (...)."""
    d = (beta - centre).norm(dim=-1)
    return torch.sigmoid((radius - d) / (0.15 * radius))


def point_magnification(u: torch.Tensor) -> torch.Tensor:
    r"""μ = (u² + 2) / (u \sqrt{u² + 4}) for a point source at offset u (θ_E units), both images together."""
    return (u**2 + 2) / (u * torch.sqrt(u**2 + 4))
