r"""Physics → tensor translation.

In a medium whose refractive index varies smoothly, a light ray r(s) (s = arc length) obeys the
ray equation of geometrical optics,

    \frac{d}{ds}\Big(n \frac{d\mathbf r}{ds}\Big) = \nabla n

Introducing the optical direction vector p_i = n\, dr_i/ds turns it into a first-order system that
RK4 can step, for every ray r of a batch at once:

    \frac{dr_{ri}}{ds} = \frac{p_{ri}}{n(r_r)},        \frac{dp_{ri}}{ds} = \partial_i n(r_r)

Since dr/ds is a unit vector, p_{rj} p_{rj} = n^2 (repeated j ⇒ summed) must hold along each ray.

Above a hot road n depends on the height y only:  n(y) = n_\infty - \Delta n\, e^{-y/h}.
Then \partial_x n = 0, so p_x = n \cos\theta is conserved, θ being the angle to the horizontal:
Snell's law n_1 \cos\theta_1 = n_2 \cos\theta_2 for a stack of thin horizontal layers.
"""

import torch

N_INF = 1.000293  # refractive index of air far above the road


def index(y: torch.Tensor, dn: float, h: float) -> torch.Tensor:
    """n(y) = n∞ − Δn e^{−y/h}.   Any shape → same shape. (Below the road we use n(0).)"""
    return N_INF - dn * torch.exp(-y.clamp(min=0) / h)


def grad_index(r: torch.Tensor, dn: float, h: float) -> torch.Tensor:
    r"""\partial_i n = (0, Δn/h e^{−y/h}).   r: (R, 2) → (R, 2)"""
    dn_dy = dn / h * torch.exp(-r[:, 1].clamp(min=0) / h)  # (R,)
    return torch.stack([torch.zeros_like(dn_dy), dn_dy], dim=-1)


def rhs(state: torch.Tensor, dn: float, h: float) -> torch.Tensor:
    """d/ds of the state (R, 2, 2): state[:, 0] = position r_i, state[:, 1] = optical direction p_i."""
    r, p = state[:, 0], state[:, 1]
    n = index(r[:, 1], dn, h)  # (R,)
    dr_ds = p / n[:, None]  # (R, 2)
    dp_ds = grad_index(r, dn, h)  # (R, 2)
    return torch.stack([dr_ds, dp_ds], dim=1)


def direction_norm_sq(p: torch.Tensor) -> torch.Tensor:
    r"""p_{rj} p_{rj}  (should equal n²).   (..., R, 2) → (..., R)"""
    return torch.einsum("...rj,...rj->...r", p, p)


def snell_invariant(path: torch.Tensor, dn: float, h: float) -> torch.Tensor:
    """n(y) cos θ measured from the traced positions (S, R, 2) → (S − 1, R).

    θ comes from the chord between consecutive points, n from the height at the chord's midpoint.
    """
    step = path[1:] - path[:-1]  # (S − 1, R, 2)
    cos_theta = step[..., 0].abs() / step.norm(dim=-1)
    y_mid = 0.5 * (path[1:, :, 1] + path[:-1, :, 1])
    return index(y_mid, dn, h) * cos_theta
