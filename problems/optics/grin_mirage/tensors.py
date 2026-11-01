"""Build the concrete tensors: the fan of viewing directions leaving the eye, as an RK4 state."""

import torch

from .equation import index

DTYPE = torch.float64  # n cos θ varies in the 7th decimal place: float32 is not enough


def viewing_angles(max_dip_mrad: float, n_rays: int) -> torch.Tensor:
    """Angles above the horizontal, from −max_dip (looking down at the road) to +2 mrad: (R,) in radians."""
    return torch.linspace(-max_dip_mrad, 2.0, n_rays, dtype=DTYPE) * 1e-3


def initial_state(angles: torch.Tensor, eye_height: float, distance: float, dn: float, h: float) -> torch.Tensor:
    """state[r] = (r_i, p_i): every ray starts at the eye (x = L, y = eye height) heading towards x = 0.

    state[r, 0, :] = (L, y_eye)                        position r_{ri}               (R, 2, 2)
    state[r, 1, :] = n(y_eye) (−cos α_r, sin α_r)      optical direction p_{ri} = n t_{ri}
    """
    r0 = torch.tensor([distance, eye_height], dtype=DTYPE).expand(len(angles), 2)
    n0 = index(torch.tensor(eye_height, dtype=DTYPE), dn, h)
    p0 = n0 * torch.stack([-torch.cos(angles), torch.sin(angles)], dim=-1)  # (R, 2)
    return torch.stack([r0, p0], dim=1)
