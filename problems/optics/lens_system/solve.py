"""Trace the ray fan through the two-lens system for every lens gap, and locate the image."""

import torch

from .equation import chain, free_space, image_distance, system_matrix, thin_lens


def ray_heights(z: torch.Tensor, rays: torch.Tensor, s: float, f1: float, d: float, f2: float) -> torch.Tensor:
    """Height of every ray at every axial position: (Z,), (R, 2) → (Z, R).

    For a point z inside segment k the transfer matrix is  M(z) = P(z - z_k) · B_k,  where B_k is
    everything the light has already passed: B_0 = 1, B_1 = L_1 P(s), B_2 = L_2 P(d) L_1 P(s).
    """
    l1, l2 = thin_lens(f1), thin_lens(f2)
    before = torch.stack([torch.eye(2), chain(free_space(s), l1), chain(free_space(s), l1, free_space(d), l2)])
    starts = torch.tensor([0.0, s, s + d])  # z_k where each segment begins   (3,)

    segment = (z >= s).long() + (z >= s + d).long()  # (Z,) which segment each z lies in
    m = torch.einsum("zij,zjk->zik", free_space(z - starts[segment]), before[segment])  # (Z, 2, 2)  M_ij(z)
    return torch.einsum("zij,rj->zri", m, rays)[..., 0]  # (Z, R)  keep the height component


def solve(
    rays: torch.Tensor, s: float, f1: float, f2: float, gaps: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (heights (F, Z, R), image position z_img (F,), image height (F,)) for every gap d."""
    h0 = rays[0, 0]
    heights, image_z, image_h = [], [], []
    for d in gaps.tolist():
        heights.append(ray_heights(z, rays, s, f1, d, f2))
        m = system_matrix(s, f1, d, f2)  # (2, 2)
        z_img = image_distance(m)  # measured from lens 2; negative ⇒ virtual image
        image_z.append(s + d + z_img)
        image_h.append((m[0, 0] + z_img * m[1, 0]) * h0)  # h' = (A + zC) h0
    return torch.stack(heights), torch.stack(image_z), torch.stack(image_h)
