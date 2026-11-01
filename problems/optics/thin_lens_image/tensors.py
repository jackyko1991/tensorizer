"""Build the concrete tensors: the object distance in every frame and the slopes of the three principal rays."""

import torch


def object_distances(s_start: float, s_end: float, f: float, n_frames: int = 120) -> torch.Tensor:
    """Object distance s_k for every frame k: shape (K,).

    s = f exactly would put the image at infinity, so any sample landing on it is nudged a little.
    """
    s = torch.linspace(s_start, s_end, n_frames)
    return torch.where((s - f).abs() < 1e-3, s + 1e-2, s)


def principal_slopes(h: float, s: torch.Tensor, f: float) -> torch.Tensor:
    """Slope of each principal ray as it leaves the object tip (x = −s, y = h): shape (K, 3).

    column 0: parallel ray          θ = 0
    column 1: ray through the centre  θ = −h / s            (aims at x = 0, y = 0)
    column 2: ray through near focus  θ = −h / (s − f)      (aims at x = −f, y = 0)
    """
    parallel = torch.zeros_like(s)
    centre = -h / s
    focal = -h / (s - f)
    return torch.stack([parallel, centre, focal], dim=-1)
