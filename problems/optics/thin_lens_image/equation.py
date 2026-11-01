r"""Physics → tensor translation.

A thin lens of focal length f images an object at distance s to a distance s' behind the lens:

    \frac{1}{s} + \frac{1}{s'} = \frac{1}{f},        m = \frac{h'}{h} = -\frac{s'}{s}

s' < 0 means the image is on the same side as the object (virtual); m < 0 means it is upside down.

Behind both formulas is one rule for rays (paraxial, small angles): a ray crossing the lens at
height y keeps its height but its slope θ is bent by -y/f,

    \theta' = \theta - y / f

We treat every animation frame k (object distance s_k) and every principal ray r at once, so
the tensors carry indices k and r:

    s'_k = (1/f - 1/s_k)^{-1},    m_k = -s'_k / s_k,    \theta'_{kr} = \theta_{kr} - y_{kr} / f

No index is repeated, so nothing is summed: these are element-wise operations on tensors.
"""

import torch


def image_distance(s: torch.Tensor, f: float) -> torch.Tensor:
    """s' = 1 / (1/f − 1/s).   s: (K,) → (K,)"""
    return 1.0 / (1.0 / f - 1.0 / s)


def magnification(s: torch.Tensor, s_img: torch.Tensor) -> torch.Tensor:
    """m = −s'/s.   (K,) → (K,)"""
    return -s_img / s


def height_at_lens(h: float, s: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    """A ray leaving the object tip (x = −s, y = h) with slope θ reaches the lens at y = h + θ s.

    s: (K,), slope: (K, R) → (K, R)
    """
    return h + slope * s[:, None]


def lens_kink(y: torch.Tensor, slope: torch.Tensor, f: float) -> torch.Tensor:
    """θ' = θ − y/f: the slope of each ray after the lens.   (K, R) → (K, R)"""
    return slope - y / f
