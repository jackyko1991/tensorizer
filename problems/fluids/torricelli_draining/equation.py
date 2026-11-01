r"""Physics → tensor translation.

Torricelli: water leaves a small hole at the speed of a stone dropped from the surface,
v = \sqrt{2 g h}. The volume lost per second, a v, lowers the level of a tank of cross-section A:

    A \dot h = -a \sqrt{2 g h}

For a batch of tanks b with different hole areas a_b (all elementwise, no summed index):

    \dot h_b = -(a_b / A) \sqrt{2 g h_b}

Separating variables gives the exact solution and the time to empty the tank:

    h_b(t) = (\sqrt{h_0} - (a_b/A) \sqrt{g/2}\; t)^2,      T_b = (A / a_b) \sqrt{2 h_0 / g}
"""

import math

import torch


def outflow_speed(h: torch.Tensor, g: float) -> torch.Tensor:
    """Torricelli's law v_b = sqrt(2 g h_b) (zero once the tank is empty).   (B,) → (B,)"""
    return torch.sqrt(2 * g * h.clamp(min=0.0))


def rhs(h: torch.Tensor, a: torch.Tensor, A: float, g: float) -> torch.Tensor:
    r"""\dot h_b = -(a_b / A) \sqrt{2 g h_b}.   (B,) → (B,)"""
    return -(a / A) * outflow_speed(h, g)


def analytic_height(t: torch.Tensor, a: torch.Tensor, A: float, h0: float, g: float) -> torch.Tensor:
    r"""h_b(t) = (\sqrt{h_0} - (a_b/A)\sqrt{g/2}\,t)^2, clamped at 0 after the drain time.   t: (T,) → (T, B)"""
    root = math.sqrt(h0) - torch.einsum("t,b->tb", t, a / A) * math.sqrt(g / 2)  # outer product t_t a_b
    return root.clamp(min=0.0) ** 2


def drain_time(a: torch.Tensor, A: float, h0: float, g: float) -> torch.Tensor:
    r"""T_b = (A / a_b) \sqrt{2 h_0 / g}.   (B,) → (B,)"""
    return (A / a) * math.sqrt(2 * h0 / g)
