"""The solver: one DFT, then every truncated chain of circles at once."""

import torch

from . import equation, tensors


def stages(n: int, max_terms: int) -> list[int]:
    """How many circles each stage of the animation keeps: 1, 2, 4, 8, … up to max_terms (and at most N)."""
    top = min(n, max_terms)
    m, out = 1, []
    while m < top:
        out.append(m)
        m *= 2
    return out + [top]


def analyse(z: torch.Tensor, m_stages: list[int], n_dense: int = 600) -> dict:
    """DFT of the samples, the error with every number of circles M = 1 … N, and the curves of the stages.

    Returns c, k, order (biggest circle first), err (N,) for M = 1 … N, the Parseval prediction (N,),
    the stage curves S (stages, n_dense) on a fine time grid, and the energy check.
    """
    n = z.shape[0]
    c, k = equation.dft(z)
    order = equation.by_radius(c)
    t = tensors.times(n, z.device)
    m_all = torch.arange(1, n + 1, device=z.device)
    err = equation.sample_error(z, equation.partial_sums(c, k, order, m_all, t))  # (N,) for M = 1 … N
    energy = c[order].abs() ** 2  # (N,) biggest first
    left_out = energy.sum() - torch.cumsum(energy, 0)  # (N,) energy of the circles NOT kept with M = 1 … N
    t_dense = tensors.times(n_dense, z.device)
    curves = equation.partial_sums(c, k, order, torch.tensor(m_stages, device=z.device), t_dense)  # (stages, T)
    return {
        "c": c,
        "k": k,
        "order": order,
        "err": err,
        "parseval": left_out.clamp_min(0),
        "curves": curves,
        "t_dense": t_dense,
        "power": equation.mean_power(z).item(),
        "coef_power": energy.sum().item(),
    }
