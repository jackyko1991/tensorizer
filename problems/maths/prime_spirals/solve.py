"""Measure the primes against the Hardy–Littlewood predictions: quadratics, twins, Goldbach and gaps."""

import torch

from .equation import bateman_horn_constant, quadratic


def conjecture_f(coef: torch.Tensor, is_prime: torch.Tensor, k_max: int, primes: torch.Tensor) -> dict:
    """For every quadratic: the running prime count Q_f(K) and the prediction C(f) Σ_{3≤k≤K} 1/ln f(k).

    Returns the constants (F,), the observed counts (F, K) and the predicted counts (F, K) for K = 3 … k_max."""
    k = torch.arange(3, k_max + 1, device=coef.device)
    values = quadratic(coef, k)  # (F, K) all > 1 for k ≥ 3
    hits = is_prime[values].long()  # look every value up in the sieve at once
    observed = torch.cumsum(hits, dim=1)
    constant = bateman_horn_constant(coef, primes)  # (F,)
    predicted = constant[:, None] * torch.cumsum(1 / torch.log(values.double()), dim=1)
    return {"k": k, "constant": constant, "observed": observed, "predicted": predicted}


def cumulative_on(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """counts[x] for an integer grid x (a running count looked up at many x at once)."""
    return counts[x.long()].double()
