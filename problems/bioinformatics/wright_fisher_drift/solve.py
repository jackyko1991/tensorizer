"""Run Wright–Fisher drift for many populations at once, the fixation experiment and Hardy–Weinberg genotype draws."""

import torch

from problems._common.progress import steps

from . import equation


def simulate(p0: torch.Tensor, N: int, s: float, generations: int, gen: torch.Generator) -> torch.Tensor:
    """Allele frequencies of every replicate in every generation.   p0 (R,) → (R, G+1)"""
    history = [p0]
    p = p0
    for _ in steps(generations, "generations"):
        p = equation.wright_fisher_step(p, N, s, gen)
        history.append(p)
    return torch.stack(history, dim=1)


def fixation(p0: torch.Tensor, N: int, s: torch.Tensor, gen: torch.Generator, max_gen: int) -> torch.Tensor:
    """Fraction of populations in which A is fixed, run until every population has fixed or lost it.

    p0 (S, R) with one row per selection value s (S,).  Returns (S,).
    """
    p = p0
    s_col = s[:, None]  # (S, 1) broadcast over replicates
    for _ in steps(max_gen, "until fixed or lost"):
        p = equation.wright_fisher_step(p, N, s_col, gen)
        if not ((p > 0) & (p < 1)).any():  # every population has fixed (p = 1) or lost (p = 0) the allele
            break
    return (p == 1).double().mean(dim=1)


def sample_genotypes(p: torch.Tensor, N: int, gen: torch.Generator) -> torch.Tensor:
    """Make N individuals per population from two random gametes each and count AA, Aa, aa.   (…,) → (…, 3) freqs."""
    gametes = torch.rand(*p.shape, N, 2, generator=gen, dtype=p.dtype, device=p.device) < p[..., None, None]
    copies = gametes.sum(dim=-1)  # (…, N) number of A copies of each individual: 0, 1 or 2
    return torch.stack([(copies == 2), (copies == 1), (copies == 0)], dim=-1).double().mean(dim=-2)
