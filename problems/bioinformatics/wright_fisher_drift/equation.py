r"""Population genetics → tensor translation.

One gene with two alleles A and a in a diploid population of N individuals (2N gene copies). p is the frequency of A.

Hardy–Weinberg: if gametes meet at random, the genotypes AA, Aa, aa occur with frequencies

    G_{r·} = (p_r², 2 p_r q_r, q_r²),      q = 1 - p

Wright–Fisher: every generation is 2N gene copies drawn at random from the previous one. With selection, A copies
are (1 + s) times as likely to be passed on, which first shifts the frequency to p' and then the drawing adds noise:

    p'_{rt}    = p_{rt} (1 + s) / (1 + s p_{rt})
    p_{r,t+1}  = K_{rt} / 2N,         K_{rt} ~ Binomial(2N, p'_{rt})

Index r labels independent replicate populations, t the generation; every replicate advances in one tensor call.
No index is summed in the update itself; averages over replicates (Σ_r / R) give the curves compared with theory:

    E[H_t] = H_0 (1 - 1/2N)^t,  H = 2pq        (neutral drift)
    u(p_0) = (1 - e^{-4Nsp_0}) / (1 - e^{-4Ns})  (Kimura: chance that A is eventually fixed; u = p_0 for s = 0)
"""

import torch


def hardy_weinberg(p: torch.Tensor) -> torch.Tensor:
    """G_{…g} = (p², 2pq, q²): genotype frequencies of AA, Aa, aa under random mating.   (…,) → (…, 3)"""
    q = 1 - p
    return torch.stack([p * p, 2 * p * q, q * q], dim=-1)


def heterozygosity(p: torch.Tensor) -> torch.Tensor:
    """H = 2pq: chance that two gene copies drawn at random differ.   same shape as p"""
    return 2 * p * (1 - p)


def select(p: torch.Tensor, s: float | torch.Tensor) -> torch.Tensor:
    """p' = p (1 + s) / (1 + s p): allele A has relative fitness 1 + s (per copy).   same shape as p"""
    return p * (1 + s) / (1 + s * p)


def wright_fisher_step(p: torch.Tensor, N: int, s: float | torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """p_{t+1} = Binomial(2N, p'_t) / 2N for every replicate at once.   (…,) → (…,)"""
    p_sel = select(p, s)
    copies = torch.full_like(p_sel, 2 * N)
    return torch.binomial(copies, p_sel, generator=gen) / (2 * N)


def expected_heterozygosity(H0: float, N: int, t: torch.Tensor) -> torch.Tensor:
    """E[H_t] = H_0 (1 - 1/2N)^t: each generation two copies share a parent with probability 1/2N.   (T,)"""
    return H0 * (1 - 1 / (2 * N)) ** t


def kimura_fixation(p0: float, N: int, s: torch.Tensor) -> torch.Tensor:
    """u = (1 - e^{-4N s p0}) / (1 - e^{-4N s}), and u = p0 in the neutral limit s → 0.   (S,)"""
    x = 4 * N * s
    neutral = x.abs() < 1e-9
    safe = torch.where(neutral, torch.ones_like(x), x)  # avoid 0 / 0; replaced by p0 below
    u = torch.expm1(-safe * p0) / torch.expm1(-safe)
    return torch.where(neutral, torch.full_like(x, p0), u)
