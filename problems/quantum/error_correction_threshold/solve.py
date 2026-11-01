"""Monte Carlo memory experiment + exact minimum-weight decoder on the space–time syndrome history.

Decoder. The unknown is the accumulated error pattern x_t ∈ {0, 1}^d after each round. A guess for the
whole history costs w_data per data flip (x_t differs from x_{t−1}) plus w_meas per check whose reading
disagrees with H x_t. The cheapest history is the minimum-weight explanation of the syndromes — the
same answer minimum-weight perfect matching gives. Because the code is 1D, it can be found exactly by
dynamic programming over time (Viterbi) on a (shots, 2^d) cost tensor:

    C_t(x) = min_y [C_{t−1}(y) + w_data · |x ⊕ y|] + w_meas · K[σ_t, x]

The min over y factorises over qubits, so it is d cheap "flip one bit" relaxations.
At the end the data qubits are read out, which fixes x_R up to a global flip (the logical operator);
the decoder fails when the flipped pattern is cheaper than the true one.
"""

import math

import torch

from problems._common.progress import steps

from . import tensors
from .equation import accumulate, log_weight, majority_vote_fails, parity_check, per_round, syndrome


def relax_data_errors(cost: torch.Tensor, w: float, d: int) -> torch.Tensor:
    """min over y of cost(y) + w·|x ⊕ y|, one qubit at a time.   cost (S, 2^d) → (S, 2^d)."""
    s = cost.shape[0]
    for i in range(d):
        c = cost.view(s, 2 ** (d - 1 - i), 2, 2**i)  # axis 2 is bit i of the pattern index
        flipped = c.flip(2).add_(w)  # cost of reaching x by flipping qubit i
        cost = torch.minimum(c, flipped, out=flipped).view(s, -1)  # in place: keeps memory traffic low
    return cost


def decode_fails(
    x: torch.Tensor, sigma: torch.Tensor, mismatch: torch.Tensor, w_data: float, w_meas: float
) -> torch.Tensor:
    """Run the space–time decoder. x (S, R, d) true errors, sigma (S, R, d − 1) measured syndromes.

    Returns (S,) float: 1 if the decoder picks the wrong logical class, ½ on an exact tie.
    """
    n_shots, rounds, d = x.shape
    cost = torch.full((n_shots, 2**d), 1e6, device=x.device)
    cost[:, 0] = 0.0  # the code starts in |0_L⟩ with no errors
    sigma_idx = tensors.pack_bits(sigma)  # (S, R)
    for t in range(rounds):
        cost = relax_data_errors(cost, w_data, d)
        cost = cost + w_meas * mismatch[sigma_idx[:, t]]  # (S, 2^d)
    true_idx = tensors.pack_bits(x[:, -1])  # (S,) index of the true final error pattern
    flipped_idx = true_idx ^ (2**d - 1)  # same syndrome, opposite logical value
    c_true = cost.gather(1, true_idx[:, None])[:, 0]
    c_flip = cost.gather(1, flipped_idx[:, None])[:, 0]
    return (c_flip < c_true).float() + 0.5 * (c_flip == c_true).float()


def run_point(
    d: int, p: float, q: float, rounds: int, shots: int, gen: torch.Generator, device: torch.device, chunk: int = 8192
) -> tuple[float, torch.Tensor]:
    """Logical error probability after `rounds` rounds and one sample detection-event history (R, d − 1).

    q = 0 and rounds = 1 is the code-capacity model, decoded by a majority vote.
    """
    h = parity_check(d, device)
    mismatch = tensors.mismatch_table(d, device).float()
    w_data, w_meas = log_weight(p), log_weight(q)
    failures, sample = 0.0, None
    for start in range(0, shots, chunk):
        n = min(chunk, shots - start)
        e, m = tensors.sample_noise(n, rounds, d, p, q, gen, device)
        x = accumulate(e)  # (n, R, d)
        if q == 0.0:
            failures += majority_vote_fails(x[:, -1]).float().sum().item()
        else:
            sigma = syndrome(x, h, m)  # (n, R, d − 1)
            failures += decode_fails(x, sigma, mismatch, w_data, w_meas).sum().item()
        if sample is None:
            sample = syndrome(x[:1], h, m[:1])[0]
    return failures / shots, sample


def sweep(
    ds: list[int], ps: torch.Tensor, q_ratio: float, model: str, shots: int, seed: int, device: torch.device
) -> dict:
    """Logical error rates for every distance and physical error rate.

    Returns P_L (D, P), its binomial standard error, the per-round ε, and sample syndrome histories.
    """
    gen = torch.Generator().manual_seed(seed)
    p_logical = torch.zeros(len(ds), len(ps), dtype=torch.float64)
    samples = {}
    jobs = [(a, d, b, p) for a, d in enumerate(ds) for b, p in enumerate(ps.tolist())]
    for k in steps(len(jobs), "(d, p) points"):
        a, d, b, p = jobs[k]
        rounds = 1 if model == "code_capacity" else d
        q = 0.0 if model == "code_capacity" else min(0.5, q_ratio * p)
        p_logical[a, b], samples[(d, b)] = run_point(d, p, q, rounds, shots, gen, device)
    stderr = torch.sqrt(p_logical * (1 - p_logical) / shots).clamp(min=1.0 / shots)
    rounds = torch.tensor([1 if model == "code_capacity" else d for d in ds], dtype=torch.float64)
    eps = torch.stack([per_round(p_logical[a], int(rounds[a])) for a in range(len(ds))])
    return {"p_logical": p_logical, "stderr": stderr, "eps": eps, "samples": samples, "shots": shots}


def crossing(ps: torch.Tensor, eps_small: torch.Tensor, eps_large: torch.Tensor) -> float:
    """Physical error rate where the larger code stops helping (log–log interpolation), nan if none."""
    ok = (eps_small > 0) & (eps_large > 0)
    lp, diff = torch.log(ps[ok]), torch.log(eps_small[ok]) - torch.log(eps_large[ok])
    for k in range(len(diff) - 1):
        if diff[k] > 0 >= diff[k + 1]:
            f = diff[k] / (diff[k] - diff[k + 1])
            return math.exp(lp[k] + f * (lp[k + 1] - lp[k]))
    return float("nan")


def suppression(eps: torch.Tensor, p_logical: torch.Tensor, shots: int, min_failures: int = 10) -> torch.Tensor:
    """Λ_{d/d+2} = ε_d / ε_{d+2} for consecutive distances; nan where too few failures were seen.  (D − 1, P)"""
    enough = p_logical * shots >= min_failures
    lam = eps[:-1] / eps[1:]
    return torch.where(enough[:-1] & enough[1:], lam, torch.full_like(lam, float("nan")))
