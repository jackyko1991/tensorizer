"""Build the PWM from training sites, scan a sequence on both strands and call hits with an empirical p-value."""

import torch

from . import equation
from .tensors import one_hot


def build_pwm(sites: torch.Tensor, pseudo: float, q: torch.Tensor) -> dict[str, torch.Tensor]:
    """sites (N, m) letter indices → counts, frequencies, log-odds (all (m, 4)) and information content (m,)."""
    C = equation.counts(one_hot(sites))
    P = equation.frequencies(C, pseudo)
    return {"C": C, "P": P, "W": equation.log_odds(P, q), "IC": equation.information_content(P, q)}


def scan_both_strands(seq: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Scores (…, 2, L-m+1) of every window of seq (…, L): row 0 the plus strand, row 1 the minus strand."""
    W_both = torch.stack([W, equation.reverse_complement_matrix(W)])  # (2, m, 4)
    return equation.scan(one_hot(seq), W_both)


def call_hits(p: torch.Tensor, p_max: float) -> list[tuple[int, int]]:
    """(start, strand ±1) of every window with p ≤ p_max, sorted by position.   p: (2, n)"""
    strand_row, start = torch.nonzero(p <= p_max, as_tuple=True)
    hits = [(int(n), 1 if int(s) == 0 else -1) for s, n in zip(strand_row, start)]
    return sorted(hits)


def match(hits: list[tuple[int, int]], starts: torch.Tensor, strands: torch.Tensor, tol: int = 0) -> torch.Tensor:
    """found_i = True if planted site i (start, strand) is among the hits.   (n_sites,) bool"""
    found = [any(abs(h - int(s0)) <= tol and hs == int(sd) for h, hs in hits) for s0, sd in zip(starts, strands)]
    return torch.tensor(found)


def score_threshold(null: torch.Tensor, p_max: float) -> float:
    """The score a window must beat to reach p ≤ p_max: the (K+1)-th highest null score, K = p_max (1 + #null) - 1."""
    null_sorted = null.flatten().sort(descending=True).values
    K = int(p_max * (1 + null_sorted.numel())) - 1
    return float(null_sorted[max(K, 0)]) if K >= 0 else float("inf")
