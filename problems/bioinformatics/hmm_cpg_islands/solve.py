"""Forward–backward, Viterbi and Baum–Welch: loops over the positions t, every sequence of the batch at once."""

import torch

from problems._common.progress import steps

from .equation import (
    backward_step,
    expected_transitions,
    forward_step,
    log_emissions,
    posterior,
    viterbi_step,
)


def forward(log_e: torch.Tensor, pi: torch.Tensor, A: torch.Tensor):
    """log α of shape (B, T, K) and the log-likelihood log P(x_b) of every sequence (B,)."""
    log_alpha = torch.empty_like(log_e)
    log_alpha[:, 0] = pi.log() + log_e[:, 0]
    for t in range(1, log_e.shape[1]):
        log_alpha[:, t] = forward_step(log_alpha[:, t - 1], A, log_e[:, t])
    return log_alpha, torch.logsumexp(log_alpha[:, -1], dim=-1)


def backward(log_e: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """log β of shape (B, T, K), with β_T = 1."""
    log_beta = torch.zeros_like(log_e)
    for t in range(log_e.shape[1] - 2, -1, -1):
        log_beta[:, t] = backward_step(log_beta[:, t + 1], A, log_e[:, t + 1])
    return log_beta


def viterbi(log_e: torch.Tensor, pi: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Most probable state path of every sequence: (B, T) state indices."""
    B, T, K = log_e.shape
    delta = pi.log() + log_e[:, 0]
    prev = torch.zeros(B, T, K, dtype=torch.long, device=log_e.device)  # back-pointers
    for t in range(1, T):
        delta, prev[:, t] = viterbi_step(delta, A, log_e[:, t])
    path = torch.empty(B, T, dtype=torch.long, device=log_e.device)
    path[:, -1] = delta.argmax(dim=1)
    for t in range(T - 1, 0, -1):  # follow the back-pointers from the end
        path[:, t - 1] = prev[:, t].gather(1, path[:, t : t + 1])[:, 0]
    return path


def e_step(x: torch.Tensor, pi, A, E):
    """Posteriors γ (B, T, K), expected transitions ξ (K, K) and log-likelihoods (B,) under one model."""
    log_e = log_emissions(x, E)
    log_alpha, log_px = forward(log_e, pi, A)
    log_beta = backward(log_e, A)
    gamma = posterior(log_alpha, log_beta)
    xi = expected_transitions(log_alpha, log_beta, log_e, A, log_px)
    return gamma, xi, log_px


def m_step(x: torch.Tensor, gamma: torch.Tensor, xi: torch.Tensor):
    """New π, A, E from expected counts: π_k = <γ_{b1k}>, A ∝ ξ, E_{kα} ∝ γ_{btk} x_{btα} (b, t summed)."""
    pi = gamma[:, 0].mean(dim=0)
    A = xi / xi.sum(dim=1, keepdim=True)
    counts = torch.einsum("btk,bta->ka", gamma, x)
    return pi, A, counts / counts.sum(dim=1, keepdim=True)


def baum_welch(x: torch.Tensor, pi, A, E, n_iter: int):
    """EM for n_iter rounds. Returns the list of models (n_iter + 1 of them, the first is the starting guess),
    the total log-likelihood before each update, and the posterior P(island) of sequence 0 under every model."""
    models, log_lik, post0 = [(pi, A, E)], [], []
    for _ in steps(n_iter, "Baum–Welch"):
        gamma, xi, log_px = e_step(x, pi, A, E)
        log_lik.append(float(log_px.sum()))
        post0.append(gamma[0, :, 1])
        pi, A, E = m_step(x, gamma, xi)
        models.append((pi, A, E))
    gamma, _, log_px = e_step(x, pi, A, E)
    log_lik.append(float(log_px.sum()))
    post0.append(gamma[0, :, 1])
    return models, log_lik, torch.stack(post0)
