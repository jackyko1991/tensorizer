r"""Hidden Markov model → tensor translation.

A DNA sequence x_1 … x_T is emitted by a hidden state z_t ∈ {background, island} that follows a Markov chain:
P(z_1 = k) = π_k, P(z_{t+1} = j | z_t = i) = A_{ij}, P(x_t = α | z_t = k) = E_{kα}. CpG islands are GC-rich, so
the island state emits C and G more often. With x_{btα} one-hot (sequence b, position t, letter α):

    e_{btk} = x_{btα} E_{kα}                                       (α summed: emission probability of every state)
    α_{btj} = e_{btj} \sum_i α_{b,t-1,i} A_{ij}                    forward  (i summed)
    β_{bti} = \sum_j A_{ij} e_{b,t+1,j} β_{b,t+1,j}                backward (j summed)
    P(x_b) = \sum_k α_{bTk},       γ_{btk} = α_{btk} β_{btk} / P(x_b)        posterior P(z_t = k | x)

α underflows after a few hundred letters, so it is stored as log α. Each step is still the matrix product above:
pull out the largest entry m, multiply exp(log α − m) by A with einsum, and add m back (the log-sum-exp trick).

Viterbi replaces the sum by a max:  δ_{btj} = log e_{btj} + max_i (δ_{b,t-1,i} + log A_{ij}).
Baum–Welch re-estimates π, A, E from the expected counts γ and
ξ_{ij} = \sum_{bt} α_{bti} A_{ij} e_{b,t+1,j} β_{b,t+1,j} / P(x_b)   (b, t summed).
"""

import torch


def log_emissions(x: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    r"""log e_{btk} = x_{btα} log E_{kα}   (α summed).   (B, T, 4), (K, 4) → (B, T, K)"""
    return torch.einsum("bta,ka->btk", x, E.log())


def forward_step(log_alpha: torch.Tensor, A: torch.Tensor, log_e: torch.Tensor) -> torch.Tensor:
    r"""log α_{btj} = log e_{btj} + log \sum_i α_{b,t-1,i} A_{ij}   (i summed).   (B, K), (K, K), (B, K) → (B, K)"""
    m = log_alpha.amax(dim=1, keepdim=True)  # (B, 1) shift that keeps exp() in range
    return log_e + m + torch.einsum("bi,ij->bj", (log_alpha - m).exp(), A).log()


def backward_step(log_beta: torch.Tensor, A: torch.Tensor, log_e_next: torch.Tensor) -> torch.Tensor:
    r"""log β_{bti} = log \sum_j A_{ij} e_{b,t+1,j} β_{b,t+1,j}   (j summed).   (B, K) → (B, K)"""
    v = log_e_next + log_beta  # (B, K)
    m = v.amax(dim=1, keepdim=True)
    return m + torch.einsum("ij,bj->bi", A, (v - m).exp()).log()


def viterbi_step(delta: torch.Tensor, A: torch.Tensor, log_e: torch.Tensor):
    r"""δ_{btj} = log e_{btj} + max_i (δ_{b,t-1,i} + log A_{ij}); also returns the best previous state i.  (B, K)"""
    scores = delta[:, :, None] + A.log()[None]  # (B, K, K): come from i (dim 1), go to j (dim 2)
    return log_e + scores.amax(dim=1), scores.argmax(dim=1)


def posterior(log_alpha: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
    r"""γ_{btk} = α_{btk} β_{btk} / \sum_k α_{btk} β_{btk}.   (B, T, K) → (B, T, K), sums to 1 over k"""
    return torch.softmax(log_alpha + log_beta, dim=-1)


def expected_transitions(log_alpha, log_beta, log_e, A, log_px) -> torch.Tensor:
    r"""ξ_{ij} = \sum_{bt} α_{bti} A_{ij} e_{b,t+1,j} β_{b,t+1,j} / P(x_b)   (b, t summed).   → (K, K)

    Every factor is rescaled by its largest entry per (b, t) so the exponentials stay in range; the scales come back
    through the weight w_{bt}.
    """
    la = log_alpha[:, :-1]  # (B, T-1, K): α at t
    lv = (log_e + log_beta)[:, 1:]  # (B, T-1, K): e β at t+1
    s, r = la.amax(-1, keepdim=True), lv.amax(-1, keepdim=True)
    w = (s + r)[..., 0] - log_px[:, None]  # (B, T-1)
    return torch.einsum("bt,bti,ij,btj->ij", w.exp(), (la - s).exp(), A, (lv - r).exp())
