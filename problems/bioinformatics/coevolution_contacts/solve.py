"""Sample the alignment from the planted Potts model, then infer contacts with mean-field DCA or pseudo-likelihood."""

import torch

from problems._common.progress import steps

from .equation import (
    apc,
    covariance,
    frobenius_scores,
    local_field,
    mean_field_couplings,
    pair_frequencies,
    pseudo_log_likelihood,
    single_frequencies,
    with_pseudocount,
)
from .tensors import one_hot


def gibbs_sample(h, J, colour, n_seq: int, n_sweeps: int, seed: int) -> torch.Tensor:
    """N independent Markov chains (one per sequence), updated together. Returns the final sequences (N, L).

    Checkerboard Gibbs sampling: contacts only join sites of opposite colour, so all sites of one colour are
    conditionally independent given the others and can be resampled AT ONCE from softmax_a E_{nia}.
    The Gumbel-max trick draws from a softmax: argmax_a (E_{nia} + g_{nia}), g = −log(−log U).
    """
    gen = torch.Generator(device=h.device).manual_seed(seed)
    L, q = h.shape
    s = torch.randint(q, (n_seq, L), generator=gen, device=h.device)  # random start
    for _ in steps(n_sweeps, "Gibbs sweeps"):
        for mask in (colour, ~colour):
            E = local_field(one_hot(s, q), h, J)  # (N, L, q)
            u = torch.rand(E.shape, generator=gen, device=h.device).clamp_min(1e-12)
            new = torch.argmax(E - torch.log(-torch.log(u)), dim=-1)  # (N, L) sample from softmax_a E
            s = torch.where(mask, new, s)  # only this colour changes
    return s


def covariance_scores(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Baseline: APC-corrected Frobenius norm of the raw covariance C_ij (direct + indirect correlations)."""
    f1, f2 = with_pseudocount(single_frequencies(x), pair_frequencies(x), lam)
    return apc(frobenius_scores(covariance(f1, f2)))


def mfdca_scores(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Mean-field DCA: frequencies → covariance → J = −C⁻¹ → Frobenius norm → APC.   (L, L)."""
    f1, f2 = with_pseudocount(single_frequencies(x), pair_frequencies(x), lam)
    return apc(frobenius_scores(mean_field_couplings(covariance(f1, f2))))


def train_plm(x: torch.Tensor, n_iter: int, lr: float, reg: float, n_snapshots: int = 40):
    """Pseudo-likelihood maximisation with autograd and Adam.

    loss = −(1/N) Σ_{n,i} log P(s_ni | s_n,rest) + reg · Σ J²   (the L2 penalty stands in for a prior).
    J is stored unconstrained and symmetrised, with the diagonal blocks masked out, before each use.
    Returns the final scores (L, L), the snapshot iterations and score snapshots (S, L, L), and the loss history.
    """
    N, L, q = x.shape
    h = torch.zeros(L, q, device=x.device, requires_grad=True)
    W = torch.zeros(L, q, L, q, device=x.device, requires_grad=True)
    off_diag = (1 - torch.eye(L, device=x.device))[:, None, :, None]  # (L, 1, L, 1)
    opt = torch.optim.Adam([h, W], lr=lr)

    def couplings() -> torch.Tensor:
        return 0.5 * (W + W.permute(2, 3, 0, 1)) * off_diag  # J_iajb = J_jbia, J_iaib = 0

    snap_iters = sorted({round(s * (n_iter - 1) / max(1, n_snapshots - 1)) for s in range(n_snapshots)})
    snaps, losses = [], torch.zeros(n_iter)
    for it in steps(n_iter, "pseudo-likelihood"):
        J = couplings()
        loss = -pseudo_log_likelihood(x, h, J) + reg * (J**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses[it] = loss.item()
        if it in snap_iters:
            with torch.no_grad():
                snaps.append(apc(frobenius_scores(couplings())).cpu())
    return snaps[-1], snap_iters, torch.stack(snaps), losses
