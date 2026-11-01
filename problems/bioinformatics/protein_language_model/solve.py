"""Sample a protein family from the planted Potts model, then train the masked language model on it."""

import math

import torch

from problems._common.progress import steps

from . import equation, tensors


def gibbs_sample(h, J, colour, n_seq: int, n_sweeps: int, seed: int) -> torch.Tensor:
    """N independent Markov chains updated together; returns the final sequences (N, L).

    Checkerboard Gibbs: contacts only join sites of opposite colour, so all sites of one colour are independent given
    the rest and are redrawn at once from softmax_a E_{nia} (Gumbel-max trick: argmax_a E_{nia} − log(−log U)).
    """
    gen = torch.Generator(device=h.device).manual_seed(seed)
    L, q = h.shape
    s = torch.randint(q, (n_seq, L), generator=gen, device=h.device)
    for _ in steps(n_sweeps, "Gibbs sweeps"):
        for mask in (colour, ~colour):
            E = equation.local_field(tensors.one_hot(s, q), h, J)  # (N, L, q)
            u = torch.rand(E.shape, generator=gen, device=h.device).clamp_min(1e-12)
            s = torch.where(mask, torch.argmax(E - torch.log(-torch.log(u)), dim=-1), s)
    return s


def mask_tokens(seqs: torch.Tensor, rate: float, q: int, gen: torch.Generator):
    """Hide a random fraction of positions behind the [MASK] token q. Returns (tokens, mask) both (N, L)."""
    mask = torch.rand(seqs.shape, generator=gen, device=seqs.device) < rate
    return torch.where(mask, torch.full_like(seqs, q), seqs), mask.float()


def baselines(train: torch.Tensor, test: torch.Tensor, q: int) -> tuple[float, float]:
    """Cross-entropy (nats) on the test letters of two context-free predictors fitted on the training set:
    the unigram (one letter distribution for all sites) and the profile (one distribution per site, like a PSSM)."""
    x_train, x_test = tensors.one_hot(train, q), tensors.one_hot(test, q)
    unigram = (x_train.mean((0, 1)) * len(train) + 1) / (len(train) + q)  # (q,) add-one smoothing
    profile = (x_train.sum(0) + 1) / (len(train) + q)  # (L, q)
    ce_uni = -(x_test * torch.log(unigram)).sum(-1).mean()
    ce_prof = -(x_test * torch.log(profile)).sum(-1).mean()
    return ce_uni.item(), ce_prof.item()


def train(seqs, test, contacts, w, q, n_steps, batch, lr, rate, n_snapshots=60, seed=0):
    """Adam on the masked-LM loss. Every few steps: test loss (fixed masks), the attention contact map from the
    unmasked test sequences, and its precision at the top-K pairs (K = number of true contacts).

    Returns a dict of histories: train loss per step, and at the snapshot steps the test loss, precision,
    contact scores (S, L, L) and head-averaged first-layer attention (S, L, L).
    """
    gen = torch.Generator(device=seqs.device).manual_seed(seed)
    opt = torch.optim.Adam(tensors.parameters(w), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / 100) * 0.5 * (1 + math.cos(math.pi * t / n_steps))
    )
    test_tokens, test_mask = mask_tokens(test, rate, q, torch.Generator(device=seqs.device).manual_seed(seed + 1))
    k_top = int(torch.triu(contacts, 1).sum())
    at = sorted({round(k * (n_steps - 1) / max(1, n_snapshots - 1)) for k in range(n_snapshots)})
    hist = {"train": torch.zeros(n_steps), "at": at, "test": [], "precision": [], "scores": [], "attention": []}
    for t in steps(n_steps, "training steps"):
        idx = torch.randint(len(seqs), (batch,), generator=gen, device=seqs.device)
        tokens, mask = mask_tokens(seqs[idx], rate, q, gen)
        logits, _ = equation.forward(tokens, w)
        loss = equation.masked_lm_loss(logits, seqs[idx], mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist["train"][t] = loss.item()
        if t in at:
            with torch.no_grad():
                logits, _ = equation.forward(test_tokens, w)
                hist["test"].append(equation.masked_lm_loss(logits, test, test_mask).item())
                _, maps = equation.forward(test, w)  # unmasked sequences, as in Rao et al.
                score = equation.attention_contacts(maps)
                hist["precision"].append(equation.precision(equation.top_pairs(score, k_top), contacts))
                hist["scores"].append(score.cpu())
                hist["attention"].append(maps[:, 0].mean((0, 1)).cpu())
    hist["scores"], hist["attention"] = torch.stack(hist["scores"]), torch.stack(hist["attention"])
    return hist


def random_precision(contacts: torch.Tensor, min_sep: int = 3) -> float:
    """Expected precision of random pairs: true contacts / all pairs with j − i ≥ min_sep."""
    L = contacts.shape[0]
    i, j = torch.triu_indices(L, L, min_sep, device=contacts.device)
    return contacts[i, j].float().mean().item()
