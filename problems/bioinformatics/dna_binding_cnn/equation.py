r"""Binding-site detection → tensor translation.

A DNA sequence of length ℓ is one-hot: x_{nlb} = 1 if sequence n has base b ∈ {A, C, G, T} at position l.
The network (one convolution layer, as in DeepBind):

    z_{nfp} = W_{fkb} x_{n,p+k,b} + c_f          filter f slid along the sequence   (k, b summed)
    m_{nf}  = max_p ReLU(z_{nfp})                 best match of filter f anywhere     (global max pooling)
    o_n     = v_f m_{nf} + v_0                    logit of "contains a binding site"  (f summed)
    L       = −⟨ y_n log σ(o_n) + (1 − y_n) log(1 − σ(o_n)) ⟩_n          binary cross-entropy

A filter W_{fkb} is a scoring matrix over a w-base window, the same object as a position weight matrix
(log-odds per base and position), so after training it can be read back as a sequence logo.
"""

import torch


def windows(x: torch.Tensor, width: int) -> torch.Tensor:
    """All length-w windows: X_{npkb} = x_{n,p+k,b}.   (N, ℓ, 4) → (N, P, w, 4) with P = ℓ − w + 1 (a view, no copy)."""
    return x.unfold(1, width, 1).transpose(2, 3)


def conv(x: torch.Tensor, W: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """z_{nfp} = W_{fkb} x_{n,p+k,b} + c_f.   x: (N, ℓ, 4), W: (F, w, 4) → (N, F, P);  k, b summed."""
    return torch.einsum("npkb,fkb->nfp", windows(x, W.shape[1]), W) + c[None, :, None]


def forward(x: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Logits o_n (N,): convolution → ReLU → max over positions → linear read-out (f summed)."""
    m = torch.relu(conv(x, params["W"], params["c"])).amax(dim=2)  # (N, F)
    return torch.einsum("nf,f->n", m, params["v"]) + params["v0"]


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits (numerically safe form of −y log σ(o) − (1−y) log(1−σ(o)))."""
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of sequences with the right call (logit > 0 ⇔ predicted binding site)."""
    return ((logits > 0).float() == y).float().mean().item()


def filter_pwms(x: torch.Tensor, W: torch.Tensor, c: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Read the filters back as position probability matrices p_{fkb} (F, w, 4), the way DeepBind/Basset do it.

    For every sequence n and filter f take the best-scoring window; keep it if its score exceeds threshold × the
    filter's top score; p_{fkb} = average of the kept one-hot windows (n summed), plus a small pseudocount.
    """
    X = windows(x, W.shape[1])  # (N, P, w, 4)
    z = torch.einsum("npkb,fkb->nfp", X, W) + c[None, :, None]  # (N, F, P)
    best, pos = z.max(dim=2)  # (N, F)
    keep = (best > threshold * best.amax(0, keepdim=True)) & (best > 0)  # (N, F)
    win = X[torch.arange(x.shape[0], device=x.device)[:, None], pos]  # (N, F, w, 4) best window per (n, f)
    counts = torch.einsum("nf,nfkb->fkb", keep.float(), win) + 0.25  # pseudocount
    return counts / counts.sum(-1, keepdim=True)


def information(pwm: torch.Tensor) -> torch.Tensor:
    """Information content per position, IC_k = 2 + Σ_b p_kb log₂ p_kb  (bits; 0 = uniform, 2 = one base).  (…, w)."""
    return 2 + (pwm * torch.log2(pwm.clamp_min(1e-9))).sum(-1)


def motif_similarity(pwm: torch.Tensor, motif: torch.Tensor) -> tuple[float, int]:
    """Best Pearson-type correlation between the planted motif (W, 4) and the filter PWM (w, 4), over all shifts.

    Both are centred (minus 1/4) so a uniform column contributes nothing; the filter is padded with uniform
    columns, so a filter that holds only part of the motif scores lower. Returns (r, shift).
    """
    W = motif.shape[0]
    pad = torch.full((W - 1, 4), 0.25, device=pwm.device)
    padded = torch.cat([pad, pwm, pad]) - 0.25  # (w + 2W − 2, 4)
    b = (motif - 0.25).flatten()
    scores = [
        (padded[s : s + W].flatten() @ b / (padded[s : s + W].norm() * b.norm() + 1e-12)).item()
        for s in range(padded.shape[0] - W + 1)
    ]
    best = max(range(len(scores)), key=scores.__getitem__)
    return scores[best], best - (W - 1)
