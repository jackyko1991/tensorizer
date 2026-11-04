r"""Associative memory → tensor translation.

Classic Hopfield network (Hopfield 1982). N binary neurons s_i = ±1 store P patterns ξ^μ through Hebb's rule,

    W_{ij} = \frac1N \sum_\mu \xi^\mu_i \xi^\mu_j\ (i \ne j),\qquad W_{ii} = 0,

and recall by flipping one neuron at a time to the sign of its input field,

    s_i \leftarrow \operatorname{sign}\Big(\sum_j W_{ij} s_j\Big),\qquad E(s) = -\tfrac12\,s_i W_{ij} s_j.

With symmetric W and zero diagonal each single-neuron flip can only lower E, so the network slides into a local
minimum. For random patterns the stored ones stay (nearly) stable up to P ≈ 0.138 N (Amit, Gutfreund and
Sompolinsky 1985); beyond that, or with correlated patterns, the minima become mixtures and spurious states.

Modern continuous Hopfield network (Ramsauer et al. 2020, "Hopfield Networks is All You Need"). Keep the patterns
X = (ξ^1 … ξ^P) themselves instead of W and update a continuous state ξ by

    \xi \leftarrow X\,\operatorname{softmax}(\beta\,X^\top \xi),\qquad
    E(\xi) = -\beta^{-1}\log\sum_\mu e^{\beta\,\xi^{\mu\top}\xi} + \tfrac12\,\xi^\top\xi
             + \beta^{-1}\log P + \tfrac12 M^2,

the attention mechanism of transformers. Each update lowers E; for well separated patterns and large β one
update lands on the closest pattern, and the number of storable patterns grows exponentially with N.
"""

import torch


def hebbian(patterns: torch.Tensor, used: torch.Tensor) -> torch.Tensor:
    r"""W_{fij} = (1/N) Σ_μ m_{fμ} ξ^μ_i ξ^μ_j with W_{fii} = 0   (μ summed; m_{fμ} = 1 if pattern μ is stored in
    network f).   (P, N), (F, P) → (F, N, N)"""
    n = patterns.shape[-1]
    w = torch.einsum("fp,pi,pj->fij", used, patterns, patterns) / n
    return w * (1 - torch.eye(n, device=w.device))


def field(s: torch.Tensor, w_row: torch.Tensor) -> torch.Tensor:
    r"""h_i = Σ_j W_{ij} s_j for one neuron i in every network and query (j summed).   (F, Q, N), (F, N) → (F, Q)"""
    return torch.einsum("fqj,fj->fq", s, w_row)


def energy(s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r"""E = −½ s_i W_{ij} s_j   (i, j summed).   (F, Q, N), (F, N, N) → (F, Q)"""
    return -0.5 * torch.einsum("fqi,fij,fqj->fq", s, w, s)


def async_sweep(s: torch.Tensor, w: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """One asynchronous sweep: neurons in the given order, each set to the sign of its field (kept if the field is
    0). All networks and queries update in parallel; only the neurons go one by one.   (F, Q, N) → (F, Q, N)"""
    s = s.clone()
    for i in order.tolist():
        h = field(s, w[:, i])  # (F, Q)
        s[..., i] = torch.where(h == 0, s[..., i], torch.sign(h))
    return s


def modern_update(xi: torch.Tensor, patterns: torch.Tensor, beta: float) -> torch.Tensor:
    r"""ξ ← X softmax(β Xᵀ ξ): attention of the query over the stored patterns.   (..., N), (P, N) → (..., N)"""
    p = torch.softmax(beta * torch.einsum("...n,pn->...p", xi, patterns), dim=-1)  # (..., P) weights of the patterns
    return torch.einsum("...p,pn->...n", p, patterns)


def modern_energy(xi: torch.Tensor, patterns: torch.Tensor, beta: float) -> torch.Tensor:
    r"""E = −β⁻¹ lse(β Xᵀξ) + ½ ξᵀξ + β⁻¹ log P + ½ M², M = max_μ |ξ^μ|.   (..., N), (P, N) → (...)"""
    lse = torch.logsumexp(beta * torch.einsum("...n,pn->...p", xi, patterns), dim=-1)
    m2 = patterns.pow(2).sum(-1).max()
    n_pat = torch.tensor(float(patterns.shape[0]))
    return -lse / beta + 0.5 * xi.pow(2).sum(-1) + torch.log(n_pat) / beta + 0.5 * m2


def overlap(s: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r"""m = (1/N) Σ_i s_i ξ_i: 1 for perfect recall, about 0 for an unrelated state.   (..., N) → (...)"""
    return (s * target).sum(-1) / s.shape[-1]
