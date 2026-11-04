r"""Parity and M stacked transformers → tensor translation, every product an einsum over the model axis m.

Indices: m = model (format × length × seed), n = sequence, i = query position, j = key position, c = channel,
h = head, k = channel within a head, v = vocabulary token.

    s_k = b_1 \oplus \dots \oplus b_k = \big(\textstyle\sum_{l \le k} b_l\big) \bmod 2      running parity (the task)
    X_{mnic} = E_{m,t_{mni},c} + P_{mic}                                    token + position embedding
    Q_{mnhik} = \hat X_{mnic} W^Q_{mhck} + b^Q_{mhk}                        (c summed; K and V alike)
    A_{mnhij} = \mathrm{softmax}_j\big(Q_{mnhik}K_{mnhjk}/\sqrt{d_k} + M_{ij}\big)  (k summed)  M_{ij} = −∞ for j > i
    X_{mnic} \mathrel{+}= A_{mnhij} V_{mnhjk} W^O_{mhkc} + b^O_{mc}         (h, j, k summed)
    X \mathrel{+}= \mathrm{GELU}(\hat X W_1 + c_1) W_2 + c_2                MLP, \hat X = LayerNorm(X)
    \mathcal L_m = \sum_{n,i} \mu_{mni}\,(-\log \mathrm{softmax}_v z_{mniv}[y_{mni}]) / \sum_{n,i} \mu_{mni}

Each model m only ever multiplies its own weights, so summing the M losses gives every model its own gradient:
M independent trainings in one set of tensor ops.
"""

import math

import torch


def running_parity(bits: torch.Tensor) -> torch.Tensor:
    """s_k = b_1 ⊕ … ⊕ b_k for every prefix: the cumulative sum mod 2.  bits (..., n) → s (..., n)."""
    return torch.cumsum(bits, dim=-1) % 2


def layer_norm(X: torch.Tensor, g: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """(X − mean) / std over the channels of every token, then a learned scale and shift.  X (M, N, T, d), g (M, d)."""
    mu = X.mean(-1, keepdim=True)
    var = (X - mu).pow(2).mean(-1, keepdim=True)  # biased variance over the channels
    return g[:, None, None] * (X - mu) / torch.sqrt(var + eps) + b[:, None, None]


def causal_mask(t: int, device: torch.device) -> torch.Tensor:
    """M_{ij} = 0 for j ≤ i (past and present) and −∞ for j > i (future).  (T, T)"""
    future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)
    return torch.zeros(t, t, device=device).masked_fill(future, float("-inf"))


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal scaled dot-product attention over leading axes.  Q, K, V (..., T, d_k) → out (..., T, d_k), A (..., T, T).

    S_{ij} = Q_{ik} K_{jk} / √d_k + M_{ij} (k summed); A = softmax_j S; out_{ik} = A_{ij} V_{jk} (j summed).
    """
    t = Q.shape[-2]
    S = torch.einsum("...ik,...jk->...ij", Q, K) / math.sqrt(Q.shape[-1])
    A = torch.softmax(S + causal_mask(t, Q.device).to(S.dtype), dim=-1)
    return torch.einsum("...ij,...jk->...ik", A, V), A


def self_attention(X: torch.Tensor, p: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-head causal self-attention of normalised tokens X (M, N, T, d) → update (M, N, T, d), A (M, N, H, T, T)."""
    Q = torch.einsum("mnic,mhck->mnhik", X, p["Wq"]) + p["bq"][:, None, :, None, :]  # c summed, per model m
    K = torch.einsum("mnjc,mhck->mnhjk", X, p["Wk"]) + p["bk"][:, None, :, None, :]
    V = torch.einsum("mnjc,mhck->mnhjk", X, p["Wv"]) + p["bv"][:, None, :, None, :]
    mix, A = attention(Q, K, V)
    return torch.einsum("mnhik,mhkc->mnic", mix, p["Wo"]) + p["bo"][:, None, None], A  # h, k summed


def mlp(X: torch.Tensor, p: dict) -> torch.Tensor:
    """Position-wise d → 4d → d network with a GELU, separate weights for every model m."""
    hidden = torch.nn.functional.gelu(torch.einsum("mnic,mcf->mnif", X, p["W1"]) + p["c1"][:, None, None])
    return torch.einsum("mnif,mfc->mnic", hidden, p["W2"]) + p["c2"][:, None, None]


def forward(tokens: torch.Tensor, w: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """tokens (M, N, T) → next-token logits z (M, N, T, V) and attention maps (M, N, layers, H, T, T)."""
    m, _, t = tokens.shape
    model = torch.arange(m, device=tokens.device)[:, None, None]
    X = w["tok"][model, tokens] + w["pos"][:, None, :t]  # (M, N, T, d): model m looks up its own embedding
    maps = []
    for p in w["layers"]:  # pre-norm blocks with residual connections
        update, A = self_attention(layer_norm(X, p["g1"], p["b1"]), p)
        X = X + update
        X = X + mlp(layer_norm(X, p["g2"], p["b2"]), p)
        maps.append(A)
    Xf = layer_norm(X, w["gf"], w["bf"])
    logits = torch.einsum("mnic,mcv->mniv", Xf, w["head"]) + w["bias"][:, None, None]
    return logits, torch.stack(maps, dim=2)


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-model mean −log p of the trained targets (nats).  logits (M, N, T, V), targets, mask (M, N, T) → (M,)."""
    nll = -torch.gather(torch.log_softmax(logits, -1), -1, targets[..., None])[..., 0]
    return (nll * mask).sum((1, 2)) / mask.sum((1, 2))


def bit_choice(logits: torch.Tensor) -> torch.Tensor:
    """The model's guess when a bit is due: the more likely of the tokens 0 and 1.  (..., V) → (...,) 0/1."""
    return logits[..., :2].argmax(-1)
