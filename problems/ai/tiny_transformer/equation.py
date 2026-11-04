r"""GPT → tensor translation: a decoder-only transformer written with einsum.

Indices: n = sequence in the batch, i = query position, j = key position, c = model channel, h = head,
k = channel within a head, v = vocabulary letter.

    X_{nic} = T_{t_{ni} c} + P_{ic}                                      token + position embedding
    Q_{nhik} = \hat X_{nic} W^Q_{hck} + b^Q_{hk}                         (c summed; K and V alike)
    S_{nhij} = Q_{nhik} K_{nhjk} / \sqrt{d_k} + M_{ij}                   (k summed)  M_{ij} = 0 if j ≤ i, −∞ if j > i
    A_{nhij} = \mathrm{softmax}_j S_{nhij}                               each row sums to 1, zero above the diagonal
    X_{nic} \mathrel{+}= A_{nhij} V_{nhjk} W^O_{hkc} + b^O_c             (h, j, k summed)
    X \mathrel{+}= \mathrm{GELU}(\hat X W_1 + c_1) W_2 + c_2             MLP, \hat X = LayerNorm(X)
    z_{niv} = \hat X_{nic} W_{cv} + b_v,   \mathcal L = -\langle \log \mathrm{softmax}_v z_{niv}[t_{n,i+1}] \rangle

The causal mask M makes position i blind to every later character, so one forward pass trains all T next-letter
predictions of a window at once.
"""

import math

import torch


def layer_norm(X: torch.Tensor, g: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """(X − mean) / std over the channels c of every token, then a learned scale g_c and shift b_c."""
    mu = X.mean(-1, keepdim=True)
    var = X.var(-1, unbiased=False, keepdim=True)
    return g * (X - mu) / torch.sqrt(var + eps) + b


def causal_mask(t: int, device: torch.device) -> torch.Tensor:
    """M_{ij} = 0 for j ≤ i (the past and present) and −∞ for j > i (the future).  (T, T)"""
    future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)
    return torch.zeros(t, t, device=device).masked_fill(future, float("-inf"))


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal scaled dot-product attention.  Q, K, V (N, H, T, d_k) → output (N, H, T, d_k) and A (N, H, T, T).

    S_{nhij} = Q_{nhik} K_{nhjk} / √d_k + M_{ij} (k summed); A = softmax_j S; out_{nhik} = A_{nhij} V_{nhjk} (j summed).
    """
    t = Q.shape[-2]
    S = torch.einsum("nhik,nhjk->nhij", Q, K) / math.sqrt(Q.shape[-1])
    A = torch.softmax(S + causal_mask(t, Q.device).to(S.dtype), dim=-1)
    return torch.einsum("nhij,nhjk->nhik", A, V), A


def self_attention(X: torch.Tensor, p: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-head causal self-attention of normalised tokens X (N, T, d) → update (N, T, d) and A (N, H, T, T)."""
    Q = torch.einsum("nic,hck->nhik", X, p["Wq"]) + p["bq"][:, None, :]  # c summed
    K = torch.einsum("njc,hck->nhjk", X, p["Wk"]) + p["bk"][:, None, :]
    V = torch.einsum("njc,hck->nhjk", X, p["Wv"]) + p["bv"][:, None, :]
    mix, A = attention(Q, K, V)
    return torch.einsum("nhik,hkc->nic", mix, p["Wo"]) + p["bo"], A  # h, k summed: heads merged into d channels


def mlp(X: torch.Tensor, p: dict) -> torch.Tensor:
    """Position-wise d → 4d → d network with a GELU in the middle."""
    return torch.nn.functional.gelu(X @ p["W1"] + p["c1"]) @ p["W2"] + p["c2"]


def block(X: torch.Tensor, p: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """One pre-norm transformer block with residual connections: X + Attn(LN X), then X + MLP(LN X)."""
    update, A = self_attention(layer_norm(X, p["g1"], p["b1"]), p)
    X = X + update
    X = X + mlp(layer_norm(X, p["g2"], p["b2"]), p)
    return X, A


def forward(tokens: torch.Tensor, w: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-character logits z (N, T, V) and every layer's attention (N, layers, H, T, T)."""
    t = tokens.shape[1]
    X = w["tok"][tokens] + w["pos"][:t]  # (N, T, d)
    maps = []
    for p in w["layers"]:
        X, A = block(X, p)
        maps.append(A)
    return layer_norm(X, w["gf"], w["bf"]) @ w["head"] + w["bias"], torch.stack(maps, dim=1)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean −log p(next character) in nats.  logits (N, T, V), targets (N, T)."""
    logp = torch.log_softmax(logits, dim=-1)
    return -torch.gather(logp, -1, targets[..., None]).mean()
