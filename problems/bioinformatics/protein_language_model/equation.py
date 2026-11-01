r"""Masked language model → tensor translation.

Data: sequences s_{ni} (n = sequence, i = position) drawn from a Potts model with couplings only on contacts,

    P(s) \propto \exp\big(h_i(s_i) + \tfrac12 J_{ij}(s_i, s_j)\big),   local field  E_{nia} = h_{ia} + J_{iajb} x_{njb}.

Model: embeddings X_{nic} = T_{t_{ni} c} + P_{ic}, then per layer (c = channel, h = head, k = channel in a head)

    Q_{nhik} = X_{nic} W^Q_{hck},   K_{nhjk} = X_{njc} W^K_{hck},   V_{nhjk} = X_{njc} W^V_{hck}     (c summed)
    A_{nhij} = \mathrm{softmax}_j\big(Q_{nhik} K_{nhjk} / \sqrt{d_k}\big)                              (k summed)
    X_{nic} \leftarrow X_{nic} + A_{nhij} V_{nhjk} W^O_{hkc}                                            (h, j, k summed)

and a small MLP; logits z_{nia} = X_{nic} W_{ca}. Training minimises the masked-token cross-entropy

    \mathcal L = -\frac{1}{|M|}\sum_{(n,i)\in M} \log \mathrm{softmax}_a(z_{nia})[s_{ni}].

Contacts from attention (Rao et al. 2021): average A over sequences, symmetrise, remove the background (APC):

    \bar A_{lhij} = \langle A_{nlhij}\rangle_n,  C_{ij} = \tfrac12(\bar A + \bar A^\top)_{ij},
    C^{APC}_{ij} = C_{ij} - C_{i\cdot}C_{\cdot j}/C_{\cdot\cdot}.
"""

import math

import torch

# --- the Potts model that generates the "protein family" ---


def local_field(x: torch.Tensor, h: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """E_{nia} = h_{ia} + J_{iajb} x_{njb}: log-weight of letter a at site i given the others.  j, b summed."""
    return h + torch.einsum("iajb,njb->nia", J, x)


# --- the transformer ---


def layer_norm(X: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Normalise every token vector to zero mean and unit variance over its channels c (no learned scale)."""
    mu = X.mean(-1, keepdim=True)
    return (X - mu) / torch.sqrt(X.var(-1, unbiased=False, keepdim=True) + eps)


def embed(tokens: torch.Tensor, w: dict) -> torch.Tensor:
    """X_{nic} = T_{t_{ni} c} + P_{ic}: token plus position embedding.  tokens (N, L) → (N, L, d)."""
    return w["tok"][tokens] + w["pos"]


def attention_weights(X: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor) -> torch.Tensor:
    r"""A_{nhij} = softmax_j(Q_{nhik} K_{nhjk} / \sqrt{d_k}).   X (N, L, d) → (N, H, L, L); each row sums to 1."""
    Q = torch.einsum("nic,hck->nhik", X, Wq)
    K = torch.einsum("njc,hck->nhjk", X, Wk)
    scores = torch.einsum("nhik,nhjk->nhij", Q, K) / math.sqrt(Q.shape[-1])
    return torch.softmax(scores, dim=-1)


def attend(X: torch.Tensor, A: torch.Tensor, Wv: torch.Tensor, Wo: torch.Tensor) -> torch.Tensor:
    """Δ_{nic} = A_{nhij} V_{nhjk} W^O_{hkc}: each position collects a weighted mix of the others.  h, j, k summed."""
    V = torch.einsum("njc,hck->nhjk", X, Wv)
    mix = torch.einsum("nhij,nhjk->nhik", A, V)  # j summed: the weighted mix of values
    return torch.einsum("nhik,hkc->nic", mix, Wo)  # h, k summed: heads merged back into d channels


def block(X: torch.Tensor, p: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """One pre-norm transformer layer: X + attention(LN X), then X + MLP(LN X). Returns (X, A)."""
    Xn = layer_norm(X)
    A = attention_weights(Xn, p["Wq"], p["Wk"])
    X = X + attend(Xn, A, p["Wv"], p["Wo"])
    X = X + torch.relu(layer_norm(X) @ p["W1"]) @ p["W2"]
    return X, A


def forward(tokens: torch.Tensor, w: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits z_{nia} (N, L, q) and the attention of every layer (N, layers, H, L, L)."""
    X = embed(tokens, w)
    maps = []
    for p in w["layers"]:
        X, A = block(X, p)
        maps.append(A)
    return layer_norm(X) @ w["out"], torch.stack(maps, dim=1)


def masked_lm_loss(logits: torch.Tensor, seqs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy (nats) of the true letter at the masked positions only."""
    logp = torch.log_softmax(logits, dim=-1)  # (N, L, q)
    true = torch.gather(logp, -1, seqs[..., None])[..., 0]  # (N, L) log-probability of the true letter
    return -(true * mask).sum() / mask.sum()


# --- contacts from attention ---


def apc(F: torch.Tensor) -> torch.Tensor:
    """Average-product correction F_ij − F_i· F_·j / F_·· (works on (..., L, L) stacks)."""
    row = F.mean(-1, keepdim=True)
    col = F.mean(-2, keepdim=True)
    return F - row * col / F.mean((-2, -1), keepdim=True)


def attention_contacts(maps: torch.Tensor) -> torch.Tensor:
    """Unsupervised contact score from attention: (N, layers, H, L, L) → (L, L).

    Average over sequences, symmetrise each head, zero the diagonal, APC per head, then average all heads.
    """
    A = maps.mean(0)  # (layers, H, L, L)
    C = 0.5 * (A + A.transpose(-1, -2))
    C = C * (1 - torch.eye(C.shape[-1], device=C.device))
    return apc(C).mean((0, 1))


def top_pairs(score: torch.Tensor, n_top: int, min_sep: int = 3) -> torch.Tensor:
    """The n_top highest-scoring pairs (i < j, j − i ≥ min_sep) as an (n_top, 2) index tensor."""
    L = score.shape[0]
    i, j = torch.triu_indices(L, L, min_sep, device=score.device)
    order = torch.argsort(score[i, j], descending=True)[:n_top]
    return torch.stack([i[order], j[order]], dim=1)


def precision(pairs: torch.Tensor, contacts: torch.Tensor) -> float:
    """Fraction of predicted pairs that are true contacts."""
    return contacts[pairs[:, 0], pairs[:, 1]].float().mean().item()
