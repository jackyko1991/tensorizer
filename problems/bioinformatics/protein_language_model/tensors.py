"""Build the concrete tensors: a 2D lattice fold and its contact map, the planted Potts model, the sequences
and the transformer weights.

Letters are integers 0 … q−1; the extra token q is [MASK]. The weights are plain tensors in a dict (no nn.Module)
so every product in equation.py is an explicit einsum.
"""

import math

import torch


def lattice_fold(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """A serpentine chain on a rows × cols square lattice: residue i sits at r_{ia}, shape (L, 2), L = rows·cols.

    Rows run left-to-right, then right-to-left, … so neighbouring rows pair up like the strands of a β-sheet.
    """
    coords = []
    for y in range(rows):
        xs = range(cols) if y % 2 == 0 else range(cols - 1, -1, -1)
        coords += [(x, y) for x in xs]
    return torch.tensor(coords, device=device)


def contact_map(coords: torch.Tensor, min_sep: int = 3) -> torch.Tensor:
    """True contacts (L, L) bool: lattice neighbours that are at least min_sep apart along the chain."""
    dist = (coords[:, None, :] - coords[None, :, :]).abs().sum(-1)  # city-block distance (L, L)
    idx = torch.arange(coords.shape[0], device=coords.device)
    return (dist == 1) & ((idx[:, None] - idx[None, :]).abs() >= min_sep)


def parity(coords: torch.Tensor) -> torch.Tensor:
    """Checkerboard colour (L,) bool. Lattice neighbours always have opposite colours."""
    return coords.sum(-1) % 2 == 0


def planted_potts(contacts: torch.Tensor, q: int, coupling: float, field: float, seed: int):
    """Potts parameters with couplings ONLY on the planted contacts.

    h_{ia} (L, q) ~ field · N(0, 1);  J_{iajb} (L, q, L, q): a random q×q block ~ coupling · N(0, 1) per contact,
    with J_{iajb} = J_{jbia} and zero everywhere else.
    """
    gen = torch.Generator().manual_seed(seed)
    L, dev = contacts.shape[0], contacts.device
    h = field * torch.randn(L, q, generator=gen)
    J = torch.zeros(L, q, L, q)
    for i, j in torch.nonzero(torch.triu(contacts.cpu(), 1)).tolist():
        block = coupling * torch.randn(q, q, generator=gen)
        J[i, :, j, :] = block
        J[j, :, i, :] = block.T
    return h.to(dev), J.to(dev)


def one_hot(seqs: torch.Tensor, q: int) -> torch.Tensor:
    """Integer sequences (N, L) → one-hot x_{nia} (N, L, q), float."""
    return torch.nn.functional.one_hot(seqs, q).float()


def init_weights(L: int, q: int, d: int, heads: int, layers: int, seed: int, device: torch.device) -> dict:
    """Transformer weights. c = model channel (d of them), h = head, k = channel within a head (d / heads).

    tok (q+1, d)  token embedding (last row = [MASK]);  pos (L, d)  learned position embedding
    per layer l:  Wq, Wk, Wv (H, d, d/H),  Wo (H, d/H, d),  W1 (d, 2d), W2 (2d, d)
    out (d, q)    read-out to letter logits
    """
    gen = torch.Generator().manual_seed(seed)
    dk = d // heads

    def rand(*shape, scale):
        return (scale * torch.randn(*shape, generator=gen)).to(device).requires_grad_(True)

    s = 1 / math.sqrt(d)  # keeps activations O(1) at the start
    w = {"tok": rand(q + 1, d, scale=0.5), "pos": rand(L, d, scale=0.5), "out": rand(d, q, scale=s)}
    w["layers"] = [
        {
            "Wq": rand(heads, d, dk, scale=s),
            "Wk": rand(heads, d, dk, scale=s),
            "Wv": rand(heads, d, dk, scale=s),
            "Wo": rand(heads, dk, d, scale=s),
            "W1": rand(d, 2 * d, scale=s),
            "W2": rand(2 * d, d, scale=s / math.sqrt(2)),
        }
        for _ in range(layers)
    ]
    return w


def parameters(w: dict) -> list[torch.Tensor]:
    """All trainable tensors, for the optimiser."""
    return [w["tok"], w["pos"], w["out"]] + [t for layer in w["layers"] for t in layer.values()]
