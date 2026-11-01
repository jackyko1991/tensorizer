"""Build the concrete tensors: a lattice fold, its contact map, the planted Potts model and one-hot sequences."""

import torch


def lattice_fold(dims: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """A compact chain on a cubic lattice: a serpentine path that visits every site of an X×Y×Z box once.

    Rows run back and forth in x, rows stack in y, layers stack in z (each layer reversed so the path stays
    connected). Returns the residue coordinates r_{ia}, shape (L, 3) with L = X·Y·Z.
    """
    X, Y, Z = dims
    coords = []
    for z in range(Z):
        ys = range(Y) if z % 2 == 0 else range(Y - 1, -1, -1)
        for row, y in enumerate(ys):
            xs = range(X) if (row + z * Y) % 2 == 0 else range(X - 1, -1, -1)
            coords += [(x, y, z) for x in xs]
    return torch.tensor(coords, device=device)


def contact_map(coords: torch.Tensor, min_sep: int = 3) -> torch.Tensor:
    """True contacts (L, L) bool: lattice neighbours (city-block distance 1) at least min_sep apart in sequence."""
    L = coords.shape[0]
    dist = (coords[:, None, :] - coords[None, :, :]).abs().sum(-1)  # (L, L) city-block distance
    idx = torch.arange(L, device=coords.device)
    far_in_sequence = (idx[:, None] - idx[None, :]).abs() >= min_sep
    return (dist == 1) & far_in_sequence


def parity(coords: torch.Tensor) -> torch.Tensor:
    """Checkerboard colour (L,) bool: x+y+z even. Lattice neighbours always have opposite colours."""
    return coords.sum(-1) % 2 == 0


def planted_potts(contacts: torch.Tensor, q: int, coupling: float, field: float, seed: int):
    """Random Potts parameters with couplings ONLY on the true contacts.

    h_{ia}:     (L, q)        site fields ~ field · N(0, 1)            (some residues more conserved than others)
    J_{iajb}:   (L, q, L, q)  for each contact a random q×q block ~ coupling · N(0, 1), J_{iajb} = J_{jbia}
    """
    gen = torch.Generator().manual_seed(seed)
    L, dev = contacts.shape[0], contacts.device
    h = field * torch.randn(L, q, generator=gen)
    J = torch.zeros(L, q, L, q)
    for i, j in torch.nonzero(torch.triu(contacts.cpu(), 1)).tolist():
        block = coupling * torch.randn(q, q, generator=gen)
        J[i, :, j, :] = block
        J[j, :, i, :] = block.T  # symmetric: the energy of pair (i, j) is the same seen from either side
    return h.to(dev), J.to(dev)


def one_hot(seqs: torch.Tensor, q: int) -> torch.Tensor:
    """Sequences (N, L) of integer states 0…q-1 → one-hot x_{nia}, shape (N, L, q), float."""
    return torch.nn.functional.one_hot(seqs, q).float()
