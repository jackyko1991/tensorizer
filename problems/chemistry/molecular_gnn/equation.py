r"""Physics → tensor translation.

THE TARGET (a known rule, so the data are exact): the Hückel π-electron energy. Each sp² carbon contributes one
p orbital and one π electron; the Hückel Hamiltonian is H = α 1 + β A (A = adjacency of the carbon skeleton,
β < 0). With eigenvalues λ_1 ≥ λ_2 ≥ … of A and the N electrons filling the lowest-energy (highest-λ) orbitals
two at a time:

    E_π = N α + β y,      y = Σ_k o_k λ_k      (o_k = 2, 2, …, then 1 for an odd electron, then 0)

We learn y, the π binding energy in units of |β|, from the graph alone.

THE MODEL: a message-passing neural network (MPNN). Indices: m = molecule, i/j = atom, c/d = channel.

    embed      h⁰_{mic}    = σ(x_{mif} W_{fc} + b_c) · mask_{mi}                      f summed
    message    s_{mic}     = A_{mij} h_{mjc}                                           j summed: neighbours only
    update     h'_{mid}    = σ(h_{mic} U_{cd} + s_{mic} V_{cd} + c_d) · mask_{mi}       c summed
    readout    y_m         = Σ_i mask_{mi} e_{mi},    e_{mi} = Q2_d σ(h_{mic} Q1_{cd} + q1_d) + q2

Every weight acts on ONE atom at a time and the only sum over atoms is through A_{mij} or the final Σ_i, so
relabelling the atoms (A → ΠAΠᵀ, x → Πx) leaves y unchanged, and the same weights work for any molecule size.
"""

import torch

# --- the target rule ---


def huckel_energy(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""y_m = Σ_k o_k λ_{mk}: π binding energy in units of |β| for every padded molecule.   (M, N, N) → (M,).

    Padded atoms get a huge negative diagonal so their eigenvalues sink to the bottom and stay unoccupied.
    """
    A = A.double()
    pad = torch.diag_embed(-100.0 * (1 - mask.double()))
    lam = torch.linalg.eigvalsh(A + pad).flip(-1)  # (M, N) eigenvalues, largest first
    n_electrons = mask.sum(-1, keepdim=True).double()  # one π electron per carbon
    k = torch.arange(A.shape[-1], device=A.device, dtype=torch.float64)
    occupation = (n_electrons - 2 * k).clamp(0, 2)  # 2, 2, …, (1), 0, 0
    return torch.einsum("mk,mk->m", occupation, lam).to(mask.dtype)


# --- the graph neural network ---


def embed(x: torch.Tensor, mask: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h⁰_{mic} = σ(x_{mif} W_{fc} + b_c) · mask_{mi}   (f summed).   (M, N, F) → (M, N, C)."""
    return torch.relu(torch.einsum("mif,fc->mic", x, W) + b) * mask[..., None]


def message(A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """s_{mic} = A_{mij} h_{mjc}   (j summed): each atom adds up its neighbours' states.   (M, N, C)."""
    return torch.einsum("mij,mjc->mic", A, h)


def update(h, s, mask, U, V, c) -> torch.Tensor:
    """h'_{mid} = σ(h_{mic} U_{cd} + s_{mic} V_{cd} + c_d) · mask_{mi}   (c summed), plus a residual h."""
    out = torch.einsum("mic,cd->mid", h, U) + torch.einsum("mic,cd->mid", s, V) + c
    return (h + torch.relu(out)) * mask[..., None]


def atom_contributions(h: torch.Tensor, Q1, q1, Q2, q2) -> torch.Tensor:
    """e_{mi} = Q2_d σ(h_{mic} Q1_{cd} + q1_d) + q2: a per-atom energy.   (M, N, C) → (M, N)."""
    return torch.einsum("mid,d->mi", torch.relu(torch.einsum("mic,cd->mid", h, Q1) + q1), Q2) + q2


def gnn(data: dict, w: dict, return_atoms: bool = False):
    """y_m = Σ_i mask_{mi} e_{mi} after len(U) rounds of message passing.   → (M,) (and e (M, N))."""
    A, mask = data["A"], data["mask"]
    h = embed(data["x"], mask, w["W_in"], w["b_in"])
    for U, V, c in zip(w["U"], w["V"], w["c"]):
        h = update(h, message(A, h), mask, U, V, c)
    e = atom_contributions(h, w["Q1"], w["q1"], w["Q2"], w["q2"]) * mask
    y = e.sum(-1)  # Σ_i: a size-extensive, permutation-invariant readout
    return (y, e) if return_atoms else y


# --- the baseline ---


def mlp(data: dict, w: dict) -> torch.Tensor:
    """Fixed-size MLP on the flattened padded adjacency a_m = vec(A_m) ∈ R^{N²}: no notion of atoms or relabelling."""
    a = data["A"].flatten(1)  # (M, N²)
    h = torch.relu(a @ w["W1"] + w["b1"])
    h = torch.relu(h @ w["W2"] + w["b2"])
    return h @ w["W3"] + w["b3"]


def mse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return ((pred - true) ** 2).mean()


def count_baseline(data: dict, y: torch.Tensor, fit: torch.Tensor | None = None):
    """Linear 'bond counting' model y ≈ a N_atoms + b N_bonds + c, fitted by least squares on the training set."""
    n_atoms = data["mask"].sum(-1)
    n_bonds = data["A"].sum((-1, -2)) / 2
    X = torch.stack([n_atoms, n_bonds, torch.ones_like(n_atoms)], dim=-1)  # (M, 3)
    if fit is None:
        fit = torch.linalg.lstsq(X.double(), y[:, None].double()).solution[:, 0].to(y.dtype)
    return X @ fit, fit
