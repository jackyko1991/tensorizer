r"""Physics → tensor translation: from a predicted distance map to 3D coordinates.

Indices: i, j residues (N);  a, b Cartesian axes (3);  k distance bins (K);  e feature channels (E).

1. Distogram (what the network predicts):  p_{ijk} = probability that residues i, j are c_k apart.
   Mean and spread:  d̂_ij = p_{ijk} c_k   (k summed),   Var_ij = p_{ijk} c_k² - d̂_ij².
2. Attention over residues (the core operation of a transformer):
   a_ij = q_{ie} k_{je} / √E + b_ij   (e summed),   A_ij = softmax_j(a_ij),   h'_ie = h_ie + A_ij v_je   (j summed).
3. Structure: find x_{ia} with r_ij = √((x_ia - x_ja)(x_ia - x_ja)) (a summed) close to d̂_ij, i.e. minimise the stress
   S = Σ_{i<j} w_ij (r_ij - d̂_ij)²  +  far pairs: Σ relu(d_far - r_ij)²  +  k_b Σ_i (r_{i,i+1} - 3.8 Å)².
4. Classical MDS: G_ij = -½ J_{ik} D²_{kl} J_{lj} (k, l summed) is the Gram matrix x_ia x_ja of centred coordinates.
5. Kabsch: H_ab = p_ia q_ib (i summed) = U S Vᵀ;  the best rotation is R = V diag(1, 1, det(V Uᵀ)) Uᵀ.
"""

import math

import torch

CA_CA = 3.8  # Å


def pair_distances(x: torch.Tensor) -> torch.Tensor:
    """r_ij = |x_i - x_j| for coordinates (..., N, 3) → (..., N, N).  The axis index a is summed."""
    diff = x[..., :, None, :] - x[..., None, :, :]  # (..., N, N, 3)
    return torch.sqrt(torch.einsum("...ija,...ija->...ij", diff, diff) + 1e-9)


def distogram(d: torch.Tensor, sigma: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """p_ijk = softmax_k(-(d_ij - c_k)² / 2σ_ij²): a predicted distance spread over K bins.   (N, N, K)

    Distances beyond the last bin pile up in it: the distogram only says "22 Å or more".
    """
    logits = -((d[..., None] - centres) ** 2) / (2 * sigma[..., None] ** 2)
    return torch.softmax(logits, dim=-1)


def distogram_mean_var(p: torch.Tensor, centres: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """d̂_ij = p_ijk c_k and Var_ij = p_ijk c_k² - d̂_ij²   (k summed).   Both (N, N)."""
    mean = torch.einsum("ijk,k->ij", p, centres)
    var = torch.einsum("ijk,k->ij", p, centres**2) - mean**2
    return mean, var


def attention(h: torch.Tensor, Wq, Wk, Wv, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One head of self-attention with a pair bias.  h: (N, E), W*: (E, E), bias: (N, N).

    a_ij = q_ie k_je / √E + b_ij    →   A_ij = softmax over j    →   h'_ie = h_ie + A_ij v_je.
    The pair bias b_ij is how AlphaFold's Evoformer lets the pair (distance) picture steer the residues.
    """
    q, k, v = h @ Wq, h @ Wk, h @ Wv  # (N, E) each
    logits = torch.einsum("ie,je->ij", q, k) / math.sqrt(h.shape[1]) + bias
    A = torch.softmax(logits, dim=1)  # each row sums to 1: residue i's attention over all j
    return h + torch.einsum("ij,je->ie", A, v), A


def stress(x: torch.Tensor, d: torch.Tensor, w: torch.Tensor, far: torch.Tensor, d_far: float, k_bond: float):
    """S per structure for coordinates x (B, N, 3); d, w, far (bool) are (N, N).   Returns (B,).

    Near pairs: weighted squared error of the distance.  Far pairs ("≥ d_far"): only penalised when too close.
    Bonds: consecutive Cα atoms must stay 3.8 Å apart.
    """
    r = pair_distances(x)  # (B, N, N)
    upper = torch.triu(torch.ones_like(w, dtype=torch.bool), diagonal=1)
    near = upper & ~far
    s_near = (w * (r - d) ** 2 * near).sum((1, 2))
    s_far = (torch.relu(d_far - r) ** 2 * (upper & far)).sum((1, 2))
    bonds = r.diagonal(offset=1, dim1=1, dim2=2)  # (B, N-1): r_{i,i+1}
    s_bond = k_bond * ((bonds - CA_CA) ** 2).sum(1)
    return s_near + s_far + s_bond


def classical_mds(D: torch.Tensor, dim: int = 3) -> torch.Tensor:
    """Coordinates (N, dim) whose distances best match D (N, N): G = -½ J D² J, then the top eigenvectors.

    For exact Euclidean distances G_ij = x_ia x_ja exactly, so x = V Λ^{1/2} recovers the structure up to a rotation,
    a translation and a mirror image (distances cannot tell left from right).
    """
    n = D.shape[0]
    J = torch.eye(n, dtype=D.dtype, device=D.device) - 1.0 / n  # centring matrix
    G = -0.5 * torch.einsum("ik,kl,lj->ij", J, D**2, J)
    evals, evecs = torch.linalg.eigh(G)  # ascending
    top = evals[-dim:].flip(0).clamp(min=0)
    return evecs[:, -dim:].flip(1) * torch.sqrt(top)


def kabsch(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Rotate and translate P (..., N, 3) onto Q (N, 3) with the best proper rotation (det R = +1).   (..., N, 3)"""
    Pc = P - P.mean(-2, keepdim=True)
    Qc = Q - Q.mean(-2, keepdim=True)
    H = torch.einsum("...ia,...ib->...ab", Pc, Qc.expand_as(Pc))  # (..., 3, 3) covariance
    U, _, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vh.mT @ U.mT))  # -1 would be a reflection: forbid it
    S = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vh.mT @ S @ U.mT  # (..., 3, 3)
    return torch.einsum("...ab,...ib->...ia", R, Pc) + Q.mean(-2, keepdim=True)


def rmsd(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """√(mean_i |p_i - q_i|²) after Kabsch superposition.   (...)"""
    diff = kabsch(P, Q) - Q
    return torch.sqrt(torch.einsum("...ia,...ia->...", diff, diff) / Q.shape[0])


def tm_score(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """TM-like score (1/N) Σ_i 1 / (1 + (|p_i - q_i| / d0)²) on the Kabsch superposition; 1 = identical.

    d0 = 1.24 (N - 15)^{1/3} - 1.8 Å is the TM-score length scale. The real TM-score also searches over
    superpositions to maximise the score, so this value is a lower bound.
    """
    n = Q.shape[0]
    d0 = max(1.24 * max(n - 15, 1) ** (1 / 3) - 1.8, 0.5)
    dist = torch.linalg.norm(kabsch(P, Q) - Q, dim=-1)
    return (1.0 / (1.0 + (dist / d0) ** 2)).mean(-1)


def pseudo_dihedrals(x: torch.Tensor) -> torch.Tensor:
    """Dihedral angle of every four consecutive Cα atoms, (..., N, 3) → (..., N-3), in degrees.

    About +50° in a right-handed α-helix; a mirror image flips every sign. That is how we pick the enantiomer.
    """
    b1, b2, b3 = (x[..., 1:, :] - x[..., :-1, :]).unfold(-2, 3, 1).unbind(-1)  # bond vectors i→i+1, ...
    n1, n2 = torch.cross(b1, b2, dim=-1), torch.cross(b2, b3, dim=-1)  # normals of the two planes
    y = torch.linalg.norm(b2, dim=-1) * (b1 * n2).sum(-1)
    return torch.rad2deg(torch.atan2(y, (n1 * n2).sum(-1)))
