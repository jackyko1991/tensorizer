"""The two stages: refine the distance map with a tiny attention block, then fold coordinates to fit it."""

import torch

from problems._common.progress import steps

from .equation import attention, classical_mds, pair_distances, pseudo_dihedrals, rmsd, stress

READOUT_DIM = 6  # dimension of the feature space the read-out maps each residue into


def refine(d: torch.Tensor, w: torch.Tensor, far: torch.Tensor, d_far: float, feats: torch.Tensor, n_steps: int):
    """Train one attention block for n_steps to reproduce the predicted distances d (N, N); returns (d_ref, A).

    Each residue gets an embedding h_i = (positional features + a learned vector); after attention, a linear
    read-out puts it at a point u_i in a 6-dimensional feature space and the refined distance is |u_i - u_j|.
    N points in 6D cannot fit N² independent noise values, so the fit smooths the noise out.
    """
    torch.manual_seed(0)
    n, e = feats.shape
    p = {
        "h": torch.zeros(n, e),
        "Wq": torch.randn(e, e) / e**0.5,
        "Wk": torch.randn(e, e) / e**0.5,
        "Wv": torch.randn(e, e) / e**0.5,
        "Wout": torch.randn(e, READOUT_DIM) * 3.0,
        "tau": torch.tensor(4.0),
    }
    p = {k: v.to(d.device).requires_grad_() for k, v in p.items()}
    opt = torch.optim.Adam(p.values(), lr=0.02)
    for _ in steps(n_steps, "attention training"):
        d_ref, A = _forward(p, feats, d)
        loss = stress_2d(d_ref, d, w, far, d_far)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        d_ref, A = _forward(p, feats, d)
    return d_ref, A


def _forward(p: dict, feats: torch.Tensor, d: torch.Tensor):
    h = feats + p["h"]  # (N, E)
    bias = -d / p["tau"].abs().clamp(min=0.5)  # pair bias: residues predicted to be close attend to each other
    h2, A = attention(h, p["Wq"], p["Wk"], p["Wv"], bias)
    u = h2 @ p["Wout"]  # (N, 6)
    return pair_distances(u), A


def stress_2d(r: torch.Tensor, d: torch.Tensor, w: torch.Tensor, far: torch.Tensor, d_far: float) -> torch.Tensor:
    """Same fit as the structure stress, written on a distance map r (N, N) instead of coordinates."""
    upper = torch.triu(torch.ones_like(far), diagonal=1)
    return (w * (r - d) ** 2 * (upper & ~far)).sum() + (torch.relu(d_far - r) ** 2 * (upper & far)).sum()


def initial_guesses(d: torch.Tensor, n_random: int, seed: int = 0) -> torch.Tensor:
    """Starts for gradient descent, shape (B, N, 3): classical MDS first, then random clouds a few Å across."""
    gen = torch.Generator().manual_seed(seed)
    n = d.shape[0]
    random = 6.0 * torch.randn(n_random, n, 3, generator=gen).to(d.device)
    return torch.cat([classical_mds(d)[None], random])


def fold(x0: torch.Tensor, d, w, far, d_far: float, k_bond: float, n_steps: int, n_snap: int = 90):
    """Gradient descent (Adam) on the stress for every start at once.

    Returns (snapshot steps (T,), snapshots (T, B, N, 3), stress history (n_steps, B), final stress (B,)).
    Snapshots are spaced quadratically in the step number: most of the folding happens early.
    """
    x = x0.clone().requires_grad_()
    opt = torch.optim.Adam([x], lr=0.5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=0.01)
    keep = set((n_steps * (torch.arange(1, n_snap + 1) / n_snap) ** 2).round().long().tolist())
    snaps, at, history = [x0.clone()], [0], []
    for step in steps(n_steps, "structure module"):
        S = stress(x, d, w, far, d_far, k_bond)  # (B,)
        opt.zero_grad()
        S.sum().backward()  # the starts are independent: one backward pass serves them all
        opt.step()
        sched.step()
        history.append(S.detach())
        if step + 1 in keep:
            snaps.append(x.detach().clone())
            at.append(step + 1)
    with torch.no_grad():
        final = stress(x, d, w, far, d_far, k_bond)
    return torch.tensor(at), torch.stack(snaps), torch.stack(history), final


def mirror(x: torch.Tensor) -> torch.Tensor:
    """The mirror image (enantiomer): flip the z coordinate.   (..., N, 3)"""
    return x * torch.tensor([1.0, 1.0, -1.0], device=x.device)


def choose_hand(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Distances cannot tell a structure from its mirror image, so pick the one whose helices are right-handed
    (mean Cα pseudo-dihedral > 0, as for L-amino acids). Returns (structure, whether it was flipped)."""
    flip = pseudo_dihedrals(x).median().item() < 0
    return (mirror(x) if flip else x), flip


def best_rmsd_any_hand(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """min(RMSD(x), RMSD(mirror x)): how close the shape is, ignoring the hand.   (...)"""
    return torch.minimum(rmsd(x, target), rmsd(mirror(x), target))
