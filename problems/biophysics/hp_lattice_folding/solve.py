"""Metropolis Monte Carlo with pull and crankshaft moves, run for all replicas at once, while the temperature falls.

Every step, each replica proposes one move. Moves that break the chain or make it cross itself are rejected; the
rest are accepted with the Metropolis rule. All R replicas advance together as one (R, N, 2) tensor.
"""

import torch

from problems._common.progress import steps

from .equation import contact_map, energy, metropolis, self_avoiding, squared_distances


def pull_move(x: torch.Tensor, i: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """Pull move (Lesh et al. 2003) towards residue 0.   x: (R, N, 2), i: (R,) in [0, N-2], side: (R,) = ±1.

    Residue i jumps to L, a free corner next to residue i+1; residue i-1 follows to C, the corner next to both L and
    the old site of i; then every residue j < i-1 moves two places up the chain (to the old site of j+2) until one
    already touches the new site of j+1. With C equal to the old site of i-1 this is a simple corner flip, and with
    i = 0 it rotates the end residue.
    """
    r = torch.arange(x.shape[0], device=x.device)
    j = torch.arange(x.shape[1], device=x.device)[None, :]  # (1, N)
    i_ = i[:, None]  # (R, 1)
    bond = x[r, i + 1] - x[r, i]  # (R, 2) unit lattice step from i to i+1
    perp = side[:, None] * torch.stack([-bond[:, 1], bond[:, 0]], dim=-1)  # (R, 2) one of the two sideways steps
    L = x[r, i + 1] + perp  # new site of residue i
    C = x[r, i] + perp  # new site of residue i-1

    candidate = torch.where((j <= i_ - 2)[..., None], torch.roll(x, -2, dims=1), x)  # residue j → old site of j+2
    candidate = torch.where((j == i_ - 1)[..., None], C[:, None], candidate)
    candidate = torch.where((j == i_)[..., None], L[:, None], candidate)

    # residue j < i stays where it is if it already touches the new site of j+1 — and then so do all below it
    touches = (x - torch.roll(candidate, -1, dims=1)).abs().sum(-1) == 1  # (R, N)
    keeps_going = (~touches | (j >= i_)).int()
    moves = keeps_going.flip(1).cumprod(1).flip(1).bool() & (j <= i_)  # (R, N)
    return torch.where(moves[..., None], candidate, x)


def crankshaft_move(x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """Flip the U-shaped pair (i, i+1) to the other side of the line from i-1 to i+2.   i: (R,) in [1, N-3].

    x'_i = 2 x_{i-1} - x_i,   x'_{i+1} = 2 x_{i+2} - x_{i+1}.   For any shape other than a U the bonds break,
    and self_avoiding() rejects the move.
    """
    r = torch.arange(x.shape[0], device=x.device)
    new = x.clone()
    new[r, i] = 2 * x[r, i - 1] - x[r, i]
    new[r, i + 1] = 2 * x[r, i + 2] - x[r, i + 1]
    return new


def propose(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """One random move per replica: a pull from either end of the chain (80 %) or a crankshaft (20 %).

    u: (R, 4) uniform random numbers that pick the residue, the side, the chain end and the move type.
    """
    R, N, _ = x.shape
    i = (u[:, 0] * (N - 1)).long()  # residue in [0, N-2]
    side = torch.where(u[:, 1] < 0.5, -1.0, 1.0)
    from_end = u[:, 2] < 0.5
    crank = u[:, 3] < 0.2

    # pulling towards the far end = the same pull on the chain read backwards
    flipped = torch.where(from_end[:, None, None], x.flip(1), x)
    pulled = pull_move(flipped, i, side)
    pulled = torch.where(from_end[:, None, None], pulled.flip(1), pulled)

    cranked = crankshaft_move(x, i.clamp(1, N - 3))
    return torch.where(crank[:, None, None], cranked, pulled)


def anneal(x0: torch.Tensor, h: torch.Tensor, temps: torch.Tensor, n_records: int, seed: int = 0) -> dict:
    """Run len(temps) Monte Carlo steps. Returns snapshots of every replica and the energy bookkeeping.

    E is never recomputed from scratch inside the loop: it is updated by ΔE of each accepted move, so the tests can
    check that it still matches a fresh recount at the end.
    """
    gen = torch.Generator().manual_seed(seed)
    n_steps = len(temps)
    every = max(1, n_steps // n_records)
    x = x0.clone()
    E = energy(contact_map(squared_distances(x)), h)  # (R,)
    best_E, best_x = E.clone(), x.clone()
    snapshots, energies, record_steps = [x.clone()], [E.clone()], [0]
    accepted = torch.zeros((), device=x.device)

    for s, T in zip(steps(n_steps, "MC steps"), temps.tolist()):
        u = torch.rand(x.shape[0], 5, generator=gen).to(x.device)  # all random numbers of this step
        new = propose(x, u[:, :4])
        d = squared_distances(new)  # (R, N, N)
        dE = energy(contact_map(d), h) - E
        accept = self_avoiding(new, d) & metropolis(dE, T, u[:, 4])  # (R,)
        x = torch.where(accept[:, None, None], new, x)
        E = E + torch.where(accept, dE, torch.zeros_like(dE))
        accepted += accept.sum()

        better = E < best_E
        best_E = torch.where(better, E, best_E)
        best_x = torch.where(better[:, None, None], x, best_x)
        if (s + 1) % every == 0 or s == n_steps - 1:
            snapshots.append(x.clone())
            energies.append(E.clone())
            record_steps.append(s + 1)

    return {
        "x": x,  # (R, N, 2) final chains
        "E": E,  # (R,) energy kept by ΔE bookkeeping
        "best_x": best_x,  # (R, N, 2) lowest-energy chain each replica ever visited
        "best_E": best_E,  # (R,)
        "snapshots": torch.stack(snapshots),  # (K, R, N, 2)
        "energies": torch.stack(energies),  # (K, R)
        "record_steps": torch.tensor(record_steps),  # (K,)
        "acceptance": accepted.item() / (n_steps * x.shape[0]),
    }
