"""Physics checks, written first (TDD): what any correct HP lattice folding code must satisfy.

Run with `pytest problems/biophysics/hp_lattice_folding/test.py`. Each test is a physical law: energies count H–H
contacts, chains stay connected and self-avoiding, and annealing never goes below the known ground-state energy.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _brute_force_energy(chain: torch.Tensor, seq: str) -> int:
    """Count H–H contacts with plain Python loops."""
    pts = [tuple(p) for p in chain.long().tolist()]
    e = 0
    for i in range(len(seq)):
        for j in range(i + 2, len(seq)):
            if seq[i] == seq[j] == "H" and abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1]) == 1:
                e -= 1
    return e


def test_energy_of_known_folds():
    # a 2×2 square of four H residues: one non-bonded contact (0–3)
    square = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    h = tensors.hydrophobic("HHHH", CPU)
    assert equation.chain_energy(square, h).item() == -1
    assert equation.chain_energy(square, tensors.hydrophobic("HPPH", CPU)).item() == -1
    assert equation.chain_energy(square, tensors.hydrophobic("HPHP", CPU)).item() == 0
    # a straight rod has no contacts at all
    rod = tensors.straight_chains(3, 20, CPU)
    assert torch.all(equation.chain_energy(rod, tensors.hydrophobic("H" * 20, CPU)) == 0)
    # the contact map is symmetric with an empty band |i - j| ≤ 1
    C = equation.contact_map(equation.squared_distances(square))[0]
    assert torch.equal(C, C.T) and C.diagonal().sum() == 0 and C.diagonal(1).sum() == 0
    assert equation.self_avoiding(square, equation.squared_distances(square)).all()


def test_self_avoidance_check_rejects_bad_chains():
    good = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 2.0]])
    overlap = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    broken = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    x = torch.stack([good, overlap, broken])
    assert equation.self_avoiding(x, equation.squared_distances(x)).tolist() == [True, False, False]


def test_moves_keep_chain_connected():
    """Pull moves always produce a connected chain (only overlaps can make them invalid)."""
    gen = torch.Generator().manual_seed(3)
    x = tensors.straight_chains(64, 12, CPU)
    for _ in range(200):
        new = solve.propose(x, torch.rand(64, 4, generator=gen))
        ok = equation.self_avoiding(new, equation.squared_distances(new))
        x = torch.where(ok[:, None, None], new, x)
    pulls = solve.pull_move(x, torch.randint(0, 11, (64,), generator=gen), torch.ones(64))
    bonds = (pulls[:, 1:] - pulls[:, :-1]).abs().sum(-1)
    assert torch.all(bonds == 1)
    assert (x != tensors.straight_chains(64, 12, CPU)).any()  # the chains really moved


def test_annealing_bookkeeping_and_ground_state():
    seq, e_ground = tensors.sequence("benchmark_20")
    h = tensors.hydrophobic(seq, CPU)
    out = solve.anneal(tensors.straight_chains(64, len(seq), CPU), h, tensors.temperatures(1.0, 0.15, 2500, CPU), 20)
    # ΔE bookkeeping equals a full recount, both with einsum and with Python loops
    assert torch.equal(out["E"], equation.chain_energy(out["x"], h))
    for r in range(4):
        assert out["E"][r].item() == _brute_force_energy(out["x"][r], seq)
    assert torch.equal(out["best_E"], equation.chain_energy(out["best_x"], h))
    # every recorded chain is a self-avoiding walk
    snaps = out["snapshots"].flatten(0, 1)
    assert equation.self_avoiding(snaps, equation.squared_distances(snaps)).all()
    # nothing goes below the known minimum, and annealing gets close to it
    assert out["best_E"].min().item() >= e_ground
    assert out["best_E"].min().item() <= e_ground + 1
    assert out["energies"][-1].mean() < out["energies"][1].mean()
