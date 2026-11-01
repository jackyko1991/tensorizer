"""Physics checks, written first (TDD): what any correct structure-prediction code must satisfy.

Run with `pytest problems/biophysics/mini_alphafold/test.py`. Each test is a geometric law: RMSD is invariant under
rigid motions but not reflections, classical MDS recovers coordinates from exact distances, attention rows are
probabilities, and exact distances fold back to the target.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _rotation(seed: int = 0) -> torch.Tensor:
    """A random proper rotation (det +1), float64."""
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    return q * torch.sign(torch.linalg.det(q))


def test_target_is_a_chain_of_right_handed_helices():
    x = tensors.target_structure(16, CPU).double()
    bonds = equation.pair_distances(x).diagonal(1)
    assert (bonds - 3.8).abs().max() < 0.05  # helix rise and radius give ≈ 3.83 Å; the loop is exactly 3.8 Å
    dihedral = equation.pseudo_dihedrals(x)
    assert abs(dihedral[:10].mean().item() - 50.0) < 1.0  # α-helix Cα pseudo-dihedral ≈ +50°
    assert torch.allclose(equation.pseudo_dihedrals(solve.mirror(x)), -dihedral)  # a mirror flips the hand
    picked, flipped = solve.choose_hand(solve.mirror(x))
    assert flipped and torch.allclose(picked, x)


def test_kabsch_rmsd_is_zero_for_a_rotated_copy_but_not_for_a_mirror_image():
    x = tensors.target_structure(12, CPU).double()
    moved = x @ _rotation().T + torch.tensor([3.0, -7.0, 1.5], dtype=torch.float64)
    assert equation.rmsd(moved, x).item() < 1e-9
    assert abs(equation.tm_score(moved, x).item() - 1.0) < 1e-12
    assert equation.rmsd(solve.mirror(moved), x).item() > 1.0  # proper rotations cannot undo a reflection
    batch = torch.stack([moved, x + 0.1])  # batched call: a uniform shift is also removed
    assert equation.rmsd(batch, x).max().item() < 1e-9


def test_classical_mds_recovers_coordinates_up_to_rigid_motion_and_reflection():
    x = tensors.target_structure(14, CPU).double()
    D = equation.pair_distances(x)
    y = equation.classical_mds(D)
    assert torch.allclose(equation.pair_distances(y), D, atol=1e-6)
    assert solve.best_rmsd_any_hand(y, x).item() < 1e-6


def test_distogram_mean_and_attention_rows():
    centres = tensors.bin_centres(CPU)
    assert len(centres) == 64 and abs(centres[0].item() - 2.15625) < 1e-6
    d = torch.tensor([[0.0, 6.0], [6.0, 0.0]])
    sigma = torch.full_like(d, 0.4)
    mean, var = equation.distogram_mean_var(equation.distogram(d, sigma, centres), centres)
    assert abs(mean[0, 1].item() - 6.0) < 0.02 and abs(var[0, 1].sqrt().item() - 0.4) < 0.05
    h = torch.randn(7, 8)
    W = [torch.randn(8, 8) for _ in range(3)]
    out, A = equation.attention(h, *W, torch.zeros(7, 7))
    assert out.shape == (7, 8)
    assert torch.allclose(A.sum(1), torch.ones(7))


def test_structure_module_folds_exact_distances():
    x = tensors.target_structure(10, CPU)
    D = equation.pair_distances(x)
    w = torch.ones_like(D)
    far = torch.zeros_like(D, dtype=torch.bool)
    x0 = solve.initial_guesses(D, 3)
    _, snaps, _, final = solve.fold(x0, D, w, far, 18.0, 10.0, 300)
    model, _ = solve.choose_hand(snaps[-1, int(final.argmin())])
    assert equation.rmsd(model, x).item() < 0.1
