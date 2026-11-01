"""Physics checks for the biophysics polymer problems: HP lattice folding and the worm-like chain."""

import math

import torch

from problems.biophysics.hp_lattice_folding import equation as hp_eq
from problems.biophysics.hp_lattice_folding import solve as hp_solve
from problems.biophysics.hp_lattice_folding import tensors as hp_t
from problems.biophysics.worm_like_chain import equation as wlc_eq
from problems.biophysics.worm_like_chain import solve as wlc_solve
from problems.biophysics.worm_like_chain import tensors as wlc_t

CPU = torch.device("cpu")


# ---------------------------------------------------------------- HP lattice protein


def _brute_force_energy(chain: torch.Tensor, seq: str) -> int:
    """Count H–H contacts with plain Python loops."""
    pts = [tuple(p) for p in chain.long().tolist()]
    e = 0
    for i in range(len(seq)):
        for j in range(i + 2, len(seq)):
            if seq[i] == seq[j] == "H" and abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1]) == 1:
                e -= 1
    return e


def test_hp_energy_of_known_folds():
    # a 2×2 square of four H residues: one non-bonded contact (0–3)
    square = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    h = hp_t.hydrophobic("HHHH", CPU)
    assert hp_eq.chain_energy(square, h).item() == -1
    assert hp_eq.chain_energy(square, hp_t.hydrophobic("HPPH", CPU)).item() == -1
    assert hp_eq.chain_energy(square, hp_t.hydrophobic("HPHP", CPU)).item() == 0
    # a straight rod has no contacts at all
    rod = hp_t.straight_chains(3, 20, CPU)
    assert torch.all(hp_eq.chain_energy(rod, hp_t.hydrophobic("H" * 20, CPU)) == 0)
    # the contact map is symmetric with an empty band |i - j| ≤ 1
    C = hp_eq.contact_map(hp_eq.squared_distances(square))[0]
    assert torch.equal(C, C.T) and C.diagonal().sum() == 0 and C.diagonal(1).sum() == 0
    assert hp_eq.self_avoiding(square, hp_eq.squared_distances(square)).all()


def test_hp_self_avoidance_check_rejects_bad_chains():
    good = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 2.0]])
    overlap = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    broken = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    x = torch.stack([good, overlap, broken])
    assert hp_eq.self_avoiding(x, hp_eq.squared_distances(x)).tolist() == [True, False, False]


def test_hp_moves_keep_chain_connected():
    """Pull moves always produce a connected chain (only overlaps can make them invalid)."""
    gen = torch.Generator().manual_seed(3)
    x = hp_t.straight_chains(64, 12, CPU)
    for _ in range(200):
        new = hp_solve.propose(x, torch.rand(64, 4, generator=gen))
        ok = hp_eq.self_avoiding(new, hp_eq.squared_distances(new))
        x = torch.where(ok[:, None, None], new, x)
    pulls = hp_solve.pull_move(x, torch.randint(0, 11, (64,), generator=gen), torch.ones(64))
    bonds = (pulls[:, 1:] - pulls[:, :-1]).abs().sum(-1)
    assert torch.all(bonds == 1)
    assert (x != hp_t.straight_chains(64, 12, CPU)).any()  # the chains really moved


def test_hp_annealing_bookkeeping_and_ground_state():
    seq, e_ground = hp_t.sequence("benchmark_20")
    h = hp_t.hydrophobic(seq, CPU)
    out = hp_solve.anneal(hp_t.straight_chains(64, len(seq), CPU), h, hp_t.temperatures(1.0, 0.15, 2500, CPU), 20)
    # ΔE bookkeeping equals a full recount, both with einsum and with Python loops
    assert torch.equal(out["E"], hp_eq.chain_energy(out["x"], h))
    for r in range(4):
        assert out["E"][r].item() == _brute_force_energy(out["x"][r], seq)
    assert torch.equal(out["best_E"], hp_eq.chain_energy(out["best_x"], h))
    # every recorded chain is a self-avoiding walk
    snaps = out["snapshots"].flatten(0, 1)
    assert hp_eq.self_avoiding(snaps, hp_eq.squared_distances(snaps)).all()
    # nothing goes below the known minimum, and annealing gets close to it
    assert out["best_E"].min().item() >= e_ground
    assert out["best_E"].min().item() <= e_ground + 1
    assert out["energies"][-1].mean() < out["energies"][1].mean()


# ---------------------------------------------------------------- worm-like chain


def test_wlc_force_extension_limits():
    P = 50.0
    b = 2 * P
    # low force: both laws are linear with the same slope z/L = 2 P F / 3 kT
    F = torch.tensor([1e-4, 3e-4], dtype=torch.float64)
    slope = 2 * P / (3 * wlc_eq.KT)
    assert torch.allclose(wlc_eq.marko_siggia_extension(F, P) / F, torch.full_like(F, slope), rtol=2e-3)
    assert torch.allclose(wlc_eq.fjc_extension(F, b) / F, torch.full_like(F, slope), rtol=2e-3)
    # high force: 1 - z/L → ½ √(kT / F P) for the WLC, kT / (F b) for the FJC
    F = torch.tensor([200.0, 1000.0], dtype=torch.float64)
    wlc = 1 - wlc_eq.marko_siggia_extension(F, P)
    assert torch.allclose(wlc, 0.5 * torch.sqrt(wlc_eq.KT / (F * P)), rtol=0.03)
    assert torch.allclose(1 - wlc_eq.fjc_extension(F, b), wlc_eq.KT / (F * b), rtol=1e-6)
    # the inversion is exact: F(z(F)) = F
    F = torch.logspace(-2, 1, 9, dtype=torch.float64)
    assert torch.allclose(wlc_eq.marko_siggia_force(wlc_eq.marko_siggia_extension(F, P), P), F, rtol=1e-8)


def test_wlc_stiffness_gives_requested_persistence_length():
    for P, a in [(50.0, 2.5), (20.0, 1.0), (100.0, 5.0)]:
        K = wlc_t.stiffness_for(P, a)
        assert math.isclose(wlc_eq.persistence_from_stiffness(K, a), P, rel_tol=1e-9)
        assert math.isclose(K, P / a + 0.5, rel_tol=0.01)


def test_wlc_force_free_chains_have_exact_tangent_correlation():
    P, a = 50.0, 2.5
    K = wlc_t.stiffness_for(P, a)
    t = wlc_t.relaxed_chains(4, 64, 200, K, seed=0, device=CPU)
    assert torch.allclose(t.norm(dim=-1), torch.ones(()), atol=1e-5)
    c = wlc_eq.tangent_correlation(t, 20)
    n = torch.arange(21, dtype=c.dtype)
    assert torch.allclose(c, torch.exp(-n * a / P), atol=0.03)
    assert math.isclose(wlc_eq.persistence_from_correlation(c, a), P, rel_tol=0.08)


def test_wlc_local_energy_change_matches_total_energy():
    K = 10.0
    f = torch.tensor([0.0, 1.5])
    t = wlc_t.relaxed_chains(2, 3, 30, K, seed=1, device=CPU).double()
    i = 7
    t_new = t.clone()
    t_new[:, :, i] = torch.nn.functional.normalize(t[:, :, i] + 0.3 * torch.randn(2, 3, 3, dtype=t.dtype), dim=-1)
    dE_local = wlc_eq.local_energy_change(wlc_eq.local_field(t, K, f.double()), t, t_new)[:, :, i]
    dE_total = wlc_eq.bending_energy(t_new, K, f.double()) - wlc_eq.bending_energy(t, K, f.double())
    assert torch.allclose(dE_local, dE_total, atol=1e-10)


def test_wlc_monte_carlo_recovers_persistence_length():
    P, a, L = 50.0, wlc_t.SEGMENT, 600.0
    K = wlc_t.stiffness_for(P, a)
    F = torch.tensor([0.0, 0.6, 1.2, 2.5])
    f = wlc_t.reduced_forces(F, a)
    t0 = wlc_t.relaxed_chains(len(F), 6, int(L / a), K, seed=1, device=CPU)
    t0[1:] = torch.tensor([0.0, 0.0, 1.0])  # pulled chains start straight: they relax fast at these forces
    out = wlc_solve.simulate(t0, K, f, sweeps=400, n_records=5, show=[1], seed=0)
    assert torch.allclose(out["t"].norm(dim=-1), torch.ones(()), atol=1e-4)  # tangents stay unit vectors
    # the free chain keeps the right correlation
    assert math.isclose(out["correlation"][1].item(), wlc_eq.segment_correlation(K), rel_tol=2e-3)
    assert math.isclose(wlc_eq.persistence_from_correlation(out["correlation"][:20], a), P, rel_tol=0.1)
    # the stretched chains follow Marko–Siggia, and a fit gives back P
    z = out["z_mean"][1:]
    assert torch.allclose(z, wlc_eq.marko_siggia_extension(F[1:], P), atol=0.02)
    P_fit = torch.exp(torch.log(wlc_eq.persistence_from_force(F[1:], z)).mean()).item()
    assert math.isclose(P_fit, P, rel_tol=0.12)
    assert torch.all(z[1:] > z[:-1])  # more force, more extension
