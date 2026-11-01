"""Checks for the structure and pattern biophysics problems: mini AlphaFold, the Gō model, Turing patterns."""

import math

import torch

from problems.biophysics.coarse_grained_md import equation as go
from problems.biophysics.coarse_grained_md import solve as go_solve
from problems.biophysics.coarse_grained_md import tensors as go_t
from problems.biophysics.mini_alphafold import equation as af
from problems.biophysics.mini_alphafold import solve as af_solve
from problems.biophysics.mini_alphafold import tensors as af_t
from problems.biophysics.turing_patterns import equation as gs
from problems.biophysics.turing_patterns import solve as gs_solve
from problems.biophysics.turing_patterns import tensors as gs_t

CPU = torch.device("cpu")


def _rotation(seed: int = 0) -> torch.Tensor:
    """A random proper rotation (det +1), float64."""
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    return q * torch.sign(torch.linalg.det(q))


# --- mini AlphaFold -------------------------------------------------------------------------------


def test_target_is_a_chain_of_right_handed_helices():
    x = af_t.target_structure(16, CPU).double()
    bonds = af.pair_distances(x).diagonal(1)
    assert (bonds - 3.8).abs().max() < 0.05  # helix rise and radius give ≈ 3.83 Å; the loop is exactly 3.8 Å
    dihedral = af.pseudo_dihedrals(x)
    assert abs(dihedral[:10].mean().item() - 50.0) < 1.0  # α-helix Cα pseudo-dihedral ≈ +50°
    assert torch.allclose(af.pseudo_dihedrals(af_solve.mirror(x)), -dihedral)  # a mirror flips the hand
    picked, flipped = af_solve.choose_hand(af_solve.mirror(x))
    assert flipped and torch.allclose(picked, x)


def test_kabsch_rmsd_is_zero_for_a_rotated_copy_but_not_for_a_mirror_image():
    x = af_t.target_structure(12, CPU).double()
    moved = x @ _rotation().T + torch.tensor([3.0, -7.0, 1.5], dtype=torch.float64)
    assert af.rmsd(moved, x).item() < 1e-9
    assert abs(af.tm_score(moved, x).item() - 1.0) < 1e-12
    assert af.rmsd(af_solve.mirror(moved), x).item() > 1.0  # proper rotations cannot undo a reflection
    batch = torch.stack([moved, x + 0.1])  # batched call: a uniform shift is also removed
    assert af.rmsd(batch, x).max().item() < 1e-9


def test_classical_mds_recovers_coordinates_up_to_rigid_motion_and_reflection():
    x = af_t.target_structure(14, CPU).double()
    D = af.pair_distances(x)
    y = af.classical_mds(D)
    assert torch.allclose(af.pair_distances(y), D, atol=1e-6)
    assert af_solve.best_rmsd_any_hand(y, x).item() < 1e-6


def test_distogram_mean_and_attention_rows():
    centres = af_t.bin_centres(CPU)
    assert len(centres) == 64 and abs(centres[0].item() - 2.15625) < 1e-6
    d = torch.tensor([[0.0, 6.0], [6.0, 0.0]])
    sigma = torch.full_like(d, 0.4)
    mean, var = af.distogram_mean_var(af.distogram(d, sigma, centres), centres)
    assert abs(mean[0, 1].item() - 6.0) < 0.02 and abs(var[0, 1].sqrt().item() - 0.4) < 0.05
    h = torch.randn(7, 8)
    W = [torch.randn(8, 8) for _ in range(3)]
    out, A = af.attention(h, *W, torch.zeros(7, 7))
    assert out.shape == (7, 8)
    assert torch.allclose(A.sum(1), torch.ones(7))


def test_structure_module_folds_exact_distances():
    x = af_t.target_structure(10, CPU)
    D = af.pair_distances(x)
    w = torch.ones_like(D)
    far = torch.zeros_like(D, dtype=torch.bool)
    x0 = af_solve.initial_guesses(D, 3)
    _, snaps, _, final = af_solve.fold(x0, D, w, far, 18.0, 10.0, 300)
    model, _ = af_solve.choose_hand(snaps[-1, int(final.argmin())])
    assert af.rmsd(model, x).item() < 0.1


# --- coarse-grained MD (Gō model) ----------------------------------------------------------------


def _go_model64():
    native = go_t.native_structure(6, CPU).double()
    return native, go_t.go_model(native)


def test_autograd_forces_match_finite_differences():
    native, model = _go_model64()
    x = (native + 0.3 * torch.randn(native.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64))[
        None
    ]
    F = go.forces(x, model)
    h = 1e-6
    for i, a in [(0, 0), (3, 1), (7, 2), (12, 0), (len(native) - 1, 1)]:
        xp, xm = x.clone(), x.clone()
        xp[0, i, a] += h
        xm[0, i, a] -= h
        numeric = -(go.total_energy(xp, model) - go.total_energy(xm, model)) / (2 * h)
        assert abs(numeric.item() - F[0, i, a].item()) < 1e-5 * max(1.0, abs(numeric.item()))


def test_native_fold_is_the_energy_minimum():
    native, model = _go_model64()
    n_contacts = int(model["native"].sum())
    E = go.total_energy(native[None], model).item()
    assert abs(E + n_contacts) < 0.05  # every contact sits at the bottom of its well (-ε); bonds and angles at rest
    assert go.forces(native[None], model).abs().max().item() < 0.05
    assert go.fraction_native(native[None], model).item() == 1.0
    # a contact's 12–10 well has its minimum −ε exactly at r⁰
    r0 = torch.tensor([6.0], dtype=torch.float64)
    r = torch.linspace(5.0, 8.0, 3001, dtype=torch.float64)[:, None]
    well = go.pair_energy(r, r0, torch.ones(1, dtype=torch.float64))
    assert abs(r[well.argmin()].item() - 6.0) < 2e-3 and abs(well.min().item() + 1.0) < 1e-9


def test_langevin_keeps_the_fold_cold_and_melts_it_hot():
    native = go_t.native_structure(6, CPU)
    model = go_t.go_model(native)
    T = torch.tensor([0.1, 3.0])
    x0 = native[None].repeat(2, 1, 1)
    _, Q, _, _ = go_solve.langevin(x0, T, model, 0.02, 1.0, 1500)
    assert Q[-200:, 0].mean() > 0.95
    assert Q[-200:, 1].mean() < 0.6


def test_melting_temperature_interpolates_the_half_point():
    T = torch.tensor([0.2, 0.4, 0.6, 0.8])
    q = torch.tensor([0.9, 0.7, 0.3, 0.2])
    assert abs(go_solve.melting_temperature(T, q) - 0.5) < 1e-6


# --- Turing patterns (Gray–Scott) ----------------------------------------------------------------


def test_uniform_states_are_steady():
    for F, k in [(0.06, 0.062), (0.078, 0.061), (0.03, 0.055)]:
        for u, v in gs.uniform_states(F, k):
            Ru, Rv = gs.reaction(torch.tensor(u), torch.tensor(v), torch.tensor(F), torch.tensor(k))
            assert abs(Ru.item()) < 1e-6 and abs(Rv.item()) < 1e-6
    assert len(gs.uniform_states(0.029, 0.057)) == 1  # F < 4 (F + k)²: only u = 1, v = 0


def test_dispersion_relation_grows_only_where_expected():
    q = torch.linspace(0, 1.5, 301, dtype=torch.float64)
    Du = gs_t.DU
    # the trivial state u = 1, v = 0 decays at every wavenumber
    lam = gs.growth_rate(gs.jacobian(1.0, 0.0, 0.06, 0.062), Du, Du / 2, q)
    assert lam.max() < 0
    # the "Turing" pair: stable at q = 0, unstable in a band of q > 0 that ends again at short wavelengths
    u, v = gs_t.turing_state(0.060, 0.062, Du, Du / 2)
    lam = gs.growth_rate(gs.jacobian(u, v, 0.060, 0.062), Du, Du / 2, q)
    band = q[lam > 0]
    assert lam[0] < 0 < lam.max()
    assert 0.3 < band.min() < band.max() < 0.8 and lam[-1] < 0
    # with equal diffusion λ(q) = λ(0) - D q²: no Turing instability is possible
    lam_eq = gs.growth_rate(gs.jacobian(u, v, 0.060, 0.062), Du, Du, q)
    assert torch.allclose(lam_eq, lam_eq[0] - Du * q**2, atol=1e-12)
    assert gs_t.turing_state(0.060, 0.062, Du, Du) is None


def test_laplacian_eigenvalue_and_linear_growth_rate():
    n, m = 64, 5
    x = torch.arange(n, dtype=torch.float64)
    qx = 2 * math.pi * m / n
    wave = torch.cos(qx * x)[None, :, None].expand(1, n, n)  # varies along i only
    assert torch.allclose(gs.laplacian(wave), -gs.lattice_q2(torch.tensor(qx)) * wave, atol=1e-12)

    # a tiny wave on the Turing-unstable state grows at the rate the (lattice) dispersion relation predicts
    F, k, Du, Dv = 0.060, 0.062, gs_t.DU, gs_t.DU / 2
    u0, v0 = gs_t.turing_state(F, k, Du, Dv)
    y = torch.stack([torch.full((1, n, n), u0), torch.full((1, n, n), v0)]).double() + 1e-7 * wave
    Ft, kt = torch.tensor(F, dtype=torch.float64), torch.tensor(k, dtype=torch.float64)
    amp = []
    for _ in range(1500):
        y = y + gs.rhs(y, Ft, kt, Du, Dv)  # explicit Euler, dt = 1
        amp.append((y[1] - v0).abs().max().item())
    measured = math.log(amp[1499] / amp[899]) / 600  # late enough for the decaying mode to have gone
    q_eff = torch.sqrt(gs.lattice_q2(torch.tensor([qx], dtype=torch.float64)))
    lam = gs.growth_rate(gs.jacobian(u0, v0, F, k), Du, Dv, q_eff).item()
    assert abs(measured - math.log(1 + lam)) < 0.03 * lam


def test_dominant_wavelength_of_stripes():
    n = 96
    x = torch.arange(n, dtype=torch.float32)
    stripes = torch.sin(2 * math.pi * 8 * x / n)[None, None, :].expand(1, n, n)
    assert gs_solve.dominant_wavelength(stripes).item() == n / 8
