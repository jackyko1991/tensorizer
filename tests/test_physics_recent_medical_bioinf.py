"""Physics checks for photon-counting CT (medical) and the tiny protein language model (bioinformatics)."""

import torch

from problems.bioinformatics.protein_language_model import equation as plm_eq
from problems.bioinformatics.protein_language_model import solve as plm_solve
from problems.bioinformatics.protein_language_model import tensors as plm_t
from problems.medical.photon_counting_ct import equation as ct_eq
from problems.medical.photon_counting_ct import solve as ct_solve
from problems.medical.photon_counting_ct import tensors as ct_t

CPU = torch.device("cpu")
THRESHOLDS = [25.0, 35.0, 50.0, 70.0]

# --- photon-counting CT ---


def _detector(kvp: float = 120.0, sigma: float = 3.0):
    e = ct_t.energies(kvp, CPU)
    return e, ct_t.basis_attenuation(e), ct_t.spectrum(kvp, e), ct_t.bin_response(e, THRESHOLDS, sigma)


def test_attenuation_table_has_iodine_k_edge():
    e, mu, _, _ = _detector()
    below, above = mu[e == 33.0, 2], mu[e == 34.0, 2]
    assert above > 4 * below  # K-edge jump at 33.2 keV
    assert torch.all(torch.diff(mu[:, 0]) < 0)  # water falls monotonically (no edge in range)


def test_bin_response_partitions_counted_photons():
    e, _, _, R = _detector(sigma=2.0)
    total = R.sum(0)  # probability of being counted at all
    assert torch.allclose(total[e >= 40], torch.ones_like(total[e >= 40]), atol=1e-6)
    assert total[e == 20].item() < 0.01  # below the lowest threshold: rejected
    # ideal detector: a photon lands in exactly the bin that contains it
    R0 = ct_t.bin_response(e, THRESHOLDS, 1e-3)
    assert R0[1, e == 40].item() > 0.999 and R0[2, e == 60].item() > 0.999


def test_beer_lambert_per_bin_matches_explicit_sum():
    e, mu, s, R = _detector()
    A = torch.tensor([[20.0, 0.0, 0.0], [15.0, 1.0, 30.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    lam = ct_eq.expected_counts(1e5, s, R, mu, A)
    for r in range(A.shape[0]):
        for b in range(R.shape[0]):
            explicit = sum(
                1e5 * R[b, k] * s[k] * torch.exp(-(mu[k] * A[r]).sum()) for k in range(len(e))
            )  # Σ_e N0 R_be S_e exp(−Σ_m μ_em A_m)
            assert torch.isclose(lam[r, b], explicit, rtol=1e-10)
    # zero thickness: open-beam counts; monochromatic limit: exactly exp(−μ t)
    assert torch.allclose(lam[2], 1e5 * R @ s)
    k = int((e == 60).nonzero())
    mono = torch.zeros_like(s)
    mono[k] = 1.0
    t = ct_eq.expected_counts(1.0, mono, R, mu, A[:1]).sum() / (R[:, k].sum())
    assert torch.isclose(t, torch.exp(-mu[k, 0] * 20.0), rtol=1e-12)


def test_decomposition_recovers_thicknesses_noise_free():
    e, mu, s, R = _detector()
    gen = torch.Generator().manual_seed(0)
    A = torch.rand(200, 3, generator=gen, dtype=torch.float64) * torch.tensor([25.0, 2.0, 60.0], dtype=torch.float64)
    lam = ct_eq.expected_counts(1e9, s, R, mu, A)
    A_hat = ct_solve.decompose(lam, 1e9, s, R, mu)
    assert (A_hat - A).abs().max() < 1e-8


def test_fisher_gradient_matches_autograd_curvature():
    e, mu, s, R = _detector()
    A = torch.tensor([[18.0, 0.5, 20.0]], dtype=torch.float64)
    lam = ct_eq.expected_counts(1e5, s, R, mu, A)
    F = ct_eq.fisher(ct_eq.jacobian(1e5, s, R, mu, A), lam)[0]
    # with y = λ (expected data) the Hessian of the NLL equals the Fisher information
    H = torch.autograd.functional.hessian(
        lambda a: ct_eq.poisson_nll(lam, ct_eq.expected_counts(1e5, s, R, mu, a[None])).sum(), A[0]
    )
    assert torch.allclose(F, H, rtol=1e-8)


def test_iodine_error_falls_with_dose_and_separates_bone():
    e, mu, s, R = _detector()
    w = ct_t.matched_bone_fraction(mu, s, e, ct_t.MATCHED_IODINE)
    x = ct_t.grid(48, CPU)
    maps, rois, _ = ct_t.phantom(x, w)
    theta = ct_t.angles(60, CPU)
    A = ct_solve.basis_sinograms(maps, theta, x)
    lam_unit = ct_eq.expected_counts(1.0, s, R, mu, A.reshape(-1, 3))
    errors = ct_solve.iodine_error_vs_dose(lam_unit, A.shape, s, R, mu, theta, x, maps, rois, [1e4, 1e5, 1e6, 1e7])
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[0] / errors[-1] > 10  # ≈ √1000 ≈ 32 for pure Poisson noise
    # conventional CT: iodine-10 and bone inserts look alike; the iodine map tells them apart
    conv = ct_eq.energy_integrating(1.0, s, e, mu, A.reshape(-1, 3)).reshape(60, 48)
    _, img = ct_solve.reconstruct(torch.cat([A.permute(2, 0, 1), conv[None]]), theta, x)
    conv_means, iodine_means = ct_solve.roi_means(img[3], rois), ct_solve.roi_means(img[2], rois)
    assert abs(conv_means[2] - conv_means[4]) / conv_means[2] < 0.03
    assert abs(iodine_means[2] - 10.0) < 1.0 and abs(iodine_means[4]) < 1.0


# --- protein language model ---


def _family(n_seq=1500, sweeps=60, coupling=1.2):
    coords = plm_t.lattice_fold(5, 6, CPU)
    contacts = plm_t.contact_map(coords)
    h, J = plm_t.planted_potts(contacts, 8, coupling, 0.5, seed=0)
    seqs = plm_solve.gibbs_sample(h, J, plm_t.parity(coords), n_seq, sweeps, seed=0)
    return contacts, seqs


def test_lattice_fold_contacts_and_parity():
    coords = plm_t.lattice_fold(5, 6, CPU)
    steps = (coords[1:] - coords[:-1]).abs().sum(-1)
    assert torch.all(steps == 1)  # a connected chain
    contacts = plm_t.contact_map(coords)
    assert torch.equal(contacts, contacts.T)
    colour = plm_t.parity(coords)
    i, j = torch.nonzero(contacts, as_tuple=True)
    assert torch.all(colour[i] != colour[j])  # contacts join opposite colours: checkerboard Gibbs is valid


def test_attention_rows_sum_to_one():
    w = plm_t.init_weights(30, 8, 32, 4, 2, seed=0, device=CPU)
    tokens = torch.randint(9, (5, 30), generator=torch.Generator().manual_seed(0))
    logits, maps = plm_eq.forward(tokens, w)
    assert logits.shape == (5, 30, 8) and maps.shape == (5, 2, 4, 30, 30)
    assert torch.allclose(maps.sum(-1), torch.ones(5, 2, 4, 30), atol=1e-5)
    assert torch.all(maps >= 0)


def test_apc_rows_and_columns_average_out():
    F = torch.rand(12, 12, generator=torch.Generator().manual_seed(3))
    F = F + F.T
    G = plm_eq.apc(F)
    assert torch.allclose(G, G.T, atol=1e-6)
    assert G.mean().abs() < 1e-6  # the overall background is removed


def test_masked_lm_beats_baselines_and_attention_finds_contacts():
    torch.manual_seed(0)
    contacts, seqs = _family(n_seq=2300)
    train, test = seqs[:2000], seqs[2000:]
    ce_uni, ce_prof = plm_solve.baselines(train, test, 8)
    assert ce_prof < ce_uni
    w = plm_t.init_weights(30, 8, 32, 4, 2, seed=0, device=CPU)
    hist = plm_solve.train(train, test, contacts, w, 8, 500, 128, 3e-3, 0.15, n_snapshots=5)
    assert hist["test"][-1] < hist["test"][0]
    assert hist["test"][-1] < ce_uni  # below the unigram baseline
    assert hist["test"][-1] < ce_prof  # and even below the per-site profile: the model uses context
    rand = plm_solve.random_precision(contacts)
    assert hist["precision"][-1] > 5 * rand  # contacts from attention far above chance
