"""Physics checks, written first (TDD): what any correct photon-counting CT code must satisfy.

Run with `pytest problems/medical/photon_counting_ct/test.py`. Each test is a physical law: bin counts follow the
polychromatic Beer–Lambert law, the detector response partitions the counted photons, iodine has its K-edge at 33.2
keV, and noise falls with dose as Poisson statistics predict.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
THRESHOLDS = [25.0, 35.0, 50.0, 70.0]


def _detector(kvp: float = 120.0, sigma: float = 3.0):
    e = tensors.energies(kvp, CPU)
    return e, tensors.basis_attenuation(e), tensors.spectrum(kvp, e), tensors.bin_response(e, THRESHOLDS, sigma)


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
    R0 = tensors.bin_response(e, THRESHOLDS, 1e-3)
    assert R0[1, e == 40].item() > 0.999 and R0[2, e == 60].item() > 0.999


def test_beer_lambert_per_bin_matches_explicit_sum():
    e, mu, s, R = _detector()
    A = torch.tensor([[20.0, 0.0, 0.0], [15.0, 1.0, 30.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    lam = equation.expected_counts(1e5, s, R, mu, A)
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
    t = equation.expected_counts(1.0, mono, R, mu, A[:1]).sum() / (R[:, k].sum())
    assert torch.isclose(t, torch.exp(-mu[k, 0] * 20.0), rtol=1e-12)


def test_decomposition_recovers_thicknesses_noise_free():
    e, mu, s, R = _detector()
    gen = torch.Generator().manual_seed(0)
    A = torch.rand(200, 3, generator=gen, dtype=torch.float64) * torch.tensor([25.0, 2.0, 60.0], dtype=torch.float64)
    lam = equation.expected_counts(1e9, s, R, mu, A)
    A_hat = solve.decompose(lam, 1e9, s, R, mu)
    assert (A_hat - A).abs().max() < 1e-8


def test_fisher_gradient_matches_autograd_curvature():
    e, mu, s, R = _detector()
    A = torch.tensor([[18.0, 0.5, 20.0]], dtype=torch.float64)
    lam = equation.expected_counts(1e5, s, R, mu, A)
    F = equation.fisher(equation.jacobian(1e5, s, R, mu, A), lam)[0]
    # with y = λ (expected data) the Hessian of the NLL equals the Fisher information
    H = torch.autograd.functional.hessian(
        lambda a: equation.poisson_nll(lam, equation.expected_counts(1e5, s, R, mu, a[None])).sum(), A[0]
    )
    assert torch.allclose(F, H, rtol=1e-8)


def test_iodine_error_falls_with_dose_and_separates_bone():
    e, mu, s, R = _detector()
    w = tensors.matched_bone_fraction(mu, s, e, tensors.MATCHED_IODINE)
    x = tensors.grid(48, CPU)
    maps, rois, _ = tensors.phantom(x, w)
    theta = tensors.angles(60, CPU)
    A = solve.basis_sinograms(maps, theta, x)
    lam_unit = equation.expected_counts(1.0, s, R, mu, A.reshape(-1, 3))
    errors = solve.iodine_error_vs_dose(lam_unit, A.shape, s, R, mu, theta, x, maps, rois, [1e4, 1e5, 1e6, 1e7])
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[0] / errors[-1] > 10  # ≈ √1000 ≈ 32 for pure Poisson noise
    # conventional CT: iodine-10 and bone inserts look alike; the iodine map tells them apart
    conv = equation.energy_integrating(1.0, s, e, mu, A.reshape(-1, 3)).reshape(60, 48)
    _, img = solve.reconstruct(torch.cat([A.permute(2, 0, 1), conv[None]]), theta, x)
    conv_means, iodine_means = solve.roi_means(img[3], rois), solve.roi_means(img[2], rois)
    assert abs(conv_means[2] - conv_means[4]) / conv_means[2] < 0.03
    assert abs(iodine_means[2] - 10.0) < 1.0 and abs(iodine_means[4]) < 1.0
