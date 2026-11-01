"""Main loop: parameters → tensors → photon counts → material decomposition → reconstruction → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors

THRESHOLDS = {"25,35,50,70": [25.0, 35.0, 50.0, 70.0], "25,50,75": [25.0, 50.0, 75.0]}


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    kvp, n0 = params["kvp"], params["dose"]
    e = tensors.energies(kvp, dev)  # (E,) keV
    mu = tensors.basis_attenuation(e)  # (E, M)
    s = tensors.spectrum(kvp, e)  # (E,)
    R = tensors.bin_response(e, THRESHOLDS[params["thresholds"]], params["sigma_kev"])  # (B, E)
    w_bone = tensors.matched_bone_fraction(mu, s, e, tensors.MATCHED_IODINE)

    n = 64 if fast else params["n_pixels"]
    n_angles = 60 if fast else params["n_angles"]
    x = tensors.grid(n, dev)  # (N,)
    maps, rois, labels = tensors.phantom(x, w_bone)  # (M, N, N), (5, N, N)
    theta = tensors.angles(n_angles, dev)  # (A,)

    # Scan: basis line integrals → expected counts per bin → Poisson counts
    A_true = solve.basis_sinograms(maps, theta, x)  # (A, S, M)
    rays = A_true.reshape(-1, 3)  # (R, M), R = A·S
    lam_unit = equation.expected_counts(1.0, s, R, mu, rays)  # (R, B) counts per open-beam photon
    y = solve.count_photons(lam_unit * n0, seed=0)  # (R, B)
    print(f"{kvp:.0f} kVp, {R.shape[0]} bins, thresholds {params['thresholds']} keV, σ = {params['sigma_kev']} keV")
    print(f"open beam per bin: {', '.join(f'{v:.0f}' for v in (n0 * R @ s).tolist())} photons per ray")
    print(f"thickest ray: {y.min(0).values.int().tolist()} counts per bin (low bin starves first)")
    print(
        f"bone insert: {100 * w_bone:.1f}% cortical bone in water, matched to {tensors.MATCHED_IODINE:g} mg/mL iodine"
    )

    # Decompose every ray (one batched ML fit), then reconstruct each basis sinogram by FBP
    A_hat = solve.decompose(y, n0, s, R, mu).reshape(A_true.shape)  # (A, S, M)
    noise_free = solve.decompose(lam_unit * 1e9, 1e9, s, R, mu, n_iter=8)
    print(f"decomposition of noise-free data: max error {(noise_free - rays).abs().max():.1e}")

    conventional = equation.energy_integrating(n0, s, e, mu, rays).reshape(n_angles, n)  # (A, S) −ln(I/I0)
    sinos = torch.cat([A_hat.permute(2, 0, 1), conventional[None]])  # (4, A, S)
    Q, images = solve.reconstruct(sinos, theta, x)  # (4, A, N, N), (4, N, N): water, bone, iodine, conventional

    print("ROI means      " + "  ".join(f"{lab:>7}" for lab in labels))
    for name, k in (("iodine mg/mL", 2), ("bone %", 1), ("conv μ 1/cm", 3)):
        scale = 100.0 if k == 1 else 1.0
        print(f"{name:13} " + "  ".join(f"{v * scale:7.2f}" for v in solve.roi_means(images[k], rois).tolist()))

    vmi_e = torch.arange(40.0, kvp - 9, 4.0, device=dev, dtype=e.dtype)  # keV energies for the VMI sweep
    vmi_mu = tensors.basis_attenuation(vmi_e)  # (K, M)
    vmi = equation.hounsfield(equation.monoenergetic(vmi_mu, images[:3]), vmi_mu[:, 0, None, None])  # (K, N, N)
    vmi_rois = torch.stack([solve.roi_means(v, rois) for v in vmi])  # (K, 5)
    i40, i70 = 0, int(torch.argmin((vmi_e - 70).abs()))
    print(f"VMI 40 keV: I10 {vmi_rois[i40, 2]:.0f} HU, bone {vmi_rois[i40, 4]:.0f} HU;  ", end="")
    print(f"70 keV: I10 {vmi_rois[i70, 2]:.0f} HU, bone {vmi_rois[i70, 4]:.0f} HU")

    doses = [n0 * f for f in ([0.25, 1.0] if fast else [0.1, 0.3, 1.0, 3.0, 10.0])]
    errors = solve.iodine_error_vs_dose(lam_unit, A_true.shape, s, R, mu, theta, x, maps, rois, doses)
    for d, err in zip(doses, errors):
        print(f"  N0 = {d:9.3g} photons/ray: iodine RMS error {err:.2f} mg/mL")

    detected = R * s  # (B, E) open-beam spectrum seen by each bin
    return plot.animate(
        x, theta, e, detected, mu, Q, images, maps, vmi_e, vmi, vmi_rois, labels, doses, errors,
        out_dir / "photon_counting_ct.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
