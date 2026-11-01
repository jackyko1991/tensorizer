"""Build the concrete tensors: attenuation table μ_{em}, the phantom masks M_{myx} and the tube spectrum S_e.

All attenuation coefficients are rounded values in 1/cm (derived from NIST XCOM mass attenuation
coefficients × typical densities). They are good enough to see the trends, not for clinical use.
"""

import torch

ENERGIES_KEV = [30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0]
MATERIALS = ["air", "lung", "soft tissue", "bone", "iodine contrast"]

# Linear attenuation coefficients μ (1/cm) at the energies above.
MU_TABLE = {
    "air": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ~2e-4 1/cm: negligible
    "lung": [0.100, 0.072, 0.061, 0.055, 0.049, 0.046, 0.043, 0.040],  # inflated lung, ρ ≈ 0.26 g/cm³
    "soft tissue": [0.398, 0.284, 0.241, 0.218, 0.195, 0.181, 0.171, 0.160],  # ρ ≈ 1.06 g/cm³
    "bone": [2.56, 1.28, 0.81, 0.60, 0.43, 0.36, 0.32, 0.28],  # cortical bone, ρ ≈ 1.92 g/cm³
}
# Mass attenuation coefficient of iodine (cm²/g); the K-edge at 33.2 keV makes it jump between 30 and 40 keV.
IODINE_MASS_MU = [8.7, 22.1, 12.3, 7.6, 3.5, 1.94, 1.2, 0.69]
# Aluminium tube filter (1/cm), removes the softest photons from the spectrum.
ALUMINIUM_MU = [3.05, 1.53, 0.99, 0.75, 0.545, 0.46, 0.42, 0.372]


def energies(device: torch.device) -> torch.Tensor:
    """Photon energies E_e in keV: shape (E,)."""
    return torch.tensor(ENERGIES_KEV, device=device)


def attenuation_table(iodine_mg_ml: float, device: torch.device) -> torch.Tensor:
    """μ_{em} of shape (E, M) in 1/cm. The contrast agent is soft tissue (blood) plus dissolved iodine."""
    columns = [MU_TABLE[m] for m in MATERIALS[:-1]]
    iodine = torch.tensor(IODINE_MASS_MU) * (iodine_mg_ml / 1000.0)  # (cm²/g)·(g/cm³) = 1/cm
    columns.append((torch.tensor(MU_TABLE["soft tissue"]) + iodine).tolist())
    return torch.tensor(columns, device=device).T  # (E, M)


def interpolate_loglog(e_table: torch.Tensor, mu: torch.Tensor, energy: float) -> torch.Tensor:
    """μ_m at one energy, interpolated linearly in log(μ) vs log(E) (μ = 0 stays 0): (E, M) → (M,)."""
    log_e = torch.log(e_table)
    target = torch.log(torch.tensor(float(energy), device=e_table.device))
    i = int(torch.searchsorted(log_e, target).clamp(1, len(e_table) - 1))  # table interval [i-1, i]
    w = (target - log_e[i - 1]) / (log_e[i] - log_e[i - 1])
    lo, hi = mu[i - 1].clamp_min(1e-12), mu[i].clamp_min(1e-12)
    out = torch.exp((1 - w) * torch.log(lo) + w * torch.log(hi))
    return torch.where(mu[i] > 0, out, torch.zeros_like(out))


def phantom(n_pixels: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A crude chest slice, 32 cm wide and 24 cm deep.

    Returns
      x (X,), y (Y,) pixel centres in cm, and masks M_{myx} of shape (M, Y, X): one-hot material per pixel.
    The beam travels along y (front to back), so each column x is one ray.
    """
    nx, ny = n_pixels, (3 * n_pixels) // 4
    x = (torch.arange(nx, device=device) + 0.5) * (32.0 / nx) - 16.0
    y = (torch.arange(ny, device=device) + 0.5) * (24.0 / ny) - 12.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (Y, X)

    def ellipse(cx, cy, ax, ay):
        return ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0

    label = torch.zeros(ny, nx, dtype=torch.long, device=device)  # 0 = air
    label[ellipse(0.0, 0.0, 15.0, 10.0)] = 2  # body: soft tissue
    label[ellipse(-6.5, 0.5, 3.8, 6.5)] = 1  # lungs
    label[ellipse(6.5, 0.5, 3.8, 6.5)] = 1
    label[ellipse(-1.0, 2.0, 3.0, 3.0)] = 4  # heart filled with iodinated blood
    label[ellipse(0.0, -6.5, 1.8, 1.8)] = 3  # spine
    label[ellipse(0.0, 8.6, 1.6, 0.6)] = 3  # sternum
    masks = torch.nn.functional.one_hot(label, len(MATERIALS)).permute(2, 0, 1).float()  # (M, Y, X)
    return x, y, masks


def path_weights(y: torch.Tensor) -> torch.Tensor:
    """Δl_y: length of the ray inside each pixel row (cm). Straight vertical rays → the pixel height: (Y,)."""
    return torch.full_like(y, float(y[1] - y[0]))


def tube_spectrum(kvp: float, e: torch.Tensor, filter_cm: float = 0.25) -> torch.Tensor:
    """S_e: Kramers' law (kVp − E) for the bremsstrahlung, × an aluminium filter, × bin width; sums to 1."""
    kramers = (kvp - e).clamp_min(0.0)
    filtered = kramers * torch.exp(-torch.tensor(ALUMINIUM_MU, device=e.device) * filter_cm)
    weights = filtered * torch.gradient(e)[0]  # bins are not equally wide
    return weights / weights.sum()
