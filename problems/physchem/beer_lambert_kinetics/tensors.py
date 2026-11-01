"""Build the concrete tensors: wavelength grid, molar absorptivities ε_ws, rate matrix, sampling times.

The spectra are synthetic: each species gets a Gaussian absorption band with a textbook-sized molar absorptivity
(10⁴ L mol⁻¹ cm⁻¹ is typical for a strongly coloured organic dye or a metal complex).
"""

import torch

DTYPE = torch.float64
SPECIES = ("A", "B", "C")
PATH_LENGTH = 1.0  # cuvette path length l (cm)
N_SPECTRA = 121  # spectra recorded during the run

# synthetic bands: peak wavelength (nm), width σ (nm), peak ε (L mol⁻¹ cm⁻¹)
BANDS = {
    "A": [(350.0, 30.0, 12000.0)],
    "B": [(450.0, 40.0, 9000.0)],
    "C": [(560.0, 45.0, 15000.0)],
}


def wavelengths(device: torch.device) -> torch.Tensor:
    """λ_w in nm, shape (W,): 300–700 nm every 2 nm."""
    return torch.linspace(300.0, 700.0, 201, device=device, dtype=DTYPE)


def absorptivities(lam: torch.Tensor) -> torch.Tensor:
    """E_ws = ε_s(λ_w) in L mol⁻¹ cm⁻¹, shape (W, S): one column per species, a sum of Gaussian bands."""
    columns = []
    for name in SPECIES:
        eps = torch.zeros_like(lam)
        for peak, width, height in BANDS[name]:
            eps = eps + height * torch.exp(-0.5 * ((lam - peak) / width) ** 2)
        columns.append(eps)
    return torch.stack(columns, dim=1)


def initial_concentrations(c0_uM: float, device: torch.device) -> torch.Tensor:
    """c_s(0) in mol/L, shape (S,): only A at the start."""
    return torch.tensor([c0_uM * 1e-6, 0.0, 0.0], device=device, dtype=DTYPE)


def sample_times(t_end: float, device: torch.device) -> torch.Tensor:
    """t_t in s, shape (T,): when the spectrophotometer takes a spectrum."""
    return torch.linspace(0.0, t_end, N_SPECTRA, device=device, dtype=DTYPE)


def noise(shape: tuple[int, ...], sigma: float, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Gaussian detector noise in absorbance units, with a fixed seed (the same numbers on every device)."""
    gen = torch.Generator().manual_seed(seed)
    return sigma * torch.randn(shape, generator=gen, dtype=DTYPE).to(device)
