"""Build the concrete tensors: the altitude grid, an exponential atmosphere, temperatures and photolysis rates.

Units: cm⁻³ for number densities, s for time, km for altitude.

Photolysis is modelled with one effective absorption band per molecule, attenuated by the column overhead along
the slanted sun path (Beer–Lambert). These band values are illustrative choices that give photolysis rates of
roughly the right size (j_O₂ ~ 10⁻¹⁰ s⁻¹ at 50 km falling to ~ 10⁻¹² s⁻¹ near 25 km; j_O₃ ~ 10⁻² s⁻¹ at the top
falling to ~ 4×10⁻⁴ s⁻¹ below the ozone layer); a real calculation integrates over the solar spectrum.
"""

import math

import torch

N0 = 2.55e19  # air number density at sea level, cm⁻³
H_SCALE = 7.0  # scale height, km
X_O2 = 0.21  # O₂ mole fraction

J1_TOP, SIGMA_O2 = 2e-10, 5e-23  # O₂ band: unattenuated rate s⁻¹, effective cross-section cm² (illustrative)
J3_UV, SIGMA_O3 = 8e-3, 1e-17  # O₃ Hartley band (cross-section ≈ 1.1×10⁻¹⁷ cm² at the 255 nm peak)
J3_VIS = 4e-4  # O₃ Chappuis band (visible light, practically unattenuated), s⁻¹


def altitudes(n: int, device: torch.device) -> torch.Tensor:
    """Levels from 15 to 60 km.   (A,)"""
    return torch.linspace(15.0, 60.0, n, dtype=torch.float64, device=device)


def air_density(z: torch.Tensor) -> torch.Tensor:
    """Exponential atmosphere n(z) = n₀ exp(-z / H), cm⁻³.   (A,)"""
    return N0 * torch.exp(-z / H_SCALE)


def temperature(z: torch.Tensor) -> torch.Tensor:
    """US Standard Atmosphere 1976 temperature profile (piecewise linear in z), K.   (A,)"""
    z_knots = torch.tensor([0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 85.0], dtype=z.dtype, device=z.device)
    T_knots = torch.tensor(
        [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95], dtype=z.dtype, device=z.device
    )
    i = (torch.searchsorted(z_knots, z) - 1).clamp(0, len(z_knots) - 2)
    w = (z - z_knots[i]) / (z_knots[i + 1] - z_knots[i])
    return T_knots[i] + w * (T_knots[i + 1] - T_knots[i])


def column_above(n: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Number of molecules per cm² above each level: a sum over the levels above plus half the own layer,
    plus the part above the top level, n_top H (exponential tail).   n, z (A,) → (A,)"""
    dz = (z[1] - z[0]) * 1e5  # km → cm
    above = n.flip(0).cumsum(0).flip(0) - 0.5 * n  # levels at and above, own level half
    return above * dz + n[-1] * H_SCALE * 1e5


def photolysis_O2(z: torch.Tensor, mu: float) -> torch.Tensor:
    """j₁ = J1_TOP exp(-σ_O₂ N_O₂ / μ): O₂ shields itself.   (A,)"""
    N_O2 = X_O2 * air_density(z) * H_SCALE * 1e5  # exact O₂ column above z for an exponential atmosphere
    return J1_TOP * torch.exp(-SIGMA_O2 * N_O2 / mu)


def photolysis_O3(O3: torch.Tensor, z: torch.Tensor, mu: float) -> torch.Tensor:
    """j₃ = J3_VIS + J3_UV exp(-σ_O₃ N_O₃ / μ): the ozone overhead shields the UV part.   (A,)"""
    return J3_VIS + J3_UV * torch.exp(-SIGMA_O3 * column_above(O3, z) / mu)


def initial_state(z: torch.Tensor) -> torch.Tensor:
    """Sunrise on an atmosphere with all its oxygen as O₂: c = ([O], [O₂], [O₃]) = (0, 0.21 n, 0).   (A, 3)"""
    c = torch.zeros(len(z), 3, dtype=torch.float64, device=z.device)
    c[:, 1] = X_O2 * air_density(z)
    return c


def time_grid(t_end: float, n: int) -> torch.Tensor:
    """Logarithmically spaced times from 10⁻⁴ s to t_end, with t = 0 first.   (n + 1,)"""
    return torch.cat(
        [torch.zeros(1, dtype=torch.float64), torch.logspace(-4, math.log10(t_end), n, dtype=torch.float64)]
    )
