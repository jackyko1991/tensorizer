r"""Physics → tensor translation.

Einstein (1905): light of frequency ν arrives in quanta of energy hν. One quantum is absorbed by one electron, which
spends the work function φ (plus its depth ε ≥ 0 below the top of the occupied band) to leave the metal:

    K_{fn} = h\nu_f - \phi - \varepsilon_{fn}          (f: frequency, n: electron — a broadcast, no loop)

The fastest electrons (ε = 0) are stopped by a retarding potential V_s with

    e V_s = h\nu - \phi ,

a straight line in ν with slope h/e and threshold ν_0 = φ/h, whatever the intensity. With ε uniform on [0, w] the
current that still reaches the collector at retarding voltage V is

    I(V) = \Phi\,\operatorname{clip}\big((h\nu - \phi - eV)/w,\ 0,\ 1\big)      (V ≥ 0; Φ: photon flux × yield)

Units: energies in eV, potentials in V, frequencies in 10^14 Hz, so h/e = 0.41357 V per 10^14 Hz.

The classical wave picture instead gives the electron energy at a rate S·A (irradiance × atomic area), so it must
wait t = φ/(S A) before it can leave, and the kinetic energy grows with the intensity, not with ν.
"""

import torch

H_PLANCK = 6.62607015e-34  # J s (exact, SI 2019)
E_CHARGE = 1.602176634e-19  # C (exact)
H_OVER_E = H_PLANCK / E_CHARGE * 1e14  # V per 10^14 Hz ≈ 0.41357


def photon_energy(nu: torch.Tensor) -> torch.Tensor:
    """hν in eV for ν in 10^14 Hz: any shape."""
    return H_OVER_E * nu


def threshold_frequency(phi: float) -> float:
    """ν0 = φ/h in 10^14 Hz."""
    return phi / H_OVER_E


def kinetic_energy(nu: torch.Tensor, phi: float, depth: torch.Tensor) -> torch.Tensor:
    r"""K_{fn} = hν_f − φ − ε_{fn} (eV).   nu: (F,), depth: (..., F, N) → (..., F, N); K ≤ 0: no emission."""
    return photon_energy(nu)[:, None] - phi - depth


def stopping_potential(nu: torch.Tensor, phi: float) -> torch.Tensor:
    """V_s = (hν − φ)/e, zero below threshold: (F,)."""
    return torch.clamp(photon_energy(nu) - phi, min=0.0)


def photocurrent(V: torch.Tensor, nu: torch.Tensor, phi: float, flux: float, width: float) -> torch.Tensor:
    """Expected current at retarding voltage V (broadcast V against nu), depth ε uniform on [0, width].

    An accelerating voltage (V < 0) collects every emitted electron but cannot create any below threshold."""
    return flux * torch.clamp((photon_energy(nu) - phi - torch.clamp(V, min=0.0)) / width, 0.0, 1.0)


def classical_delay(S: float, phi: float, area: float) -> float:
    """Wave picture: seconds an electron needs to soak up φ (eV) from irradiance S (W/m²) over an area (m²)."""
    return phi * E_CHARGE / (S * area)


def classical_kinetic_energy(intensity: torch.Tensor, K1: float) -> torch.Tensor:
    """Wave picture: the electron is shaken by the field E0, so K ∝ E0² ∝ intensity (scaled so K(1) = K1)."""
    return K1 * intensity
