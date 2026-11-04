"""Build the concrete tensors: load the sky map, random rotations, isotropic random skies and a planted sky.

data/wmap_ilc9_nside16.csv is the WMAP 9-year Internal Linear Combination (ILC) temperature map (Bennett et al.
2013), degraded from HEALPix N_side = 512 to N_side = 16 (3072 equal-area pixels) by averaging the 1024 sub-pixels
of each pixel: Galactic longitude, latitude (degrees) and temperature (μK). Low resolution is plenty for ℓ = 2, 3.
"""

import math
from pathlib import Path

import torch

from . import equation

DATA = Path(__file__).parent / "data" / "wmap_ilc9_nside16.csv"
DTYPE = torch.float64
COLUMNS = {  # accepted header spellings (case-insensitive)
    "lon": ("glon_deg", "lon", "glon", "l", "longitude", "ra"),
    "lat": ("glat_deg", "lat", "glat", "b", "latitude", "dec"),
    "T": ("t_uk", "t", "temperature", "temp", "i", "i_stokes"),
    "w": ("weight", "w", "area_sr"),
}


def unit_vectors(lon_deg: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
    """(lon, lat) in degrees → unit vectors (P, 3); x towards lon = 0, z towards lat = +90°."""
    lon, lat = torch.deg2rad(lon_deg), torch.deg2rad(lat_deg)
    return torch.stack([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], -1)


def lonlat(n: torch.Tensor) -> tuple[float, float]:
    """Unit vector → (longitude in [0, 360), latitude) in degrees."""
    lon = math.degrees(math.atan2(n[1].item(), n[0].item())) % 360
    return lon, math.degrees(math.asin(max(-1.0, min(1.0, n[2].item()))))


def load_sky_map(path: Path = DATA, device: torch.device | None = None):
    """Read a sky-map CSV: pixel directions (P, 3), temperatures (P,) and solid-angle weights (P,).

    Any comma-separated file with a header naming longitude, latitude (degrees) and temperature columns works
    (see COLUMNS; order free, extra columns ignored, # lines skipped), so an uploaded map can replace the bundled
    WMAP one. Without a weight column the pixels are taken as equal-area (HEALPix), each 4π/P steradians.
    """
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and ln[0] != "#"]
    header = [h.strip().lower() for h in lines[0].split(",")]
    index = {}
    for key, names in COLUMNS.items():
        found = [header.index(n) for n in names if n in header]
        if found:
            index[key] = found[0]
        elif key != "w":
            raise ValueError(f"no {key} column in {path}: expected one of {names}, got {header}")
    rows = [ln.split(",") for ln in lines[1:]]
    keys = [k for k in ("lon", "lat", "T", "w") if k in index]
    table = torch.tensor([[float(r[index[k]]) for k in keys] for r in rows], dtype=DTYPE, device=device)
    dirs = unit_vectors(table[:, 0], table[:, 1])
    w = table[:, 3] if "w" in index else torch.full_like(table[:, 0], 4 * math.pi / len(table))
    return dirs, table[:, 2], w


def random_rotation(generator: torch.Generator) -> torch.Tensor:
    """Uniformly random 3×3 rotation (QR of a Gaussian matrix with the sign fixed, det = +1)."""
    q, r = torch.linalg.qr(torch.randn(3, 3, dtype=DTYPE, generator=generator))
    q = q * torch.sign(torch.diagonal(r))
    return q if torch.det(q) > 0 else -q


def isotropic_coefficients(n_skies: int, ell: int, generator: torch.Generator) -> torch.Tensor:
    """Complex a_{ℓm} of n_skies isotropic Gaussian skies: (K, 2ℓ+1).

    Independent unit Gaussians on the real harmonic coefficients are exactly isotropic; the overall power C_ℓ
    does not matter because the axis only depends on the shape of a_m.
    """
    r = torch.randn(n_skies, 2 * ell + 1, dtype=DTYPE, generator=generator)
    return equation.real_to_complex(r, ell)


def planted_sky(dirs: torch.Tensor, axis: torch.Tensor, amplitudes=(30.0, 25.0)) -> torch.Tensor:
    r"""A sky whose quadrupole and octopole are both pure m = ±ℓ around the same axis: (P,).

    Re[(\hat n\cdot(e_1 + i e_2))^ℓ] with e_1, e_2 ⊥ axis is a sectoral harmonic: all its power is in m = ±ℓ.
    """
    helper = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE) if abs(axis[0]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
    e1 = torch.nn.functional.normalize(torch.linalg.cross(axis, helper.to(axis)), dim=0)
    e2 = torch.linalg.cross(axis, e1)
    zeta = dirs @ e1 + 1j * (dirs @ e2)  # (P,) complex
    return amplitudes[0] * (zeta**2).real + amplitudes[1] * (zeta**3).real
